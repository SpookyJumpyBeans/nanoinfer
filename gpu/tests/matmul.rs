//! The GPU int8 matmul against the CPU kernel that is already verified.
//!
//! Not bitwise: the shader sums 64 strided slices and folds them in a tree,
//! the CPU sums four chains of eight lanes. Same products, different order,
//! so the comparison is to float32 tolerance -- set by the arithmetic, not by
//! what happened to pass.
//!
//! On a machine with no WebGPU adapter at all the tests say so and pass,
//! rather than failing a build that has nothing to run them on.

use std::sync::OnceLock;

use nanoinfer_gpu::Gpu;
use nanoinfer_kernels::{matmul_i8_auto, pseudo_random, quantize_rows_i8};

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| match Gpu::new() {
        Ok(gpu) => {
            eprintln!("adapter: {}", gpu.describe());
            Some(gpu)
        }
        Err(e) => {
            eprintln!("skipping GPU tests: {e}");
            None
        }
    })
    .as_ref()
}

/// GPU result against the CPU kernel, which returns [rows, tokens] transposed.
fn check(rows: usize, in_features: usize, tokens: usize, seed: u64) {
    let Some(gpu) = gpu() else { return };

    let (quantized, scales) = quantize_rows_i8(&pseudo_random(rows * in_features, seed), rows);
    let x = pseudo_random(tokens * in_features, seed + 1);

    let mut cpu_t = vec![0.0; rows * tokens];
    matmul_i8_auto(&quantized, &scales, &x, tokens, &mut cpu_t);

    let matrix = gpu.upload(&quantized, &scales, in_features);
    let out = gpu.matmul(&matrix, &x, tokens);
    assert_eq!(out.len(), tokens * rows);

    for t in 0..tokens {
        for r in 0..rows {
            let (g, c) = (out[t * rows + r], cpu_t[r * tokens + t]);
            let tolerance = 1e-4 * c.abs().max(1.0) * (in_features as f32).sqrt().max(1.0) / 8.0;
            assert!(
                (g - c).abs() <= tolerance,
                "{rows}x{in_features}, token {t}, row {r}: gpu {g} vs cpu {c}"
            );
        }
    }
}

#[test]
fn matches_the_cpu_kernel_on_every_decode_shape() {
    for &(rows, in_features) in &[(896, 896), (128, 896), (4864, 896), (896, 4864)] {
        check(rows, in_features, 1, 21);
    }
}

#[test]
fn matches_on_a_batch_of_tokens() {
    check(4864, 896, 7, 22);
}

#[test]
fn pads_widths_that_are_not_a_multiple_of_four() {
    // 1, 2 and 3 leftover int8 in the last word, and fewer words than lanes.
    for in_features in [1, 5, 6, 7, 33, 63, 65, 130] {
        check(9, in_features, 2, 23 + in_features as u64);
    }
}

#[test]
fn rows_past_the_dispatch_limit_wrap_correctly() {
    // 70000 rows: past the 65535-workgroup cap on one dimension, which the
    // 151936-row LM head needs. A wrong wrap leaves rows unwritten (zero).
    check(70_000, 8, 1, 31);
}

#[test]
fn a_matrix_split_into_chunks_lands_every_row_in_place() {
    // The LM head does not fit one binding on every device, so large
    // matrices are uploaded as row chunks. Force 7 uneven chunks here and
    // compare against one upload of the same matrix.
    let Some(gpu) = gpu() else { return };
    let (rows, in_features, tokens) = (1000, 40, 3);
    let (quantized, scales) = quantize_rows_i8(&pseudo_random(rows * in_features, 41), rows);
    let x = pseudo_random(tokens * in_features, 42);

    let whole = gpu.upload(&quantized, &scales, in_features);
    let split = gpu.upload_chunked(&quantized, &scales, in_features, 150);
    assert_eq!(Gpu::chunk_count(&whole), 1);
    assert_eq!(Gpu::chunk_count(&split), 7);

    // Each row is computed by the same shader on the same data either way.
    assert_eq!(gpu.matmul(&split, &x, tokens), gpu.matmul(&whole, &x, tokens));
}

#[test]
fn a_hand_worked_example() {
    let Some(gpu) = gpu() else { return };
    // [[1, -2, 3], [-128, 0, 127]] with scales [0.5, 2], against [1, 1, 1].
    let matrix = gpu.upload(&[1, -2, 3, -128, 0, 127], &[0.5, 2.0], 3);
    assert_eq!(gpu.matmul(&matrix, &[1.0, 1.0, 1.0], 1), vec![1.0, -2.0]);
}
