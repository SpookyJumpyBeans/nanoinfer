use nanoinfer_kernels::{matvec_f32, pseudo_random};

/// A hand-computed case, so a wrong answer cannot hide behind a wrong oracle.
#[test]
fn matches_a_worked_example() {
    // [[1, 2], [3, 4], [5, 6]] . [7, 8]
    let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [7.0, 8.0];
    let mut out = [0.0; 3];

    matvec_f32(&weights, &x, &mut out);
    assert_eq!(out, [23.0, 53.0, 83.0]);
}

#[test]
fn a_zero_weight_row_yields_zero() {
    let weights = [0.0, 0.0, 1.0, 1.0];
    let x = [3.0, 4.0];
    let mut out = [9.0; 2];

    matvec_f32(&weights, &x, &mut out);
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 7.0);
}

#[test]
fn rows_are_independent() {
    // Changing one row must not disturb any other.
    let x = pseudo_random(64, 1);
    let mut weights = pseudo_random(8 * 64, 2);
    let mut before = vec![0.0; 8];
    matvec_f32(&weights, &x, &mut before);

    for w in weights.iter_mut().take(64) {
        *w = 0.0; // clobber row 0 only
    }
    let mut after = vec![0.0; 8];
    matvec_f32(&weights, &x, &mut after);

    assert_eq!(after[0], 0.0);
    assert_eq!(&after[1..], &before[1..]);
}

#[test]
#[should_panic(expected = "weight shape mismatch")]
fn rejects_a_shape_that_does_not_divide() {
    let mut out = [0.0; 3];
    matvec_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0], &mut out);
}

#[test]
fn handles_the_real_decode_shape() {
    // gate_proj: [4864, 896] against a single token.
    let (out_features, in_features) = (4864, 896);
    let weights = pseudo_random(out_features * in_features, 3);
    let x = pseudo_random(in_features, 4);
    let mut out = vec![0.0; out_features];

    matvec_f32(&weights, &x, &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|v| *v != 0.0));
}

use nanoinfer_kernels::{matvec_i8, quantize_rows_i8};

#[test]
fn int8_matvec_matches_a_worked_example() {
    // Row 0: [1, 2] * 0.5 against [7, 8] -> (7 + 16) * 0.5 = 11.5
    // Row 1: [3, 4] * 2.0 against [7, 8] -> (21 + 32) * 2.0 = 106
    let quantized = [1i8, 2, 3, 4];
    let scales = [0.5f32, 2.0];
    let x = [7.0f32, 8.0];
    let mut out = [0.0; 2];

    matvec_i8(&quantized, &scales, &x, &mut out);
    assert_eq!(out, [11.5, 106.0]);
}

#[test]
fn int8_tracks_fp32_on_real_shapes() {
    // Quantizing and running the int8 kernel must land close to the fp32
    // answer -- the same ~1% weight error phase 6 measured, not a new one.
    let (out_features, in_features) = (512, 896);
    let weights = pseudo_random(out_features * in_features, 7);
    let x = pseudo_random(in_features, 8);

    let mut reference = vec![0.0; out_features];
    matvec_f32(&weights, &x, &mut reference);

    let (quantized, scales) = quantize_rows_i8(&weights, out_features);
    let mut actual = vec![0.0; out_features];
    matvec_i8(&quantized, &scales, &x, &mut actual);

    let error: f32 = reference
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let norm: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(error / norm < 0.02, "relative error {}", error / norm);
}

#[test]
fn quantization_uses_the_full_range() {
    let weights = pseudo_random(4 * 64, 9);
    let (quantized, scales) = quantize_rows_i8(&weights, 4);

    assert!(scales.iter().all(|s| *s > 0.0));
    for row in 0..4 {
        let peak = quantized[row * 64..(row + 1) * 64]
            .iter()
            .map(|q| q.abs())
            .max()
            .unwrap();
        assert_eq!(peak, 127, "row {row} should reach the endpoint");
    }
}

#[test]
fn an_all_zero_row_does_not_divide_by_zero() {
    let weights = vec![0.0f32; 32];
    let (quantized, scales) = quantize_rows_i8(&weights, 1);
    assert!(scales[0].is_finite() && scales[0] > 0.0);
    assert!(quantized.iter().all(|q| *q == 0));
}

#[test]
#[should_panic(expected = "one scale per output row")]
fn rejects_a_missing_scale() {
    let mut out = [0.0; 2];
    matvec_i8(&[1, 2, 3, 4], &[1.0], &[1.0, 1.0], &mut out);
}

use nanoinfer_kernels::matvec_i8_auto;

#[test]
fn simd_and_scalar_agree() {
    // Not bit-identical: the SIMD version accumulates in eight lanes and sums
    // them at the end, so the additions happen in a different order. Float
    // addition is not associative, so the tolerance is real, not laziness.
    for &(out_features, in_features) in &[(1, 8), (3, 896), (64, 4864), (5, 17)] {
        let weights = pseudo_random(out_features * in_features, 11);
        let x = pseudo_random(in_features, 12);
        let (quantized, scales) = quantize_rows_i8(&weights, out_features);

        let mut scalar = vec![0.0; out_features];
        matvec_i8(&quantized, &scales, &x, &mut scalar);

        let mut simd = vec![0.0; out_features];
        matvec_i8_auto(&quantized, &scales, &x, &mut simd);

        for (a, b) in scalar.iter().zip(&simd) {
            let tolerance = a.abs().max(1.0) * 1e-4;
            assert!(
                (a - b).abs() < tolerance,
                "scalar {a} vs simd {b} at shape {out_features}x{in_features}"
            );
        }
    }
}

#[test]
fn simd_handles_a_ragged_tail() {
    // 17 is not a multiple of 8, so the tail loop has to run.
    let weights = pseudo_random(4 * 17, 13);
    let x = pseudo_random(17, 14);
    let (quantized, scales) = quantize_rows_i8(&weights, 4);

    let mut scalar = vec![0.0; 4];
    matvec_i8(&quantized, &scales, &x, &mut scalar);
    let mut simd = vec![0.0; 4];
    matvec_i8_auto(&quantized, &scales, &x, &mut simd);

    for (a, b) in scalar.iter().zip(&simd) {
        assert!((a - b).abs() < a.abs().max(1.0) * 1e-4, "{a} vs {b}");
    }
}

use nanoinfer_kernels::{matvec_i8_parallel, matvec_i8_spawn_per_call};

#[test]
fn parallel_agrees_with_single_threaded() {
    for &(out_features, in_features) in &[(64, 896), (4864, 896), (896, 4864), (7, 128)] {
        let weights = pseudo_random(out_features * in_features, 31);
        let x = pseudo_random(in_features, 32);
        let (quantized, scales) = quantize_rows_i8(&weights, out_features);

        let mut single = vec![0.0; out_features];
        matvec_i8_auto(&quantized, &scales, &x, &mut single);

        let mut parallel = vec![0.0; out_features];
        matvec_i8_parallel(&quantized, &scales, &x, &mut parallel);
        // Each row is computed entirely by one thread, so splitting rows
        // changes nothing about the arithmetic: this one IS bitwise.
        assert_eq!(single, parallel, "pooled, shape={out_features}x{in_features}");

        // The slow implementation has to agree too, or the benchmark would be
        // comparing a correct kernel against a broken one.
        for threads in [1, 2, 4, 0] {
            let mut spawned = vec![0.0; out_features];
            matvec_i8_spawn_per_call(&quantized, &scales, &x, &mut spawned, threads);
            assert_eq!(
                single, spawned,
                "threads={threads} shape={out_features}x{in_features}"
            );
        }
    }
}

#[test]
fn a_small_matrix_falls_back_rather_than_spawning() {
    // 7 rows across 20 threads would be mostly join overhead.
    let weights = pseudo_random(7 * 64, 33);
    let x = pseudo_random(64, 34);
    let (quantized, scales) = quantize_rows_i8(&weights, 7);

    let mut expected = vec![0.0; 7];
    matvec_i8_auto(&quantized, &scales, &x, &mut expected);
    let mut actual = vec![0.0; 7];
    matvec_i8_parallel(&quantized, &scales, &x, &mut actual);

    assert_eq!(expected, actual);
}
