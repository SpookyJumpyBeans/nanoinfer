//! GPU int8 matmul against the CPU kernel, on the shapes decode runs.
//!
//!     cd gpu && cargo run --release --example bench
//!
//! Each GPU time is a whole round trip -- activations up, dispatch, outputs
//! read back -- because that is what the engine pays per call; the weights
//! are uploaded once beforehand and stay resident, as they would.
//!
//! On a software adapter (llvmpipe and friends) the shader runs on the CPU
//! through an emulation layer, and the GPU column means nothing. The binary
//! says so rather than print a number someone might quote.

use std::time::Instant;

use nanoinfer_gpu::Gpu;
use nanoinfer_kernels::{matmul_i8_parallel, pseudo_random, quantize_rows_i8};

const SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj  896x896", 896, 896),
    ("kv_proj 128x896", 128, 896),
    ("o_proj  896x896", 896, 896),
    ("gate   4864x896", 4864, 896),
    ("down   896x4864", 896, 4864),
    ("lm_head 151936x896", 151_936, 896),
];

fn best_of<F: FnMut()>(mut f: F, runs: usize) -> f64 {
    f();
    (0..runs)
        .map(|_| {
            let started = Instant::now();
            f();
            started.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("adapter: {}", gpu.describe());
    if gpu.is_software() {
        println!("WARNING: software adapter -- the gpu column is CPU emulation, not a GPU timing\n");
    }

    println!("{:<20} {:>10} {:>10} {:>9}", "shape", "cpu ms", "gpu ms", "gpu/cpu");
    for &(name, rows, in_features) in SHAPES {
        let (quantized, scales) = quantize_rows_i8(&pseudo_random(rows * in_features, 5), rows);
        let x = pseudo_random(in_features, 6);
        let matrix = gpu.upload(&quantized, &scales, in_features);
        let mut out = vec![0.0; rows];

        let runs = if rows > 100_000 { 5 } else { 20 };
        let cpu = best_of(|| matmul_i8_parallel(&quantized, &scales, &x, 1, &mut out), runs);
        let gpu_ms = best_of(|| drop(gpu.matmul(&matrix, &x, 1)), runs);
        println!("{name:<20} {cpu:>10.3} {gpu_ms:>10.3} {:>8.2}x", gpu_ms / cpu);
    }
}
