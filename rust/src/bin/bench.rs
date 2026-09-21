//! Times the kernels on the shapes the engine actually decodes with.
//!
//! Reports the best of many runs, not the mean. This laptop produces sporadic
//! multi-second stalls under load -- phase 4 measured the same generation
//! varying between 2.97 s and 17.32 s -- so a mean measures the machine's mood
//! and a minimum measures the code.

use nanoinfer_kernels::*;
use std::time::Instant;

fn best_of<F: FnMut()>(mut f: F, runs: usize) -> f64 {
    f(); // warm the caches and let the branch predictor settle
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("one core against {cores}, best of 50
");
    println!(
        "{:<22} {:>9} {:>9} {:>10} {:>9} {:>9}",
        "shape", "fp32 ms", "i8+avx2", "mt spawn", "mt pool", "vs fp32"
    );
    println!("{}", "-".repeat(74));

    // Every linear layer Qwen2.5-0.5B runs per decoded token.
    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj  896x896", 896, 896),
        ("k_proj  128x896", 128, 896),
        ("o_proj  896x896", 896, 896),
        ("gate   4864x896", 4864, 896),
        ("up     4864x896", 4864, 896),
        ("down   896x4864", 896, 4864),
    ];

    let mut total_f32 = 0.0;
    let mut total_one = 0.0;
    let mut total_spawn = 0.0;
    let mut total_pool = 0.0;

    for (name, out_features, in_features) in shapes {
        let weights = pseudo_random(out_features * in_features, 21);
        let x = pseudo_random(*in_features, 22);
        let (quantized, scales) = quantize_rows_i8(&weights, *out_features);
        let mut out = vec![0.0f32; *out_features];

        let runs = 50;
        let f32_ms = best_of(|| matvec_f32(&weights, &x, &mut out), runs);
        let avx_ms = best_of(|| matvec_i8_auto(&quantized, &scales, &x, &mut out), runs);
        let spawn_ms = best_of(
            || matvec_i8_spawn_per_call(&quantized, &scales, &x, &mut out, 0),
            runs,
        );
        let pool_ms = best_of(
            || matvec_i8_parallel(&quantized, &scales, &x, &mut out),
            runs,
        );

        total_f32 += f32_ms;
        total_one += avx_ms;
        total_spawn += spawn_ms;
        total_pool += pool_ms;

        println!(
            "{name:<22} {f32_ms:>9.3} {avx_ms:>9.3} {spawn_ms:>10.3} {pool_ms:>9.3} {:>8.2}x",
            f32_ms / pool_ms
        );
    }

    println!("{}", "-".repeat(74));
    println!(
        "{:<22} {total_f32:>9.3} {total_one:>9.3} {total_spawn:>10.3} {total_pool:>9.3} {:>8.2}x",
        "one layer total",
        total_f32 / total_pool
    );
    println!(
        "
NOTE: the fp32 column is this crate's scalar loop, not OpenBLAS.
         Compare against numpy with: python -m tools.bench_kernels"
    );
    println!(
        "
24 layers: fp32 {:.1} ms, int8+avx2 {:.1} ms, +pool {:.1} ms per decoded token",
        total_f32 * 24.0,
        total_one * 24.0,
        total_pool * 24.0
    );
    println!(
        "spawning threads per call costs {:.1}x against the same work on a pool",
        total_spawn / total_pool
    );
}
