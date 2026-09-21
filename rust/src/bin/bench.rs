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
    println!("{:<22} {:>10} {:>10} {:>10} {:>9}", "shape", "fp32 ms", "i8 ms", "i8+avx2", "speedup");
    println!("{}", "-".repeat(66));

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
    let mut total_avx = 0.0;

    for (name, out_features, in_features) in shapes {
        let weights = pseudo_random(out_features * in_features, 21);
        let x = pseudo_random(*in_features, 22);
        let (quantized, scales) = quantize_rows_i8(&weights, *out_features);
        let mut out = vec![0.0f32; *out_features];

        let runs = 50;
        let f32_ms = best_of(|| matvec_f32(&weights, &x, &mut out), runs);
        let i8_ms = best_of(|| matvec_i8(&quantized, &scales, &x, &mut out), runs);
        let avx_ms = best_of(|| matvec_i8_auto(&quantized, &scales, &x, &mut out), runs);

        total_f32 += f32_ms;
        total_avx += avx_ms;

        println!(
            "{name:<22} {f32_ms:>10.3} {i8_ms:>10.3} {avx_ms:>10.3} {:>8.2}x",
            f32_ms / avx_ms
        );
    }

    println!("{}", "-".repeat(66));
    println!(
        "{:<22} {total_f32:>10.3} {:>10} {total_avx:>10.3} {:>8.2}x",
        "one layer total",
        "",
        total_f32 / total_avx
    );
    println!(
        "\n24 layers: fp32 {:.1} ms, int8+avx2 {:.1} ms per decoded token",
        total_f32 * 24.0,
        total_avx * 24.0
    );
}
