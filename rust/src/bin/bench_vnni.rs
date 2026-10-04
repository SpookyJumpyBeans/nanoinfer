//! Times the AVX-VNNI int8 kernel against the float-activation AVX2 one.
//!
//! Both read the same int8 weights. The difference is the inner loop: AVX2
//! widens eight bytes to float at a time and multiplies in floating point,
//! VNNI consumes thirty-two bytes per instruction and accumulates in i32.
//!
//! Best of many runs, not the mean. This laptop stalls for seconds at a time
//! under load, so a mean measures the machine's mood and a minimum measures
//! the code -- the same convention phases 4, 7 and 8 settled on.

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
    println!("avx-vnni available: {}", vnni_available());
    if !vnni_available() {
        println!("nothing to compare: the VNNI entry point would fall back");
        return;
    }
    println!("best of 50\n");
    println!(
        "{:<22} {:>10} {:>10} {:>10}",
        "shape", "avx2 ms", "vnni ms", "speedup"
    );
    println!("{}", "-".repeat(56));

    // Every linear projection a decode step runs, plus the LM head, which is
    // the largest single read and so the one most likely to be memory-bound
    // rather than instruction-bound.
    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj   896x896", 896, 896),
        ("kv_proj  128x896", 128, 896),
        ("o_proj   896x896", 896, 896),
        ("gate    4864x896", 4864, 896),
        ("up      4864x896", 4864, 896),
        ("down    896x4864", 896, 4864),
        ("lm_head 151936x896", 151936, 896),
    ];

    let mut total_avx = 0.0;
    let mut total_vnni = 0.0;

    for (name, out_features, in_features) in shapes {
        let weights = pseudo_random(out_features * in_features, 21);
        let x = pseudo_random(*in_features, 22);
        let (quantized, scales) = quantize_rows_i8(&weights, *out_features);
        let mut out = vec![0.0f32; *out_features];

        let runs = if *out_features > 100_000 { 10 } else { 50 };
        let avx = best_of(|| matvec_i8_auto(&quantized, &scales, &x, &mut out), runs);
        let vnni = best_of(|| matvec_i8_vnni(&quantized, &scales, &x, &mut out), runs);

        total_avx += avx;
        total_vnni += vnni;

        println!(
            "{name:<22} {avx:>10.3} {vnni:>10.3} {:>9.2}x",
            avx / vnni
        );
    }

    println!("{}", "-".repeat(56));
    println!(
        "{:<22} {total_avx:>10.3} {total_vnni:>10.3} {:>9.2}x",
        "one token, all of it",
        total_avx / total_vnni
    );

    // What the whole model costs, which is the number that has to beat
    // llama.cpp's Q8_0. Twenty-four layers of projections share the per-layer
    // shapes; the LM head runs once.
    let per_layer_avx = total_avx - 0.0;
    println!(
        "\nnote: the rows above are one call each. A decode step runs the six\n\
         projection shapes 24 times over and the LM head once, so the engine\n\
         number comes from tools/bench_llamacpp.py, not from here."
    );
    let _ = per_layer_avx;
}
