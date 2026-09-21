//! Hot-loop kernels for nanoinfer.
//!
//! Phase 6 ended on a specific, testable claim. Decoding one token is bound by
//! memory bandwidth -- it reads every weight in the model to produce a single
//! vector -- so storing weights as int8 should make it roughly four times
//! cheaper. In NumPy it does the opposite, because there is no integer GEMM:
//! the int8 weights have to be widened into a materialised float32 array
//! before any matmul can touch them, and that widening costs far more than the
//! matmul it feeds. Measured on a 4864x896 weight: 0.24 ms for plain fp32
//! against 20.03 ms going through int8.
//!
//! The claim was that this is a NumPy limitation, not a real one. A kernel can
//! widen each int8 in a register, use it, and discard it -- never materialising
//! the float32 array at all, and reading a quarter of the bytes. This crate
//! exists to find out whether that is actually true.
//!
//! Everything here is a mat*vec*, not a matmul. Decode runs one token at a
//! time, so every linear layer is `[1, in] x [out, in]` -- the shape where
//! bandwidth dominates and where BLAS has the least room to amortise.

/// Row-major `[out_features, in_features]` float32 weights times a vector.
///
/// The baseline. This is what the engine does today via NumPy, restated in
/// Rust so the comparison is against the same arithmetic rather than against
/// a different algorithm.
pub fn matvec_f32(weights: &[f32], x: &[f32], out: &mut [f32]) {
    let in_features = x.len();
    assert_eq!(weights.len(), out.len() * in_features, "weight shape mismatch");

    for (row, y) in out.iter_mut().enumerate() {
        let start = row * in_features;
        let w = &weights[start..start + in_features];

        // Written as an explicit accumulator loop rather than an iterator
        // chain so the reduction order is obvious: strictly sequential, which
        // is what makes it comparable with the SIMD versions later.
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += w[i] * x[i];
        }
        *y = sum;
    }
}

/// Deterministic pseudo-random floats, for tests and benchmarks.
///
/// A tiny xorshift rather than a `rand` dependency: the crate has none, the
/// values only need to be varied and repeatable, and a fixed generator means a
/// benchmark measures the same work every run.
pub fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map into roughly [-1, 1); the exact distribution does not matter,
            // only that it is not all zeros and does not change between runs.
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}
