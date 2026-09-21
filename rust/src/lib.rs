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

/// Row-major int8 weights with one float32 scale per row, times a vector.
///
/// This is the kernel phase 6 could not write in NumPy. The int8 is widened to
/// f32 *inside the loop*, one value at a time, so the widened weights exist
/// only in registers. Nothing the size of the weight matrix is ever allocated,
/// and the bytes actually read from memory are a quarter of the fp32 version.
///
/// The scale is applied once per row, to the finished dot product, rather than
/// per element. That is algebraically the same -- `sum(s * q_i * x_i)` is
/// `s * sum(q_i * x_i)` -- and it turns `in_features` multiplies into one.
/// It is not bit-identical to scaling per element, because float addition is
/// not associative; it is the more accurate order, since the accumulation
/// happens before the magnitudes are rescaled.
pub fn matvec_i8(quantized: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    let in_features = x.len();
    assert_eq!(
        quantized.len(),
        out.len() * in_features,
        "weight shape mismatch"
    );
    assert_eq!(scales.len(), out.len(), "expected one scale per output row");

    for (row, y) in out.iter_mut().enumerate() {
        let start = row * in_features;
        let w = &quantized[start..start + in_features];

        let mut sum = 0.0f32;
        for i in 0..in_features {
            // The widening that NumPy has to do to a whole array, done here to
            // a single value that never leaves the register file.
            sum += (w[i] as f32) * x[i];
        }
        *y = sum * scales[row];
    }
}

/// Quantize a row-major `[out, in]` float32 matrix to symmetric per-row int8.
///
/// Mirrors `nanoinfer.quantization.quantize_int8` exactly, including the
/// [-127, 127] range and rounding half away from zero, so the Rust and Python
/// sides agree on what the stored integers should be. Tests compare them.
pub fn quantize_rows_i8(weights: &[f32], out_features: usize) -> (Vec<i8>, Vec<f32>) {
    let in_features = weights.len() / out_features;
    let mut quantized = vec![0i8; weights.len()];
    let mut scales = vec![0.0f32; out_features];

    for row in 0..out_features {
        let start = row * in_features;
        let w = &weights[start..start + in_features];

        let magnitude = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // An all-zero row would divide by zero; any scale reconstructs zeros.
        let scale = if magnitude == 0.0 { 1.0 } else { magnitude / 127.0 };
        scales[row] = scale;

        for i in 0..in_features {
            let scaled = w[i] / scale;
            // Round half away from zero, matching the Python side. Rust's
            // f32::round already does this; it is named here because the
            // NumPy default (banker's rounding) does not.
            let rounded = scaled.round();
            quantized[start + i] = rounded.clamp(-127.0, 127.0) as i8;
        }
    }

    (quantized, scales)
}

// -- AVX2 ------------------------------------------------------------------
//
// The scalar kernel already reads a quarter of the bytes, which is the point.
// AVX2 is about whether the widening itself can keep up: eight int8 values are
// sign-extended to i32, converted to f32, and fused-multiply-added against the
// activations in a handful of instructions, rather than one conversion per
// element.
//
// Availability is checked at runtime rather than assumed at compile time. This
// laptop is Alder Lake, so AVX2 yes and AVX-512 no -- but a binary that crashes
// on an older machine is worse than one that is slightly slower everywhere, and
// the dispatch costs one predictable branch per call.

/// Dispatches to the AVX2 kernel when the CPU has it, else the scalar one.
pub fn matvec_i8_auto(quantized: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above, and the
            // shape assertions inside match the scalar kernel's.
            unsafe { return matvec_i8_avx2(quantized, scales, x, out) }
        }
    }
    matvec_i8(quantized, scales, x, out)
}

/// AVX2 + FMA int8 matvec.
///
/// # Safety
/// The caller must ensure AVX2 and FMA are available; use `matvec_i8_auto`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matvec_i8_avx2(quantized: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let in_features = x.len();
    assert_eq!(quantized.len(), out.len() * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out.len(), "expected one scale per output row");

    let chunks = in_features / 8;
    let tail = chunks * 8;

    for (row, y) in out.iter_mut().enumerate() {
        let w = quantized.as_ptr().add(row * in_features);

        let mut acc = _mm256_setzero_ps();
        for chunk in 0..chunks {
            let offset = chunk * 8;

            // Eight int8 -> eight i32 (sign-extended) -> eight f32. The whole
            // widening happens in registers; no array is ever materialised.
            let packed = _mm_loadl_epi64(w.add(offset) as *const __m128i);
            let widened = _mm256_cvtepi8_epi32(packed);
            let weights = _mm256_cvtepi32_ps(widened);

            let activations = _mm256_loadu_ps(x.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(weights, activations, acc);
        }

        // Horizontal sum of the eight lanes. Done once per row, so its cost is
        // amortised over in_features multiply-adds.
        let high = _mm256_extractf128_ps(acc, 1);
        let low = _mm256_castps256_ps128(acc);
        let mut sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut sum = _mm_cvtss_f32(sum128);

        // Whatever did not fill a lane. in_features is 896 or 4864 for this
        // model, both multiples of 8, so this is empty in practice -- but a
        // kernel that silently drops the tail is a trap for the next shape.
        for i in tail..in_features {
            sum += (*w.add(i) as f32) * x[i];
        }

        *y = sum * scales[row];
    }
}
