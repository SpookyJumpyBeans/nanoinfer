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
    let in_features = x.len();
    assert_eq!(quantized.len(), out.len() * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out.len(), "expected one scale per output row");

    for (row, y) in out.iter_mut().enumerate() {
        let w = &quantized[row * in_features..(row + 1) * in_features];
        *y = dot_i8_avx2(w, x) * scales[row];
    }
}

/// How far ahead of the current block the AVX2 dot product prefetches.
#[cfg(target_arch = "x86_64")]
const PREFETCH_BYTES: usize = 4096;

/// One int8 row against one float32 vector, unscaled.
///
/// Split out of the matvec so the batched kernel below runs *exactly* the
/// same instructions per row. That is what lets the tests demand bitwise
/// equality between one batched call and a loop of matvecs, rather than a
/// tolerance that could hide a real difference.
///
/// # Safety
/// AVX2 and FMA must be available, and `w.len() == x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_i8_avx2(w: &[i8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let in_features = x.len();
    let w = w.as_ptr();
    let x = x.as_ptr();

    // Four independent accumulators, 32 weights per iteration. With one
    // accumulator every FMA waits ~4 cycles on the previous one; four chains
    // let the core overlap them. That doubled the cache-hot speed and did
    // nothing for whole-model decode, which streams from DRAM -- the prefetch
    // below is what fixed that.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    // Eight int8 -> eight i32 (sign-extended) -> eight f32. The whole
    // widening happens in registers; no array is ever materialised.
    let widen = |bytes: __m128i| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));

    let wide = in_features / 32 * 32;
    let mut offset = 0;
    while offset < wide {
        // Ask for the weights PREFETCH_BYTES ahead now, so they are in flight
        // while this block is being multiplied. Rows are contiguous, so near
        // the end of a row this reaches into the next one -- which is exactly
        // the row this thread will read next.
        _mm_prefetch::<_MM_HINT_T0>(w.add(offset + PREFETCH_BYTES) as *const i8);
        let low = _mm_loadu_si128(w.add(offset) as *const __m128i);
        let high = _mm_loadu_si128(w.add(offset + 16) as *const __m128i);

        acc0 = _mm256_fmadd_ps(widen(low), _mm256_loadu_ps(x.add(offset)), acc0);
        acc1 = _mm256_fmadd_ps(
            widen(_mm_srli_si128(low, 8)),
            _mm256_loadu_ps(x.add(offset + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(widen(high), _mm256_loadu_ps(x.add(offset + 16)), acc2);
        acc3 = _mm256_fmadd_ps(
            widen(_mm_srli_si128(high, 8)),
            _mm256_loadu_ps(x.add(offset + 24)),
            acc3,
        );
        offset += 32;
    }

    // Leftover whole lanes of eight, into the first chain.
    let tail = in_features / 8 * 8;
    while offset < tail {
        let packed = _mm_loadl_epi64(w.add(offset) as *const __m128i);
        acc0 = _mm256_fmadd_ps(widen(packed), _mm256_loadu_ps(x.add(offset)), acc0);
        offset += 8;
    }

    // Combine the chains in a fixed order, then the horizontal sum of the
    // eight lanes. Done once per row, so its cost is amortised over
    // in_features multiply-adds.
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut sum = _mm_cvtss_f32(sum128);

    // Whatever did not fill a lane. in_features is 896 or 4864 for this
    // model, both multiples of 32, so this is empty in practice -- but a
    // kernel that silently drops the tail is a trap for the next shape.
    for i in tail..in_features {
        sum += (*w.add(i) as f32) * *x.add(i);
    }
    sum
}

/// The scalar counterpart of [`dot_i8_avx2`], same order as [`matvec_i8`].
fn dot_i8_scalar(w: &[i8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..x.len() {
        sum += (w[i] as f32) * x[i];
    }
    sum
}

// -- AVX-VNNI ---------------------------------------------------------------
//
// The AVX2 kernel above reads a quarter of the bytes and then throws most of
// that advantage away: every eight int8 weights are sign-extended to i32,
// converted to f32, and multiplied against f32 activations. On the real model
// that kernel runs at about 6.4 GB/s where the machine can do ~24, so it is
// compute-bound on the conversion rather than waiting for memory -- which is
// how llama.cpp's Q8_0 beats it while pinned to a single thread.
//
// VPDPBUSD takes thirty-two bytes per instruction and accumulates into i32,
// with no conversion in the inner loop at all. It needs both operands as
// bytes, so the activations are quantized too; that is a real loss of
// precision, 8 bits where there were 32, and the reason this is a separate
// entry point rather than a silent replacement.
//
// The instruction multiplies *unsigned* bytes by *signed* bytes. Two ways to
// satisfy that, and only one is cheap: make the weights unsigned, not the
// activations. XOR with 0x80 reinterprets i8 as u8 in order, which adds 128
// to every weight, so
//
//     dpbusd(w ^ 0x80, xq) = sum (w + 128) * xq
//                          = sum (w * xq) + 128 * sum xq
//
// and the correction is a single scalar per call, because the activation
// vector is shared by every row. Had the activations been offset instead, the
// correction would have been 128 * sum(w) -- a different value per row, and a
// second pass over the weights to compute it.

/// Quantize an activation vector to symmetric int8, returning its scale.
///
/// One scale for the whole vector rather than per anything: the vector is a
/// single row of activations and the kernel multiplies all of it by the same
/// weights. An all-zero vector would divide by zero, so its scale is 1 and it
/// reconstructs as zeros either way.
pub fn quantize_activations_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let magnitude = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if magnitude == 0.0 { 1.0 } else { magnitude / 127.0 };
    let mut out = vec![0i8; x.len()];
    for i in 0..x.len() {
        out[i] = (x[i] / scale).round().clamp(-127.0, 127.0) as i8;
    }
    (out, scale)
}

/// Whether this CPU can run the VNNI kernel.
pub fn vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// int8 matvec with int8 activations, through VPDPBUSD where available.
///
/// Quantizes `x` internally, so the caller's interface is the same as
/// [`matvec_i8_auto`] -- but the result is not the same, because the
/// activations lose 24 bits on the way in. Expect agreement to about one part
/// in a hundred, not to float32 precision, and see tests/vnni.rs for the bound
/// this is held to.
///
/// Falls back to [`matvec_i8_auto`] on a CPU without VNNI, which keeps the
/// float activations and is therefore *more* accurate, not less.
pub fn matvec_i8_vnni(quantized: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    let in_features = x.len();
    let out_features = out.len();
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out_features, "expected one scale per output row");

    #[cfg(target_arch = "x86_64")]
    {
        if vnni_available() {
            let (xq, x_scale) = quantize_activations_i8(x);
            // sum(xq) once, not once per row: this is the whole reason the
            // weights carry the offset instead of the activations.
            let correction: i32 = xq.iter().map(|v| *v as i32).sum();
            for row in 0..out_features {
                let start = row * in_features;
                // SAFETY: feature-checked above; the slice bounds come from
                // the assertions at the top of this function.
                let raw = unsafe {
                    dot_i8_i8_vnni(&quantized[start..start + in_features], &xq)
                };
                out[row] = (raw - 128 * correction) as f32 * scales[row] * x_scale;
            }
            return;
        }
    }
    matvec_i8_auto(quantized, scales, x, out)
}

/// `sum (w ^ 0x80) * xq`, as i32. The caller subtracts `128 * sum(xq)`.
///
/// # Safety
/// AVX2 and AVX-VNNI must be available, and `w.len() == xq.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
#[inline]
unsafe fn dot_i8_i8_vnni(w: &[i8], xq: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let in_features = w.len();
    let wp = w.as_ptr();
    let xp = xq.as_ptr();

    // Four chains again, for the same reason as the AVX2 kernel: VPDPBUSD has
    // a multi-cycle latency and independent accumulators let the core overlap
    // them. Thirty-two bytes per instruction, so 128 per iteration.
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    // XOR with this reinterprets each signed byte as the unsigned byte 128
    // greater, which is what VPDPBUSD's first operand has to be.
    let sign_flip = _mm256_set1_epi8(-128i8); // 0x80

    let wide = in_features / 128 * 128;
    let mut offset = 0;
    while offset < wide {
        _mm_prefetch::<_MM_HINT_T0>(wp.add(offset + PREFETCH_BYTES) as *const i8);

        let w0 = _mm256_xor_si256(_mm256_loadu_si256(wp.add(offset) as *const __m256i), sign_flip);
        let w1 = _mm256_xor_si256(_mm256_loadu_si256(wp.add(offset + 32) as *const __m256i), sign_flip);
        let w2 = _mm256_xor_si256(_mm256_loadu_si256(wp.add(offset + 64) as *const __m256i), sign_flip);
        let w3 = _mm256_xor_si256(_mm256_loadu_si256(wp.add(offset + 96) as *const __m256i), sign_flip);

        let x0 = _mm256_loadu_si256(xp.add(offset) as *const __m256i);
        let x1 = _mm256_loadu_si256(xp.add(offset + 32) as *const __m256i);
        let x2 = _mm256_loadu_si256(xp.add(offset + 64) as *const __m256i);
        let x3 = _mm256_loadu_si256(xp.add(offset + 96) as *const __m256i);

        acc0 = _mm256_dpbusd_avx_epi32(acc0, w0, x0);
        acc1 = _mm256_dpbusd_avx_epi32(acc1, w1, x1);
        acc2 = _mm256_dpbusd_avx_epi32(acc2, w2, x2);
        acc3 = _mm256_dpbusd_avx_epi32(acc3, w3, x3);
        offset += 128;
    }

    // Whole 32-byte vectors that did not fill a 128-byte iteration.
    let vectors = in_features / 32 * 32;
    while offset < vectors {
        let wv = _mm256_xor_si256(_mm256_loadu_si256(wp.add(offset) as *const __m256i), sign_flip);
        let xv = _mm256_loadu_si256(xp.add(offset) as *const __m256i);
        acc0 = _mm256_dpbusd_avx_epi32(acc0, wv, xv);
        offset += 32;
    }

    let acc = _mm256_add_epi32(
        _mm256_add_epi32(acc0, acc1),
        _mm256_add_epi32(acc2, acc3),
    );
    let high = _mm256_extracti128_si256(acc, 1);
    let low = _mm256_castsi256_si128(acc);
    let mut sum128 = _mm_add_epi32(low, high);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    let mut sum = _mm_cvtsi128_si32(sum128);

    // The remainder, done the same way the vector path does it -- including
    // the 0x80 offset, so the caller's single correction still applies to the
    // whole row. 896 and 4864 are both multiples of 32, so this is empty for
    // this model, but a kernel that drops its tail is a trap for the next one.
    for i in offset..in_features {
        let unsigned = (*wp.add(i) as i32) + 128;
        sum += unsigned * (*xp.add(i) as i32);
    }
    sum
}

// -- multithreading --------------------------------------------------------
//
use rayon::prelude::*;

// The single-threaded AVX2 kernel loses to OpenBLAS by about 2.75x, and
// OpenBLAS is using every core. Output rows are completely independent -- row
// j reads its own slice of the weights and writes one float -- so the work
// splits with no synchronisation beyond the join, and no locking at all.
//
// That reasoning is correct and the first implementation of it was still 12x
// slower than one thread. Both kernels are kept below, because the gap between
// them is the whole lesson: the parallelism was never the problem, the thread
// *lifetime* was.

/// Parallel int8 matvec that spawns fresh threads on every call.
///
/// Kept, and benchmarked, because it is the obvious implementation and it is
/// dramatically wrong. `std::thread::scope` creates real OS threads at the
/// scope and joins them at its end, so every call pays the full construction
/// and teardown of one thread per worker. A 896x896 matvec takes ~90us of
/// actual work; spawning 20 threads to do it measured **1.19 ms**, an order of
/// magnitude worse than not parallelising at all.
///
/// Use [`matvec_i8_parallel`] instead. This exists so `bench` can show the
/// difference rather than assert it.
pub fn matvec_i8_spawn_per_call(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) {
    let in_features = x.len();
    let out_features = out.len();
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out_features, "expected one scale per output row");

    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };

    if threads <= 1 || out_features < threads * 8 {
        return matvec_i8_auto(quantized, scales, x, out);
    }

    let rows_per_thread = out_features.div_ceil(threads);

    std::thread::scope(|scope| {
        let mut remaining_out = out;
        let mut row_start = 0;

        while row_start < out_features {
            let block = rows_per_thread.min(out_features - row_start);
            let (chunk, rest) = remaining_out.split_at_mut(block);
            remaining_out = rest;

            let weights = &quantized[row_start * in_features..(row_start + block) * in_features];
            let block_scales = &scales[row_start..row_start + block];

            scope.spawn(move || matvec_i8_auto(weights, block_scales, x, chunk));
            row_start += block;
        }
    });
}

/// Parallel int8 matvec over a persistent worker pool.
///
/// Identical decomposition to [`matvec_i8_spawn_per_call`] -- contiguous
/// blocks of output rows, which keeps each worker's weight reads sequential,
/// the access pattern this is bandwidth-bound on. The only difference is that
/// rayon's workers already exist and are parked, so a call costs a wakeup
/// rather than a `CreateThread`.
///
/// That single change is the entire fix, and it is why the crate has one
/// dependency: writing a correct pool that can borrow the caller's slices
/// needs either unsafe lifetime transmutation or rayon's scope, and the
/// no-dependency version measured above is not a real alternative.
pub fn matvec_i8_parallel(quantized: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    let in_features = x.len();
    let out_features = out.len();
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out_features, "expected one scale per output row");

    let threads = rayon::current_num_threads();

    // Under this, the wakeup still costs more than the rows would.
    if threads <= 1 || out_features < threads * 8 {
        return matvec_i8_auto(quantized, scales, x, out);
    }

    let rows_per_thread = out_features.div_ceil(threads);

    out.par_chunks_mut(rows_per_thread)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let row_start = block_index * rows_per_thread;
            let weights = &quantized[row_start * in_features..(row_start + chunk.len()) * in_features];
            let block_scales = &scales[row_start..row_start + chunk.len()];
            matvec_i8_auto(weights, block_scales, x, chunk);
        });
}

// -- batched: more than one token ------------------------------------------
//
// Decode is one token at a time, but prefill is the whole prompt at once, and
// so is a perplexity pass. Calling the matvec once per token from Python would
// pay the ~8 us ctypes floor `tokens x 7 x 24` times, and stream every weight
// in from memory once per token.
//
// So the loop over tokens goes inside the loop over weight rows. Each row is
// read from memory once and then stays in L1 while every token is dotted
// against it. Results land transposed, `[out_features, tokens]`, which keeps
// each worker's output a single contiguous block: it owns a run of rows, and
// in this layout those rows are adjacent. Python transposes the view back.

/// int8 weights times `tokens` activation vectors, single-threaded.
///
/// `x` is row-major `[tokens, in_features]`; `out_t` is row-major
/// `[out_features, tokens]` -- note the transpose. Each output is computed by
/// the same per-row dot product as [`matvec_i8_auto`], so a batch of one is
/// bitwise identical to a matvec, and a batch of n to n matvecs.
pub fn matmul_i8_auto(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    tokens: usize,
    out_t: &mut [f32],
) {
    let out_features = scales.len();
    assert!(tokens > 0, "expected at least one token");
    assert_eq!(x.len() % tokens, 0, "activation shape mismatch");
    let in_features = x.len() / tokens;
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(out_t.len(), out_features * tokens, "output shape mismatch");

    #[cfg(target_arch = "x86_64")]
    let simd = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;

    // One token is a decode step: stream the int8 row straight through the
    // prefetching dot product, with nothing to amortise a widened copy over.
    if tokens == 1 {
        for (row, y) in out_t.iter_mut().enumerate() {
            let w = &quantized[row * in_features..(row + 1) * in_features];
            *y = dot_i8(simd, w, x) * scales[row];
        }
        return;
    }

    // More than one token: widen each row to f32 ONCE and dot every token in
    // the block against the widened copy. Re-widening per token made the
    // conversion -- not the multiply -- the cost, and left int8 prefill 2-6x
    // behind fp32 BLAS. int8 -> f32 is exact, so the widened row holds the
    // very values the int8 dot would have produced in registers, and
    // dot_f32 adds them in the same order: the result is bitwise unchanged.
    //
    // Tokens go in blocks whose activations fit in L2. A long prompt against
    // a 4864-wide down_proj is 2.5 MB of activations; walking all of it for
    // every row streamed it from L3 once per row.
    let block = (TOKEN_BLOCK_BYTES / (in_features * 4)).max(1);
    let mut widened = vec![0.0f32; in_features];

    for first in (0..tokens).step_by(block) {
        let last = (first + block).min(tokens);
        for row in 0..out_features {
            let w = &quantized[row * in_features..(row + 1) * in_features];
            for (dst, &src) in widened.iter_mut().zip(w) {
                *dst = src as f32;
            }
            let scale = scales[row];
            let token = |t: usize| &x[t * in_features..(t + 1) * in_features];
            let mut t = first;

            // Three tokens per pass over the widened row: each weight load
            // then feeds three FMAs instead of one, which is what the
            // one-token dot was starved of -- two loads per FMA saturate the
            // load ports long before the FMA units are busy.
            #[cfg(target_arch = "x86_64")]
            if simd {
                while t + 3 <= last {
                    // SAFETY: feature-checked; every slice is in_features long.
                    let [a, b, c] = unsafe { dot3_f32_avx2(&widened, token(t), token(t + 1), token(t + 2)) };
                    out_t[row * tokens + t] = a * scale;
                    out_t[row * tokens + t + 1] = b * scale;
                    out_t[row * tokens + t + 2] = c * scale;
                    t += 3;
                }
            }
            while t < last {
                out_t[row * tokens + t] = dot_f32(simd, &widened, token(t)) * scale;
                t += 1;
            }
        }
    }
}

/// Activations per token block in [`matmul_i8_auto`]: comfortably inside a
/// per-core L2 (1.25 MB on Alder Lake P-cores, 2 MB on the cloud Xeon).
const TOKEN_BLOCK_BYTES: usize = 256 * 1024;

/// One int8 row against one vector, through AVX2 when `simd` says it is live.
fn dot_i8(simd: bool, w: &[i8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: the caller feature-checked AVX2 and FMA; lengths match.
        return unsafe { dot_i8_avx2(w, x) };
    }
    let _ = simd;
    dot_i8_scalar(w, x)
}

/// An already-widened row against one vector. Same lane layout, same chain
/// assignment and same tail order as [`dot_i8_avx2`] / [`dot_i8_scalar`], so
/// on exactly-widened weights it returns the same bits.
fn dot_f32(simd: bool, w: &[f32], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: the caller feature-checked AVX2 and FMA; lengths match.
        return unsafe { dot_f32_avx2(w, x) };
    }
    let _ = simd;
    let mut sum = 0.0f32;
    for i in 0..x.len() {
        sum += w[i] * x[i];
    }
    sum
}

/// [`dot_f32_avx2`] for three activation vectors at once, sharing each
/// weight load. Every token keeps its own four chains, fed in the same order
/// and combined the same way, so each result is bitwise what
/// [`dot_f32_avx2`] returns for that token alone.
///
/// # Safety
/// AVX2 and FMA must be available, and every slice the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot3_f32_avx2(w: &[f32], x0: &[f32], x1: &[f32], x2: &[f32]) -> [f32; 3] {
    use std::arch::x86_64::*;

    let in_features = w.len();
    let w = w.as_ptr();
    let xs = [x0.as_ptr(), x1.as_ptr(), x2.as_ptr()];

    // acc[token][chain]: 12 registers, plus one weight and one activation.
    let mut acc = [[_mm256_setzero_ps(); 4]; 3];

    let wide = in_features / 32 * 32;
    let mut offset = 0;
    while offset < wide {
        for chain in 0..4 {
            let at = offset + 8 * chain;
            let weights = _mm256_loadu_ps(w.add(at));
            for token in 0..3 {
                let activations = _mm256_loadu_ps(xs[token].add(at));
                acc[token][chain] = _mm256_fmadd_ps(weights, activations, acc[token][chain]);
            }
        }
        offset += 32;
    }

    let tail = in_features / 8 * 8;
    while offset < tail {
        let weights = _mm256_loadu_ps(w.add(offset));
        for token in 0..3 {
            acc[token][0] = _mm256_fmadd_ps(weights, _mm256_loadu_ps(xs[token].add(offset)), acc[token][0]);
        }
        offset += 8;
    }

    let mut out = [0.0f32; 3];
    for token in 0..3 {
        let [a0, a1, a2, a3] = acc[token];
        let sum8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let high = _mm256_extractf128_ps(sum8, 1);
        let low = _mm256_castps256_ps128(sum8);
        let mut sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        let mut sum = _mm_cvtss_f32(sum128);
        for i in tail..in_features {
            sum += *w.add(i) * *xs[token].add(i);
        }
        out[token] = sum;
    }
    out
}

/// # Safety
/// AVX2 and FMA must be available, and `w.len() == x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(w: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let in_features = x.len();
    let w = w.as_ptr();
    let x = x.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let wide = in_features / 32 * 32;
    let mut offset = 0;
    while offset < wide {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(offset)), _mm256_loadu_ps(x.add(offset)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(offset + 8)), _mm256_loadu_ps(x.add(offset + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(offset + 16)), _mm256_loadu_ps(x.add(offset + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(offset + 24)), _mm256_loadu_ps(x.add(offset + 24)), acc3);
        offset += 32;
    }

    let tail = in_features / 8 * 8;
    while offset < tail {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(offset)), _mm256_loadu_ps(x.add(offset)), acc0);
        offset += 8;
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut sum = _mm_cvtss_f32(sum128);

    for i in tail..in_features {
        sum += *w.add(i) * *x.add(i);
    }
    sum
}

/// [`matmul_i8_auto`] split across the pool by blocks of output rows.
///
/// The same decomposition as [`matvec_i8_parallel`], and the same threshold
/// below which the wakeup costs more than the rows.
pub fn matmul_i8_parallel(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    tokens: usize,
    out_t: &mut [f32],
) {
    let out_features = scales.len();
    assert!(tokens > 0, "expected at least one token");
    assert_eq!(x.len() % tokens, 0, "activation shape mismatch");
    let in_features = x.len() / tokens;
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(out_t.len(), out_features * tokens, "output shape mismatch");

    let threads = rayon::current_num_threads();
    if threads <= 1 || out_features < threads * 8 {
        return matmul_i8_auto(quantized, scales, x, tokens, out_t);
    }

    let rows_per_thread = out_features.div_ceil(threads);

    out_t
        .par_chunks_mut(rows_per_thread * tokens)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let row_start = block_index * rows_per_thread;
            let rows = chunk.len() / tokens;
            let weights = &quantized[row_start * in_features..(row_start + rows) * in_features];
            let block_scales = &scales[row_start..row_start + rows];
            matmul_i8_auto(weights, block_scales, x, tokens, chunk);
        });
}

// -- C ABI --------------------------------------------------------------
//
// The engine is NumPy, so the kernels have to be reachable from Python. This
// is ctypes against a cdylib rather than PyO3: the whole surface is four
// pointers and two lengths, PyO3 would add a build step and a compiled
// extension per Python version, and ctypes needs neither.
//
// Everything here is unsafe by nature -- the caller passes raw pointers and
// asserts the lengths. The Python side owns that contract and checks shapes
// before it calls; see nanoinfer/kernels.py.

/// Runtime CPU feature report, so Python can say which kernel it got.
///
/// Returns 1 if the AVX2 path is live, 0 if this build fell back to scalar.
#[no_mangle]
pub extern "C" fn nanoinfer_has_avx2() -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        i32::from(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// Number of worker threads the parallel kernel will use.
#[no_mangle]
pub extern "C" fn nanoinfer_num_threads() -> i32 {
    rayon::current_num_threads() as i32
}

/// int8 matvec over the thread pool. `out` must have room for `out_features`.
///
/// # Safety
/// `quantized` must point to `out_features * in_features` readable bytes,
/// `scales` to `out_features` floats, `x` to `in_features` floats, and `out`
/// to `out_features` writable floats. No aliasing between `out` and the rest.
#[no_mangle]
pub unsafe extern "C" fn nanoinfer_matvec_i8(
    quantized: *const i8,
    scales: *const f32,
    x: *const f32,
    out: *mut f32,
    out_features: usize,
    in_features: usize,
) {
    let quantized = std::slice::from_raw_parts(quantized, out_features * in_features);
    let scales = std::slice::from_raw_parts(scales, out_features);
    let x = std::slice::from_raw_parts(x, in_features);
    let out = std::slice::from_raw_parts_mut(out, out_features);
    matvec_i8_parallel(quantized, scales, x, out);
}

/// Single-threaded int8 matvec, same contract as [`nanoinfer_matvec_i8`].
///
/// Exposed so the Python benchmark can separate the SIMD win from the
/// threading win rather than reporting one number for both.
///
/// # Safety
/// As [`nanoinfer_matvec_i8`].
#[no_mangle]
pub unsafe extern "C" fn nanoinfer_matvec_i8_single(
    quantized: *const i8,
    scales: *const f32,
    x: *const f32,
    out: *mut f32,
    out_features: usize,
    in_features: usize,
) {
    let quantized = std::slice::from_raw_parts(quantized, out_features * in_features);
    let scales = std::slice::from_raw_parts(scales, out_features);
    let x = std::slice::from_raw_parts(x, in_features);
    let out = std::slice::from_raw_parts_mut(out, out_features);
    matvec_i8_auto(quantized, scales, x, out);
}

/// Batched int8 matmul over the thread pool; see [`matmul_i8_parallel`].
///
/// This is the one the engine calls. A batch of one is a decode step, so the
/// forward pass needs no separate matvec path.
///
/// # Safety
/// `quantized` must point to `out_features * in_features` readable bytes,
/// `scales` to `out_features` floats, `x` to `tokens * in_features` floats,
/// and `out_t` to `out_features * tokens` writable floats. No aliasing between
/// `out_t` and the rest.
#[no_mangle]
pub unsafe extern "C" fn nanoinfer_matmul_i8(
    quantized: *const i8,
    scales: *const f32,
    x: *const f32,
    out_t: *mut f32,
    out_features: usize,
    in_features: usize,
    tokens: usize,
) {
    let quantized = std::slice::from_raw_parts(quantized, out_features * in_features);
    let scales = std::slice::from_raw_parts(scales, out_features);
    let x = std::slice::from_raw_parts(x, tokens * in_features);
    let out_t = std::slice::from_raw_parts_mut(out_t, out_features * tokens);
    matmul_i8_parallel(quantized, scales, x, tokens, out_t);
}

/// int8 weights times `tokens` activation vectors, through VPDPBUSD.
///
/// Same shapes as [`matmul_i8_auto`] -- `x` row-major `[tokens, in_features]`,
/// `out_t` row-major `[out_features, tokens]` -- and a different answer, by
/// about one part in a hundred: the activations are quantized to int8 so the
/// instruction can take them. Each token gets its own scale, so a long prompt
/// does not share one token's dynamic range with the rest.
///
/// Falls back to [`matmul_i8_auto`] without VNNI, which keeps f32 activations.
pub fn matmul_i8_vnni(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    tokens: usize,
    out_t: &mut [f32],
) {
    let out_features = scales.len();
    assert!(tokens > 0, "expected at least one token");
    assert_eq!(x.len() % tokens, 0, "activation shape mismatch");
    let in_features = x.len() / tokens;
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(out_t.len(), out_features * tokens, "output shape mismatch");

    if !vnni_available() {
        return matmul_i8_auto(quantized, scales, x, tokens, out_t);
    }

    // Quantize every token once, before touching the weights. Each token's
    // own scale and its own sum(xq) correction, both reused across all rows.
    let mut xq = vec![0i8; x.len()];
    let mut x_scales = vec![0.0f32; tokens];
    let mut corrections = vec![0i32; tokens];
    for t in 0..tokens {
        let row = &x[t * in_features..(t + 1) * in_features];
        let (q, scale) = quantize_activations_i8(row);
        corrections[t] = q.iter().map(|v| *v as i32).sum();
        x_scales[t] = scale;
        xq[t * in_features..(t + 1) * in_features].copy_from_slice(&q);
    }

    // Weights outermost: each row is read once and dotted against every
    // token, which is the access pattern the f32 path uses for the same
    // reason -- the weights are what streams from DRAM, not the activations.
    // Rows are independent, so they split across threads with no
    // synchronisation, as in matmul_i8_parallel.
    let rows = |weights: &[i8], row_scales: &[f32], chunk: &mut [f32]| {
        for (row, row_scale) in row_scales.iter().enumerate() {
            let w = &weights[row * in_features..(row + 1) * in_features];
            for t in 0..tokens {
                let activations = &xq[t * in_features..(t + 1) * in_features];
                // SAFETY: vnni_available() checked above; lengths match by the
                // assertions at the top and the slicing here.
                let raw = unsafe { dot_i8_i8_vnni(w, activations) };
                chunk[row * tokens + t] =
                    (raw - 128 * corrections[t]) as f32 * row_scale * x_scales[t];
            }
        }
    };

    let threads = rayon::current_num_threads();
    if threads <= 1 || out_features < threads * 8 {
        rows(quantized, scales, out_t);
        return;
    }

    let rows_per_thread = out_features.div_ceil(threads);
    out_t
        .par_chunks_mut(rows_per_thread * tokens)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let row_start = block_index * rows_per_thread;
            let count = chunk.len() / tokens;
            rows(
                &quantized[row_start * in_features..(row_start + count) * in_features],
                &scales[row_start..row_start + count],
                chunk,
            );
        });
}

/// Whether the VNNI kernel is live, for Python to report.
#[no_mangle]
pub extern "C" fn nanoinfer_has_vnni() -> i32 {
    i32::from(vnni_available())
}

/// VNNI batch matmul over the C ABI; see [`nanoinfer_matmul_i8`] for shapes.
///
/// # Safety
/// As [`nanoinfer_matmul_i8`].
#[no_mangle]
pub unsafe extern "C" fn nanoinfer_matmul_i8_vnni(
    quantized: *const i8,
    scales: *const f32,
    x: *const f32,
    out_t: *mut f32,
    out_features: usize,
    in_features: usize,
    tokens: usize,
) {
    let quantized = std::slice::from_raw_parts(quantized, out_features * in_features);
    let scales = std::slice::from_raw_parts(scales, out_features);
    let x = std::slice::from_raw_parts(x, tokens * in_features);
    let out_t = std::slice::from_raw_parts_mut(out_t, out_features * tokens);
    matmul_i8_vnni(quantized, scales, x, tokens, out_t);
}

// -- block-wise activation scales ------------------------------------------
//
// One scale per activation row is set by that row's largest element, and these
// activations are outlier-heavy: 28.9x the median on the 896-wide inputs and
// 68.3x on the 4864-wide ones, measured on the real model. The other 895 or
// 4863 values then share a range sized for the outlier, which is phase 6's
// per-tensor-versus-per-channel failure happening on the activation side.
//
// Blocking confines each outlier to its own 32 values. Simulated in Python
// before any of this was written, the cost against an fp32 baseline went from
// +5.42% to +0.46%, where llama.cpp's Q8_0 costs +0.87% -- so this is the
// change that makes the engine both faster and more accurate than llama.cpp
// rather than faster and worse.
//
// The arithmetic has to avoid a horizontal sum per block, which would give
// back the speed. With one scale per block the i32 accumulators cannot simply
// run to the end of the row, so instead each block's i32 lanes are converted
// to f32 and fused-multiply-added by that block's scale into an f32
// accumulator, and the 0x80 correction is factored out:
//
//     acc = sum_b (block_dot_b + 128 * block_sum_b) * x_scale_b
//     row = w_scale * (acc - 128 * sum_b block_sum_b * x_scale_b)
//
// The subtracted term depends only on the activations, not on the weight row,
// so it is computed once per call and reused for every row. One horizontal sum
// per row, as before, rather than one per block.

/// Quantize activations to int8 with one scale per `block` values.
///
/// Returns the bytes, one scale per block, and `128 * sum_b(block_sum_b *
/// scale_b)` -- the offset term the kernel subtracts, precomputed here because
/// it is the same for every weight row.
pub fn quantize_activations_blocked(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>, f32) {
    assert!(block > 0, "block size must be positive");
    let blocks = x.len().div_ceil(block);
    let mut values = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; blocks];
    let mut offset = 0.0f32;

    for b in 0..blocks {
        let start = b * block;
        let end = (start + block).min(x.len());
        let group = &x[start..end];

        let magnitude = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if magnitude == 0.0 { 1.0 } else { magnitude / 127.0 };
        scales[b] = scale;

        let mut block_sum = 0i32;
        for i in start..end {
            let q = (x[i] / scale).round().clamp(-127.0, 127.0) as i8;
            values[i] = q;
            block_sum += q as i32;
        }
        offset += block_sum as f32 * scale;
    }

    (values, scales, 128.0 * offset)
}

/// [`quantize_activations_blocked`] writing into caller-owned slices.
///
/// Same arithmetic; it exists so a batch can quantize every token into one
/// flat pair of buffers rather than allocating a Vec per token. The blocked
/// matmul ran 97 times per decoded token, so allocating three Vecs per call
/// was ~300 allocations a token and showed up as a 4.37x spread.
pub fn quantize_activations_blocked_into(
    x: &[f32],
    block: usize,
    values: &mut [i8],
    scales: &mut [f32],
) -> f32 {
    assert!(block > 0, "block size must be positive");
    assert_eq!(values.len(), x.len(), "value buffer must match the input");
    let blocks = x.len().div_ceil(block);
    assert!(scales.len() >= blocks, "scale buffer too small");

    let mut offset = 0.0f32;
    for b in 0..blocks {
        let start = b * block;
        let end = (start + block).min(x.len());

        let magnitude = x[start..end].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if magnitude == 0.0 { 1.0 } else { magnitude / 127.0 };
        scales[b] = scale;

        let mut block_sum = 0i32;
        for i in start..end {
            let q = (x[i] / scale).round().clamp(-127.0, 127.0) as i8;
            values[i] = q;
            block_sum += q as i32;
        }
        offset += block_sum as f32 * scale;
    }
    128.0 * offset
}

/// int8 matvec with block-wise int8 activations, through VPDPBUSD.
///
/// `block` is the number of activation values sharing a scale; 32 is the size
/// llama.cpp's Q8_0 and Q8_1 use, and the size where this model's error stops
/// improving -- 16 measured no better than 32.
///
/// Falls back to [`matvec_i8_auto`] without VNNI.
pub fn matvec_i8_vnni_blocked(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    block: usize,
    out: &mut [f32],
) {
    let in_features = x.len();
    let out_features = out.len();
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(scales.len(), out_features, "expected one scale per output row");

    #[cfg(target_arch = "x86_64")]
    {
        if vnni_available() {
            let (xq, x_scales, offset) = quantize_activations_blocked(x, block);
            for row in 0..out_features {
                let start = row * in_features;
                // SAFETY: feature-checked above; lengths come from the
                // assertions and the slicing here.
                let acc = unsafe {
                    dot_i8_blocked_vnni(
                        &quantized[start..start + in_features],
                        &xq,
                        &x_scales,
                        block,
                    )
                };
                out[row] = (acc - offset) * scales[row];
            }
            return;
        }
    }
    let _ = block;
    matvec_i8_auto(quantized, scales, x, out)
}

/// `sum_b (block_dot_b + 128 * block_sum_b) * scale_b`, as f32.
///
/// The caller subtracts the precomputed offset and multiplies by the row's
/// weight scale.
///
/// # Safety
/// AVX2, FMA and AVX-VNNI must be available; `w.len() == xq.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avxvnni")]
#[inline]
unsafe fn dot_i8_blocked_vnni(w: &[i8], xq: &[i8], x_scales: &[f32], block: usize) -> f32 {
    use std::arch::x86_64::*;

    let in_features = w.len();
    let wp = w.as_ptr();
    let xp = xq.as_ptr();
    let sign_flip = _mm256_set1_epi8(-128i8); // 0x80

    let mut acc = _mm256_setzero_ps();
    let mut offset = 0;

    // Whole 32-byte groups. One dpbusd, one convert, one fmadd each: three
    // instructions more per 32 bytes than the per-row kernel, against the
    // eight-at-a-time float widening this replaced.
    while offset + 32 <= in_features && block >= 32 {
        _mm_prefetch::<_MM_HINT_T0>(wp.add(offset + PREFETCH_BYTES) as *const i8);

        let wv = _mm256_xor_si256(
            _mm256_loadu_si256(wp.add(offset) as *const __m256i),
            sign_flip,
        );
        let xv = _mm256_loadu_si256(xp.add(offset) as *const __m256i);
        let products = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wv, xv);

        // The block index comes from the offset, not from a running counter
        // scaled by the block size. Incrementing by a ratio was wrong for any
        // block wider than 32: with block = 128 it walked a one-element scale
        // array four times over and read whatever followed it.
        let b = offset / block;
        debug_assert!(b < x_scales.len(), "block index past the scale array");
        let scale = _mm256_set1_ps(*x_scales.get_unchecked(b));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(products), scale, acc);

        offset += 32;
    }

    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let mut sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    // Whatever is left: a block narrower than 32, or a ragged tail. Same
    // arithmetic in scalar form, including the 0x80 offset, so the caller's
    // precomputed correction still covers the whole row.
    while offset < in_features {
        let b_index = offset / block;
        let scale = *x_scales.get_unchecked(b_index);
        let end = (offset + block.min(in_features - offset)).min(in_features);
        let mut partial = 0i32;
        for i in offset..end {
            let unsigned = (*wp.add(i) as i32) + 128;
            partial += unsigned * (*xp.add(i) as i32);
        }
        total += partial as f32 * scale;
        offset = end;
    }
    total
}

/// int8 weights times `tokens` activation vectors, block-wise through VPDPBUSD.
///
/// Shapes as [`matmul_i8_auto`]. Activations are quantized once, up front, into
/// flat buffers shared read-only by every worker; the rows are then split
/// across threads the way [`matmul_i8_parallel`] splits them, since a row is
/// computed entirely by one thread and needs no synchronisation.
///
/// Both of those were missing from the first version, and together they were
/// worth more than the kernel: it allocated three Vecs per call and ran the row
/// loop on one core while the float-activation path it was being compared
/// against used all of them.
pub fn matmul_i8_vnni_blocked(
    quantized: &[i8],
    scales: &[f32],
    x: &[f32],
    tokens: usize,
    block: usize,
    out_t: &mut [f32],
) {
    let out_features = scales.len();
    assert!(tokens > 0, "expected at least one token");
    assert_eq!(x.len() % tokens, 0, "activation shape mismatch");
    let in_features = x.len() / tokens;
    assert_eq!(quantized.len(), out_features * in_features, "weight shape mismatch");
    assert_eq!(out_t.len(), out_features * tokens, "output shape mismatch");

    if !vnni_available() {
        return matmul_i8_auto(quantized, scales, x, tokens, out_t);
    }

    // Quantize every token once. Three allocations for the whole call rather
    // than three per token, and the result is read-only from here on.
    let per_token_blocks = in_features.div_ceil(block);
    let mut xq = vec![0i8; tokens * in_features];
    let mut x_scales = vec![0.0f32; tokens * per_token_blocks];
    let mut offsets = vec![0.0f32; tokens];
    for t in 0..tokens {
        offsets[t] = quantize_activations_blocked_into(
            &x[t * in_features..(t + 1) * in_features],
            block,
            &mut xq[t * in_features..(t + 1) * in_features],
            &mut x_scales[t * per_token_blocks..(t + 1) * per_token_blocks],
        );
    }

    let rows = |weights: &[i8], row_scales: &[f32], chunk: &mut [f32]| {
        for (row, row_scale) in row_scales.iter().enumerate() {
            let w = &weights[row * in_features..(row + 1) * in_features];
            for t in 0..tokens {
                let activations = &xq[t * in_features..(t + 1) * in_features];
                let block_scales =
                    &x_scales[t * per_token_blocks..(t + 1) * per_token_blocks];
                // SAFETY: vnni_available() checked above; lengths match by the
                // assertions and the slicing here.
                let acc = unsafe {
                    dot_i8_blocked_vnni(w, activations, block_scales, block)
                };
                chunk[row * tokens + t] = (acc - offsets[t]) * row_scale;
            }
        }
    };

    let threads = rayon::current_num_threads();
    if threads <= 1 || out_features < threads * 8 {
        rows(quantized, scales, out_t);
        return;
    }

    let rows_per_thread = out_features.div_ceil(threads);
    out_t
        .par_chunks_mut(rows_per_thread * tokens)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let row_start = block_index * rows_per_thread;
            let count = chunk.len() / tokens;
            rows(
                &quantized[row_start * in_features..(row_start + count) * in_features],
                &scales[row_start..row_start + count],
                chunk,
            );
        });
}

/// Block-wise VNNI batch matmul over the C ABI.
///
/// # Safety
/// As [`nanoinfer_matmul_i8`]. `block` must be positive.
#[no_mangle]
pub unsafe extern "C" fn nanoinfer_matmul_i8_vnni_blocked(
    quantized: *const i8,
    scales: *const f32,
    x: *const f32,
    out_t: *mut f32,
    out_features: usize,
    in_features: usize,
    tokens: usize,
    block: usize,
) {
    let quantized = std::slice::from_raw_parts(quantized, out_features * in_features);
    let scales = std::slice::from_raw_parts(scales, out_features);
    let x = std::slice::from_raw_parts(x, tokens * in_features);
    let out_t = std::slice::from_raw_parts_mut(out_t, out_features * tokens);
    matmul_i8_vnni_blocked(quantized, scales, x, tokens, block, out_t);
}
