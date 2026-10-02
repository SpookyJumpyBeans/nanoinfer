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

    for row in 0..out_features {
        let w = &quantized[row * in_features..(row + 1) * in_features];
        let scale = scales[row];
        for t in 0..tokens {
            let xt = &x[t * in_features..(t + 1) * in_features];
            #[cfg(target_arch = "x86_64")]
            let dot = if simd {
                // SAFETY: feature-checked above; slices are equal length.
                unsafe { dot_i8_avx2(w, xt) }
            } else {
                dot_i8_scalar(w, xt)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let dot = {
                let _ = simd;
                dot_i8_scalar(w, xt)
            };
            out_t[row * tokens + t] = dot * scale;
        }
    }
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
