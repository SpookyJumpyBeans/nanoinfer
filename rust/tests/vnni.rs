//! Tests for the AVX-VNNI int8 kernel.
//!
//! The existing kernel holds weights as int8 and activations as float32, so
//! its dot product runs in floating point. VNNI's VPDPBUSD wants both operands
//! as bytes, which means quantizing the activations too -- so this kernel is
//! not bit-comparable with the float-activation one, and these tests pin the
//! error it is allowed to have rather than demanding equality.
//!
//! The arithmetic that needs pinning most is the sign correction. VPDPBUSD
//! multiplies *unsigned* by *signed*, so the weights are XORed with 0x80 on
//! load to reinterpret i8 as u8, which adds 128 to every weight:
//!
//!     dpbusd(w ^ 0x80, xq) = sum (w + 128) * xq = sum (w * xq) + 128 * sum xq
//!
//! and the 128 * sum(xq) term is subtracted back. Get that wrong and results
//! are plausible but off, which is the failure mode worth a test of its own.

use nanoinfer_kernels::{
    matvec_i8_auto, matvec_i8_vnni, pseudo_random, quantize_activations_i8, quantize_rows_i8,
    vnni_available,
};

/// The shapes a decode step actually runs, plus awkward ones.
const SHAPES: &[(usize, usize)] = &[
    (896, 896),    // q_proj, o_proj
    (128, 896),    // k_proj, v_proj
    (4864, 896),   // gate, up
    (896, 4864),   // down
    (7, 33),       // neither dimension a multiple of anything
    (3, 32),       // exactly one vector wide
    (5, 31),       // one short of a vector
];

fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((*x - *y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
fn vnni_is_available_on_this_machine_or_the_tests_are_vacuous() {
    // Not an assertion about the kernel: a note in the output so a pass on a
    // machine without VNNI cannot be mistaken for a pass of the VNNI path.
    if !vnni_available() {
        eprintln!("note: no AVX-VNNI here; matvec_i8_vnni runs its fallback");
    }
}

#[test]
fn tracks_the_float_activation_kernel_on_every_decode_shape() {
    for &(out_features, in_features) in SHAPES {
        let weights = pseudo_random(out_features * in_features, 11);
        let x = pseudo_random(in_features, 12);
        let (quantized, scales) = quantize_rows_i8(&weights, out_features);

        let mut reference = vec![0.0f32; out_features];
        matvec_i8_auto(&quantized, &scales, &x, &mut reference);

        let mut actual = vec![0.0f32; out_features];
        matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

        // Activations now carry 8 bits instead of 32. The error is the
        // quantization of x, not a mistake in the kernel: one part in ~127
        // per element, partly cancelling across a long dot product.
        let error = relative_error(&actual, &reference);
        assert!(
            error < 0.02,
            "{out_features}x{in_features}: relative error {error} too large"
        );
    }
}

#[test]
fn the_sign_correction_is_right_for_negative_weights() {
    // All-negative weights against all-positive activations: if the 128 *
    // sum(xq) term were dropped or mis-signed, the result would come out
    // positive, or far too large, rather than merely imprecise.
    let in_features = 64;
    let weights: Vec<f32> = (0..in_features).map(|_| -1.0).collect();
    let x: Vec<f32> = (0..in_features).map(|_| 1.0).collect();
    let (quantized, scales) = quantize_rows_i8(&weights, 1);

    let mut actual = vec![0.0f32; 1];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    // -1 * 1, sixty-four times.
    assert!(
        (actual[0] + 64.0).abs() < 0.5,
        "expected about -64, got {}",
        actual[0]
    );
}

#[test]
fn the_sign_correction_is_right_for_negative_activations() {
    let in_features = 64;
    let weights: Vec<f32> = (0..in_features).map(|_| 1.0).collect();
    let x: Vec<f32> = (0..in_features).map(|_| -1.0).collect();
    let (quantized, scales) = quantize_rows_i8(&weights, 1);

    let mut actual = vec![0.0f32; 1];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    assert!(
        (actual[0] + 64.0).abs() < 0.5,
        "expected about -64, got {}",
        actual[0]
    );
}

#[test]
fn mixed_signs_cancel_the_way_they_should() {
    // Alternating signs on both sides: every product is +1, so a sign error
    // in the correction shows up as a wrong magnitude immediately.
    let in_features = 64;
    let weights: Vec<f32> = (0..in_features)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let x = weights.clone();
    let (quantized, scales) = quantize_rows_i8(&weights, 1);

    let mut actual = vec![0.0f32; 1];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    assert!(
        (actual[0] - 64.0).abs() < 0.5,
        "expected about 64, got {}",
        actual[0]
    );
}

#[test]
fn a_hand_worked_example() {
    // weights row = [1, 2, 3, 4], x = [1, 1, 1, 1]
    // Quantized per row: scale = 4/127, values = [32, 64, 95, 127]
    // Reconstructed dot = (32 + 64 + 95 + 127) * (4/127) * 1 = 318 * 4/127
    let weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let x = vec![1.0f32, 1.0, 1.0, 1.0];
    let (quantized, scales) = quantize_rows_i8(&weights, 1);
    assert_eq!(quantized, vec![32i8, 64, 95, 127]);

    let mut actual = vec![0.0f32; 1];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    let expected = 318.0 * scales[0];
    assert!(
        (actual[0] - expected).abs() < 0.05,
        "expected about {expected}, got {}",
        actual[0]
    );
}

#[test]
fn a_tail_shorter_than_one_vector_is_not_dropped() {
    // 31 is one short of the 32-byte step, so the whole row lands in the tail
    // path. A kernel that skipped it would return zero.
    let in_features = 31;
    let weights: Vec<f32> = (0..in_features).map(|_| 2.0).collect();
    let x: Vec<f32> = (0..in_features).map(|_| 1.0).collect();
    let (quantized, scales) = quantize_rows_i8(&weights, 1);

    let mut actual = vec![0.0f32; 1];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    assert!(actual[0] > 60.0, "tail dropped: got {}", actual[0]);
}

#[test]
fn an_all_zero_activation_vector_gives_zeros() {
    // The scale of an all-zero vector would divide by zero if unguarded.
    let weights = pseudo_random(16 * 64, 3);
    let x = vec![0.0f32; 64];
    let (quantized, scales) = quantize_rows_i8(&weights, 16);

    let mut actual = vec![0.0f32; 16];
    matvec_i8_vnni(&quantized, &scales, &x, &mut actual);

    assert_eq!(actual, vec![0.0f32; 16]);
}

#[test]
fn activation_quantization_round_trips_within_one_step() {
    let x = pseudo_random(256, 7);
    let (xq, scale) = quantize_activations_i8(&x);

    assert_eq!(xq.len(), x.len());
    let largest = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(xq.iter().any(|v| v.abs() == 127), "range is not used fully");

    for (original, quantized) in x.iter().zip(xq.iter()) {
        let restored = *quantized as f32 * scale;
        assert!(
            (restored - original).abs() <= largest / 127.0,
            "{original} restored as {restored}"
        );
    }
}
