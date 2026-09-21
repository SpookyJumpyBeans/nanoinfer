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
