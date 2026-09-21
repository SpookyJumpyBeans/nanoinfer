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
