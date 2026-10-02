use nanoinfer_kernels::{
    matmul_i8_auto, matmul_i8_parallel, matvec_i8_auto, pseudo_random, quantize_rows_i8,
};

/// Build `tokens` activation rows and the int8 weights for one shape.
fn case(out_features: usize, in_features: usize, tokens: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let weights = pseudo_random(out_features * in_features, 11);
    let (quantized, scales) = quantize_rows_i8(&weights, out_features);
    let x = pseudo_random(tokens * in_features, 12);
    (quantized, scales, x)
}

/// The batched kernel must be the matvec, run once per token -- bitwise, not
/// approximately. Both call the same per-row dot product, so any difference
/// at all means the batching changed which numbers get added in which order.
#[test]
fn a_batch_is_bitwise_a_loop_of_matvecs() {
    for &(out_features, in_features, tokens) in
        &[(4, 8, 1), (7, 33, 3), (128, 896, 5), (4864, 896, 9), (896, 4864, 2)]
    {
        let (quantized, scales, x) = case(out_features, in_features, tokens);

        let mut out_t = vec![0.0; out_features * tokens];
        matmul_i8_parallel(&quantized, &scales, &x, tokens, &mut out_t);

        for t in 0..tokens {
            let mut expected = vec![0.0; out_features];
            matvec_i8_auto(
                &quantized,
                &scales,
                &x[t * in_features..(t + 1) * in_features],
                &mut expected,
            );
            for row in 0..out_features {
                assert_eq!(
                    out_t[row * tokens + t].to_bits(),
                    expected[row].to_bits(),
                    "{out_features}x{in_features}, token {t}, row {row}"
                );
            }
        }
    }
}

#[test]
fn threaded_and_single_batches_are_bitwise_identical() {
    let (quantized, scales, x) = case(4864, 896, 6);
    let mut pooled = vec![0.0; 4864 * 6];
    let mut single = vec![0.0; 4864 * 6];

    matmul_i8_parallel(&quantized, &scales, &x, 6, &mut pooled);
    matmul_i8_auto(&quantized, &scales, &x, 6, &mut single);

    assert!(pooled.iter().zip(&single).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// A block boundary that does not divide the rows evenly is where an
/// off-by-one in the chunk arithmetic would write a token into its
/// neighbour's slot.
#[test]
fn uneven_row_blocks_land_in_the_right_place() {
    let (quantized, scales, x) = case(1001, 64, 3);
    let mut pooled = vec![f32::NAN; 1001 * 3];
    let mut single = vec![0.0; 1001 * 3];

    matmul_i8_parallel(&quantized, &scales, &x, 3, &mut pooled);
    matmul_i8_auto(&quantized, &scales, &x, 3, &mut single);

    assert!(pooled.iter().all(|v| v.is_finite()), "a slot was never written");
    assert_eq!(pooled, single);
}

#[test]
#[should_panic(expected = "output shape mismatch")]
fn rejects_an_output_of_the_wrong_size() {
    let (quantized, scales, x) = case(4, 8, 2);
    let mut out_t = vec![0.0; 4];
    matmul_i8_auto(&quantized, &scales, &x, 2, &mut out_t);
}
