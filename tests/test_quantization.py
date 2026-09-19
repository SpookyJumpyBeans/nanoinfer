"""Tests for weight quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.quantization import (
    INT8_MAX,
    pack_nibbles,
    quantization_error,
    quantize_int4,
    quantize_int8,
    round_half_away_from_zero,
    unpack_nibbles,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# -- rounding --------------------------------------------------------------


def test_rounds_half_away_from_zero_not_to_even():
    """np.round would send 0.5 -> 0 and 2.5 -> 2, biasing small magnitudes."""
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    np.testing.assert_array_equal(
        round_half_away_from_zero(x), [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
    )
    assert not np.array_equal(round_half_away_from_zero(x), np.round(x))


def test_rounding_is_symmetric(rng):
    x = rng.standard_normal(1000) * 10
    np.testing.assert_array_equal(
        round_half_away_from_zero(-x), -round_half_away_from_zero(x)
    )


# -- the quantizer ---------------------------------------------------------


def test_shapes_and_dtypes(rng):
    weights = rng.standard_normal((16, 64)).astype(np.float32)
    q = quantize_int8(weights)

    assert q.values.dtype == np.int8
    assert q.values.shape == (16, 64)
    assert q.scales.dtype == np.float32
    assert q.scales.shape == (16,)
    assert q.bits == 8


def test_values_stay_inside_the_symmetric_range(rng):
    weights = rng.standard_normal((32, 128)).astype(np.float32) * 100
    q = quantize_int8(weights)
    assert q.values.min() >= -INT8_MAX
    assert q.values.max() <= INT8_MAX


def test_the_largest_weight_in_a_row_maps_to_the_endpoint():
    weights = np.array([[1.0, -4.0, 2.0]], dtype=np.float32)
    q = quantize_int8(weights)
    assert abs(q.values[0]).max() == INT8_MAX


def test_each_row_gets_its_own_scale():
    """A tiny row must not be crushed by a large one elsewhere."""
    weights = np.array([[1000.0, -1000.0], [0.001, -0.001]], dtype=np.float32)
    q = quantize_int8(weights)

    assert q.scales[0] > q.scales[1]
    restored = q.dequantize()
    np.testing.assert_allclose(restored[1], weights[1], rtol=1e-5)


def test_per_channel_beats_per_tensor(rng):
    """Quantified, because it is the main design decision in this module."""
    weights = rng.standard_normal((64, 256)).astype(np.float32)
    weights[0] *= 200                       # one outlier row

    per_channel = quantize_int8(weights).dequantize()

    scale = np.abs(weights).max() / INT8_MAX
    per_tensor = (
        np.clip(round_half_away_from_zero(weights / scale), -INT8_MAX, INT8_MAX) * scale
    ).astype(np.float32)

    channel_error = quantization_error(weights, per_channel)["rel_frobenius"]
    tensor_error = quantization_error(weights, per_tensor)["rel_frobenius"]
    assert channel_error < tensor_error / 5


def test_round_trip_error_is_small(rng):
    weights = rng.standard_normal((128, 512)).astype(np.float32)
    error = quantization_error(weights, quantize_int8(weights).dequantize())
    assert error["rel_frobenius"] < 0.02


def test_negation_is_exact(rng):
    """Symmetric range: a weight and its negation quantize to exact opposites."""
    weights = rng.standard_normal((8, 32)).astype(np.float32)
    np.testing.assert_array_equal(
        quantize_int8(-weights).values, -quantize_int8(weights).values
    )


def test_values_already_on_the_grid_survive_exactly():
    """If the weights are already representable, quantizing loses nothing."""
    scale = 0.01
    weights = (np.arange(-127, 128, dtype=np.float32) * scale).reshape(1, -1)
    restored = quantize_int8(weights).dequantize()
    np.testing.assert_allclose(restored, weights, rtol=1e-5, atol=1e-8)


def test_all_zero_row_does_not_divide_by_zero():
    weights = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        q = quantize_int8(weights)
    np.testing.assert_array_equal(q.dequantize()[0], [0.0, 0.0, 0.0])


def test_scales_are_positive(rng):
    weights = rng.standard_normal((16, 16)).astype(np.float32)
    assert np.all(quantize_int8(weights).scales > 0)


def test_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2-D"):
        quantize_int8(np.zeros(8, dtype=np.float32))


def test_storage_is_four_times_smaller(rng):
    weights = rng.standard_normal((256, 1024)).astype(np.float32)
    q = quantize_int8(weights)
    # int8 values plus one float32 scale per row.
    assert q.nbytes < weights.nbytes / 3.9


# -- on the real weights ---------------------------------------------------


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_real_weights_quantize_within_two_percent():
    from nanoinfer.weights import ModelWeights

    weights = ModelWeights.load(MODEL_DIR)
    for name, tensor in (
        ("q_proj", weights.layers[0].q_proj_weight),
        ("gate_proj", weights.layers[0].gate_proj_weight),
        ("down_proj", weights.layers[0].down_proj_weight),
        ("embed_tokens", weights.embed_tokens),
    ):
        error = quantization_error(tensor, quantize_int8(tensor).dequantize())
        assert error["rel_frobenius"] < 0.02, f"{name}: {error}"


# -- INT4 ------------------------------------------------------------------


def test_nibble_packing_round_trips(rng):
    values = rng.integers(-7, 8, size=(8, 64)).astype(np.int8)
    np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(values)), values)


def test_packing_halves_the_width():
    values = np.zeros((4, 64), dtype=np.int8)
    assert pack_nibbles(values).shape == (4, 32)
    assert pack_nibbles(values).dtype == np.uint8


def test_packing_covers_the_whole_signed_range():
    values = np.arange(-7, 8, dtype=np.int8).reshape(1, -1)
    values = np.pad(values, ((0, 0), (0, 1)))     # even width for packing
    np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(values)), values)


def test_int4_shape_survives_padding(rng):
    """An input width that is not a multiple of the group must still restore."""
    weights = rng.standard_normal((4, 100)).astype(np.float32)
    q = quantize_int4(weights, group_size=32)     # 100 is not a multiple of 32
    assert q.dequantize().shape == weights.shape


def test_int4_stores_about_eight_times_smaller(rng):
    weights = rng.standard_normal((256, 1024)).astype(np.float32)
    q = quantize_int4(weights, group_size=128)
    assert 7.0 < weights.nbytes / q.nbytes < 8.0


def test_int4_smaller_groups_are_more_accurate(rng):
    """Fifteen levels is coarse, so how local the scale is matters a lot."""
    weights = rng.standard_normal((32, 512)).astype(np.float32)
    coarse = quantization_error(weights, quantize_int4(weights, 256).dequantize())
    fine = quantization_error(weights, quantize_int4(weights, 32).dequantize())
    assert fine["rel_frobenius"] < coarse["rel_frobenius"]


def test_int4_is_worse_than_int8(rng):
    """Stated explicitly: the compression is not free."""
    weights = rng.standard_normal((64, 512)).astype(np.float32)
    int8 = quantization_error(weights, quantize_int8(weights).dequantize())
    int4 = quantization_error(weights, quantize_int4(weights).dequantize())
    assert int4["rel_frobenius"] > int8["rel_frobenius"] * 5


def test_int4_rejects_an_odd_group_size():
    with pytest.raises(ValueError, match="even"):
        quantize_int4(np.zeros((2, 8), np.float32), group_size=33)


def test_int4_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2-D"):
        quantize_int4(np.zeros(8, dtype=np.float32))


def test_int4_all_zero_group_does_not_divide_by_zero():
    weights = np.zeros((2, 64), dtype=np.float32)
    weights[1] = 1.0
    with np.errstate(divide="raise", invalid="raise"):
        restored = quantize_int4(weights, group_size=32).dequantize()
    np.testing.assert_array_equal(restored[0], np.zeros(64, dtype=np.float32))


# -- whole-model quantization ----------------------------------------------


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_quantize_model_leaves_norms_and_biases_alone():
    from nanoinfer.quantization import quantize_model
    from nanoinfer.weights import ModelWeights

    original = ModelWeights.load(MODEL_DIR)
    quantized, report = quantize_model(original, bits=8)

    # 24 layers x 7 linear tensors.
    assert report.tensors == 24 * 7
    np.testing.assert_array_equal(
        quantized.layers[0].input_layernorm, original.layers[0].input_layernorm
    )
    np.testing.assert_array_equal(
        quantized.layers[0].q_proj_bias, original.layers[0].q_proj_bias
    )


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_quantize_model_leaves_embeddings_alone_by_default():
    from nanoinfer.quantization import quantize_model
    from nanoinfer.weights import ModelWeights

    original = ModelWeights.load(MODEL_DIR)
    quantized, report = quantize_model(original, bits=8)

    assert not report.embeddings_quantized
    np.testing.assert_array_equal(quantized.embed_tokens, original.embed_tokens)


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_group_size_actually_reaches_the_quantizer():
    """Regression: the flag was accepted and silently ignored.

    Both group sizes produced byte-identical results, which is how the bug
    surfaced -- the measurement sweep reported the same perplexity twice.
    """
    from nanoinfer.quantization import quantize_model
    from nanoinfer.weights import ModelWeights

    original = ModelWeights.load(MODEL_DIR)
    _, coarse = quantize_model(original, bits=4, group_size=128)
    _, fine = quantize_model(original, bits=4, group_size=32)
    assert fine.quantized_bytes > coarse.quantized_bytes


def test_quantize_model_rejects_unsupported_widths():
    from nanoinfer.quantization import quantize_model

    with pytest.raises(ValueError, match="only INT4 and INT8"):
        quantize_model(None, bits=16)
