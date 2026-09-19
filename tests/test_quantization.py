"""Tests for weight quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.quantization import (
    INT8_MAX,
    quantization_error,
    quantize_int8,
    round_half_away_from_zero,
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
