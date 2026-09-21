"""Tests for the Rust kernel bridge.

These are the tests that matter most in phase 7. Every other kernel test lives
in Rust and checks the arithmetic; these check the *boundary*, where a wrong
dtype or a non-contiguous view is read as raw bytes and nothing complains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer import kernels
from nanoinfer.quantization import quantize_int8

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()

needs_kernels = pytest.mark.skipif(
    not kernels.available(), reason="rust kernels not built"
)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# -- loading ---------------------------------------------------------------


def test_describe_never_raises_even_without_a_library():
    """The engine asks this before it knows whether anything was built."""
    report = kernels.describe()
    assert set(report) == {"available", "path", "avx2", "threads"}
    assert isinstance(report["available"], bool)


def test_calling_without_a_library_says_how_to_build_it(monkeypatch):
    monkeypatch.setattr(kernels, "_LIBRARY", None)
    with pytest.raises(RuntimeError, match="cargo build --release"):
        kernels.matvec_i8(
            np.zeros((4, 8), np.int8), np.ones(4, np.float32), np.ones(8, np.float32)
        )


# -- the numbers -----------------------------------------------------------


@needs_kernels
def test_agrees_with_numpy(rng):
    for out_features, in_features in [(4, 8), (64, 896), (896, 4864), (7, 33)]:
        weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
        q = quantize_int8(weights)
        x = rng.standard_normal(in_features).astype(np.float32)

        expected = kernels.matvec_i8_numpy(q.values, q.scales, x)
        actual = kernels.matvec_i8(q.values, q.scales, x)

        # Rust accumulates in i32 and scales once at the end; NumPy widens to
        # float32 first. Same arithmetic, different rounding, so this is
        # allclose rather than exact -- with a tolerance set by float32, not
        # by what happened to pass.
        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-4,
            err_msg=f"{out_features}x{in_features}",
        )


@needs_kernels
def test_threaded_and_single_are_bitwise_identical(rng):
    """Each output row is computed by exactly one thread, so this is exact."""
    weights = rng.standard_normal((4864, 896)).astype(np.float32)
    q = quantize_int8(weights)
    x = rng.standard_normal(896).astype(np.float32)

    np.testing.assert_array_equal(
        kernels.matvec_i8(q.values, q.scales, x, threaded=True),
        kernels.matvec_i8(q.values, q.scales, x, threaded=False),
    )


@needs_kernels
def test_returns_float32_of_the_right_length(rng):
    q = quantize_int8(rng.standard_normal((37, 64)).astype(np.float32))
    out = kernels.matvec_i8(q.values, q.scales, rng.standard_normal(64).astype(np.float32))
    assert out.shape == (37,)
    assert out.dtype == np.float32


@needs_kernels
def test_a_zero_vector_gives_zeros(rng):
    q = quantize_int8(rng.standard_normal((16, 32)).astype(np.float32))
    out = kernels.matvec_i8(q.values, q.scales, np.zeros(32, np.float32))
    np.testing.assert_array_equal(out, np.zeros(16, np.float32))


# -- the boundary ----------------------------------------------------------


@needs_kernels
def test_a_transposed_view_is_not_read_as_contiguous(rng):
    """The failure this guards against is silent, so it is worth a test.

    A transposed array has the right shape and the wrong memory order. Rust
    reads out*in elements straight off the pointer, so without the copy in
    _as_kernel_input this would return plausible, wrong numbers.
    """
    weights = rng.standard_normal((64, 32)).astype(np.float32)
    q = quantize_int8(weights)
    transposed = np.ascontiguousarray(q.values.T).T      # same values, F-order
    assert not transposed.flags["C_CONTIGUOUS"]

    x = rng.standard_normal(32).astype(np.float32)
    np.testing.assert_array_equal(
        kernels.matvec_i8(transposed, q.scales, x),
        kernels.matvec_i8(q.values, q.scales, x),
    )


@needs_kernels
def test_rejects_a_mismatched_scale_count(rng):
    q = quantize_int8(rng.standard_normal((16, 32)).astype(np.float32))
    with pytest.raises(ValueError, match="one scale per output row"):
        kernels.matvec_i8(q.values, q.scales[:8], np.ones(32, np.float32))


@needs_kernels
def test_rejects_a_mismatched_input_width(rng):
    q = quantize_int8(rng.standard_normal((16, 32)).astype(np.float32))
    with pytest.raises(ValueError, match="weights expect 32"):
        kernels.matvec_i8(q.values, q.scales, np.ones(31, np.float32))


@needs_kernels
def test_rejects_a_non_matrix_weight():
    with pytest.raises(ValueError, match="2-D"):
        kernels.matvec_i8(
            np.zeros(8, np.int8), np.ones(1, np.float32), np.ones(8, np.float32)
        )


@needs_kernels
def test_a_float64_input_vector_is_narrowed_not_reinterpreted(rng):
    """float64 x would be read as twice as many float32s without the coercion."""
    q = quantize_int8(rng.standard_normal((16, 32)).astype(np.float32))
    x = rng.standard_normal(32)
    assert x.dtype == np.float64

    np.testing.assert_allclose(
        kernels.matvec_i8(q.values, q.scales, x),
        kernels.matvec_i8(q.values, q.scales, x.astype(np.float32)),
        rtol=1e-6,
    )


# -- against the real weights ----------------------------------------------


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@needs_kernels
@pytest.mark.slow
def test_real_projections_track_the_float32_engine():
    """The kernel against what the engine actually computes, end to end.

    Not against the NumPy int8 path -- against the float32 matmul the engine
    really runs, so this measures quantization error plus kernel error
    together. If either were wrong this is where it would show.
    """
    from nanoinfer.weights import ModelWeights

    weights = ModelWeights.load(MODEL_DIR)
    layer = weights.layers[0]
    rng = np.random.default_rng(11)

    for name, tensor in (
        ("q_proj", layer.q_proj_weight),
        ("gate_proj", layer.gate_proj_weight),
        ("down_proj", layer.down_proj_weight),
    ):
        x = rng.standard_normal(tensor.shape[1]).astype(np.float32)
        reference = tensor.astype(np.float32) @ x

        q = quantize_int8(tensor)
        actual = kernels.matvec_i8(q.values, q.scales, x)

        relative = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert relative < 0.02, f"{name}: relative error {relative:.4f}"
