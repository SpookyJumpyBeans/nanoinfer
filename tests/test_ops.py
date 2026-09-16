"""Tests for the forward-pass primitives, against closed forms and against torch.

Each primitive is checked twice: once against a definition written out
independently in the test, and once against the reference implementation where
one exists. The first catches a misreading of the maths; the second catches a
misreading of what the reference actually does, which is the failure mode that
produces a model that runs and is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoinfer.ops import causal_mask, repeat_kv, rms_norm, sigmoid, silu, softmax

torch = pytest.importorskip("torch", reason="reference oracle not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# -- rms_norm --------------------------------------------------------------


def test_rms_norm_matches_the_written_out_definition(rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    eps = 1e-6

    expected = np.empty_like(x)
    for i, row in enumerate(x):
        ms = sum(float(v) ** 2 for v in row) / len(row)
        expected[i] = [float(v) / np.sqrt(ms + eps) * float(g) for v, g in zip(row, w)]

    np.testing.assert_allclose(rms_norm(x, w, eps), expected, rtol=1e-5, atol=1e-6)


def test_rms_norm_matches_torch(rng):
    x = rng.standard_normal((6, 16)).astype(np.float32)
    w = rng.standard_normal(16).astype(np.float32)
    eps = 1e-6

    xt = torch.from_numpy(x)
    variance = xt.pow(2).mean(-1, keepdim=True)
    expected = (torch.from_numpy(w) * (xt * torch.rsqrt(variance + eps))).numpy()

    np.testing.assert_allclose(rms_norm(x, w, eps), expected, rtol=1e-5, atol=1e-6)


def test_rms_norm_does_not_subtract_the_mean(rng):
    """RMSNorm is not LayerNorm: a constant offset must survive it."""
    x = np.full((1, 8), 3.0, dtype=np.float32)
    w = np.ones(8, dtype=np.float32)
    out = rms_norm(x, w, 1e-6)
    assert np.all(out > 0.99), "a constant row must not normalize to zero"


def test_rms_norm_epsilon_is_inside_the_square_root():
    """eps is added to the mean square, not to its root."""
    x = np.zeros((1, 4), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    eps = 1e-6
    # With x all zero, mean square is 0, so the scale is 1/sqrt(eps).
    # If eps were added outside, the scale would be 1/eps instead.
    assert np.allclose(rms_norm(x, w, eps), 0.0)
    x = np.full((1, 4), 1e-8, dtype=np.float32)
    scale = float(rms_norm(x, w, eps)[0, 0] / x[0, 0])
    assert abs(scale - 1.0 / np.sqrt(eps)) < abs(scale - 1.0 / eps)


def test_rms_norm_applies_the_gain(rng):
    x = rng.standard_normal((3, 5)).astype(np.float32)
    ones = rms_norm(x, np.ones(5, dtype=np.float32), 1e-6)
    doubled = rms_norm(x, np.full(5, 2.0, dtype=np.float32), 1e-6)
    np.testing.assert_allclose(doubled, ones * 2, rtol=1e-6)


def test_rms_norm_rejects_mismatched_weight():
    with pytest.raises(ValueError, match="last dim"):
        rms_norm(np.zeros((2, 8), np.float32), np.zeros(4, np.float32), 1e-6)


def test_rms_norm_handles_3d_input(rng):
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    assert rms_norm(x, w, 1e-6).shape == (2, 3, 8)


# -- sigmoid and silu ------------------------------------------------------


def test_sigmoid_matches_torch(rng):
    x = rng.standard_normal(1000).astype(np.float32) * 10
    np.testing.assert_allclose(
        sigmoid(x), torch.sigmoid(torch.from_numpy(x)).numpy(), rtol=1e-6, atol=1e-7
    )


def test_sigmoid_does_not_overflow_on_large_negatives():
    """The textbook form raises a warning here; this one must not."""
    x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_silu_matches_torch(rng):
    x = rng.standard_normal(1000).astype(np.float32) * 10
    np.testing.assert_allclose(
        silu(x), torch.nn.functional.silu(torch.from_numpy(x)).numpy(), rtol=1e-6, atol=1e-6
    )


def test_silu_is_not_relu(rng):
    """SiLU dips below zero for small negative x; ReLU does not."""
    x = np.array([-1.0], dtype=np.float32)
    assert silu(x)[0] < 0


def test_silu_at_zero_is_zero():
    assert silu(np.array([0.0], dtype=np.float32))[0] == 0.0


def test_silu_handles_extremes():
    x = np.array([-1e4, 1e4], dtype=np.float32)
    out = silu(x)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-3)
    assert out[1] == pytest.approx(1e4, rel=1e-5)


# -- softmax ---------------------------------------------------------------


def test_softmax_matches_torch(rng):
    x = rng.standard_normal((5, 9)).astype(np.float32) * 20
    np.testing.assert_allclose(
        softmax(x), torch.softmax(torch.from_numpy(x), dim=-1).numpy(), rtol=1e-6, atol=1e-7
    )


def test_softmax_rows_sum_to_one(rng):
    x = rng.standard_normal((7, 13)).astype(np.float32) * 5
    np.testing.assert_allclose(softmax(x).sum(axis=-1), 1.0, rtol=1e-6)


def test_softmax_survives_large_scores():
    """Without the max subtraction this is inf/inf, i.e. NaN."""
    x = np.array([[800.0, 801.0, 799.0]], dtype=np.float32)
    out = softmax(x)
    assert np.all(np.isfinite(out))
    assert out.sum() == pytest.approx(1.0)


def test_softmax_handles_masked_entries():
    x = np.array([[1.0, -np.inf, 2.0]], dtype=np.float32)
    out = softmax(x)
    assert out[0, 1] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_softmax_of_an_all_masked_row_returns_zeros_not_nan():
    """A deliberate divergence from torch, on an input the model cannot produce.

    Torch returns NaN for a fully masked row. Zeros is the safer answer: a NaN
    propagates silently through every later layer, whereas an all-zero
    attention row is inspectable. A causal mask never produces such a row, so
    the divergence is unreachable in practice.
    """
    x = np.full((1, 4), -np.inf, dtype=np.float32)
    out = softmax(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, np.zeros((1, 4), dtype=np.float32))
    assert torch.isnan(torch.softmax(torch.from_numpy(x), dim=-1)).all(), (
        "torch is expected to NaN here; that is why this guard exists"
    )


def test_softmax_is_shift_invariant(rng):
    x = rng.standard_normal((3, 6)).astype(np.float32)
    np.testing.assert_allclose(softmax(x), softmax(x + 100.0), rtol=1e-5, atol=1e-6)


# -- repeat_kv -------------------------------------------------------------


def test_repeat_kv_expands_consecutively():
    """KV head 0 must serve query heads 0-6, not 0,2,4,...

    This is the grouped-query bug that tile() would introduce: same shape,
    no error, wrong pairing, fluent output.
    """
    kv = np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3)
    out = repeat_kv(kv, 7)

    assert out.shape == (14, 1, 3)
    for head in range(7):
        np.testing.assert_array_equal(out[head], kv[0])
    for head in range(7, 14):
        np.testing.assert_array_equal(out[head], kv[1])


def test_repeat_kv_differs_from_tile():
    """Pin the distinction explicitly so the mistake cannot be reintroduced."""
    kv = np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3)
    assert not np.array_equal(repeat_kv(kv, 7), np.tile(kv, (7, 1, 1)))


def test_repeat_kv_matches_torch_expand_reshape():
    """The reference implements this as expand + reshape; match it exactly."""
    kv = np.random.default_rng(1).standard_normal((2, 5, 64)).astype(np.float32)
    n_rep = 7

    t = torch.from_numpy(kv)[:, None, :, :].expand(2, n_rep, 5, 64)
    expected = t.reshape(2 * n_rep, 5, 64).numpy()

    np.testing.assert_array_equal(repeat_kv(kv, n_rep), expected)


def test_repeat_kv_is_identity_for_one():
    kv = np.zeros((4, 2, 8), dtype=np.float32)
    assert repeat_kv(kv, 1) is kv


def test_repeat_kv_rejects_wrong_rank():
    with pytest.raises(ValueError, match="heads, seq, dim"):
        repeat_kv(np.zeros((2, 3), np.float32), 2)


# -- causal_mask -----------------------------------------------------------


def test_causal_mask_allows_the_diagonal():
    """Position i must be able to attend to itself. j <= i, not j < i."""
    mask = causal_mask(4)
    for i in range(4):
        assert mask[i, i] == 0.0, "a token must attend to itself"


def test_causal_mask_blocks_the_future():
    mask = causal_mask(4)
    for i in range(4):
        for j in range(i + 1, 4):
            assert mask[i, j] == -np.inf


def test_causal_mask_permits_the_past():
    mask = causal_mask(5)
    for i in range(5):
        for j in range(i + 1):
            assert mask[i, j] == 0.0


def test_causal_mask_first_row_attends_only_to_itself():
    mask = causal_mask(3)
    assert mask[0, 0] == 0.0
    assert np.all(np.isneginf(mask[0, 1:]))


def test_causal_mask_matches_torch_triu():
    n = 6
    expected = torch.full((n, n), float("-inf")).triu(diagonal=1).numpy()
    np.testing.assert_array_equal(causal_mask(n), expected)


def test_causal_mask_of_length_one():
    np.testing.assert_array_equal(causal_mask(1), np.zeros((1, 1), dtype=np.float32))
