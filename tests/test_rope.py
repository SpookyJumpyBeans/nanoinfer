"""Tests for rotary position embeddings, including the convention trap.

The dangerous failure here is not an exception, it is a model that works. Both
RoPE conventions produce finite, plausible numbers from the same weights, and a
model running the wrong one emits grammatical text that degrades with sequence
length. So these tests do two things a self-consistent test would not:

* compare against the reference's own rotary module, not just against a
  definition restated in the test file, and
* assert explicitly that the convention we did *not* choose produces different
  output, so the choice is checked rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoinfer.rope import (
    RotaryEmbedding,
    inverse_frequencies,
    rotate_half,
    rotate_interleaved,
)

torch = pytest.importorskip("torch", reason="reference oracle not installed")

HEAD_DIM = 64
THETA = 1_000_000.0


@pytest.fixture(scope="module")
def rope() -> RotaryEmbedding:
    return RotaryEmbedding(HEAD_DIM, THETA)


# -- inverse frequencies ---------------------------------------------------


def test_inverse_frequencies_shape():
    assert inverse_frequencies(HEAD_DIM, THETA).shape == (HEAD_DIM // 2,)


def test_inverse_frequencies_match_torch():
    """The reference builds these as base ** (arange(0, d, 2) / d), in float32."""
    expected = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM)
    )
    np.testing.assert_allclose(
        inverse_frequencies(HEAD_DIM, THETA), expected.numpy(), rtol=1e-6, atol=0
    )


def test_first_frequency_is_one():
    """Plane 0 has exponent 0, so it rotates a full radian per token."""
    assert inverse_frequencies(HEAD_DIM, THETA)[0] == pytest.approx(1.0)


def test_frequencies_decrease_monotonically():
    freqs = inverse_frequencies(HEAD_DIM, THETA)
    assert np.all(np.diff(freqs) < 0)


def test_larger_theta_slows_the_slowest_plane():
    """theta=1e6 is how Qwen2.5 reaches a 32k context."""
    slow_small = inverse_frequencies(HEAD_DIM, 10_000.0)[-1]
    slow_large = inverse_frequencies(HEAD_DIM, 1_000_000.0)[-1]
    assert slow_large < slow_small


def test_odd_head_dim_is_rejected():
    with pytest.raises(ValueError, match="even"):
        inverse_frequencies(63, THETA)


# -- the convention --------------------------------------------------------


def test_rotate_half_pairs_i_with_i_plus_half():
    x = np.arange(8, dtype=np.float32)
    # [0 1 2 3 | 4 5 6 7] -> [-4 -5 -6 -7 | 0 1 2 3]
    np.testing.assert_array_equal(
        rotate_half(x), np.array([-4, -5, -6, -7, 0, 1, 2, 3], dtype=np.float32)
    )


def test_rotate_interleaved_pairs_adjacent_components():
    x = np.arange(8, dtype=np.float32)
    # [(0,1) (2,3) (4,5) (6,7)] -> [(-1,0) (-3,2) (-5,4) (-7,6)]
    np.testing.assert_array_equal(
        rotate_interleaved(x), np.array([-1, 0, -3, 2, -5, 4, -7, 6], dtype=np.float32)
    )


def test_the_two_conventions_genuinely_differ():
    """If these ever agreed, the convention tests would prove nothing."""
    x = np.arange(HEAD_DIM, dtype=np.float32)
    assert not np.array_equal(rotate_half(x), rotate_interleaved(x))


def test_rotate_half_matches_the_reference_implementation():
    from transformers.models.qwen2.modeling_qwen2 import rotate_half as ref_rotate_half

    x = np.random.default_rng(0).standard_normal((2, 5, HEAD_DIM)).astype(np.float32)
    np.testing.assert_array_equal(rotate_half(x), ref_rotate_half(torch.from_numpy(x)).numpy())


def test_rotate_half_applied_four_times_is_the_identity():
    """It is a 90-degree rotation, so four of them come back around."""
    x = np.random.default_rng(1).standard_normal(HEAD_DIM).astype(np.float32)
    np.testing.assert_allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x)


# -- applying the rotation -------------------------------------------------


def test_position_zero_is_the_identity(rope):
    """cos(0)=1, sin(0)=0, so the first token is never rotated."""
    x = np.random.default_rng(2).standard_normal((14, 1, HEAD_DIM)).astype(np.float32)
    np.testing.assert_allclose(rope.apply(x, np.array([0])), x, rtol=1e-6, atol=1e-7)


def test_rotation_preserves_vector_norm(rope):
    """A rotation cannot change length; if it does, the pairing is wrong."""
    x = np.random.default_rng(3).standard_normal((4, 6, HEAD_DIM)).astype(np.float32)
    rotated = rope.apply(x, np.arange(6))
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5
    )


def test_dot_product_depends_only_on_relative_position(rope):
    """The defining property of RoPE, and the reason it works at all.

    Two vectors rotated to absolute positions 3 and 7 must have the same dot
    product as the same two rotated to 103 and 107.
    """
    rng = np.random.default_rng(4)
    q = rng.standard_normal((1, 1, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((1, 1, HEAD_DIM)).astype(np.float32)

    def score(pos_q: int, pos_k: int) -> float:
        qr = rope.apply(q, np.array([pos_q]))
        kr = rope.apply(k, np.array([pos_k]))
        return float(np.sum(qr * kr))

    assert score(3, 7) == pytest.approx(score(103, 107), rel=1e-3)
    assert score(3, 7) != pytest.approx(score(3, 20), rel=1e-3)


def test_apply_matches_the_reference_rotary_module(rope):
    """End-to-end against transformers' own Qwen2 rotary embedding."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    rng = np.random.default_rng(5)
    seq = 12
    q = rng.standard_normal((14, seq, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((2, seq, HEAD_DIM)).astype(np.float32)
    positions = np.arange(seq)

    cos, sin = rope.tables(positions)
    qt = torch.from_numpy(q)[None]      # [batch, heads, seq, dim]
    kt = torch.from_numpy(k)[None]
    ref_q, ref_k = apply_rotary_pos_emb(
        qt, kt, torch.from_numpy(cos)[None], torch.from_numpy(sin)[None]
    )

    np.testing.assert_allclose(rope.apply(q, positions), ref_q[0].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rope.apply(k, positions), ref_k[0].numpy(), rtol=1e-5, atol=1e-6)


def test_interleaved_convention_would_give_different_results(rope):
    """Proof that choosing wrong is silent: finite, plausible, different."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 5, HEAD_DIM)).astype(np.float32)
    positions = np.arange(5)
    cos, sin = rope.tables(positions)

    correct = x * cos[None] + rotate_half(x) * sin[None]
    wrong = x * cos[None] + rotate_interleaved(x) * sin[None]

    assert np.all(np.isfinite(wrong)), "the wrong convention does not blow up"
    assert not np.allclose(correct, wrong), "and it is not the same answer"


# -- table management ------------------------------------------------------


def test_tables_have_the_right_shape(rope):
    cos, sin = rope.tables(np.arange(7))
    assert cos.shape == (7, HEAD_DIM)
    assert sin.shape == (7, HEAD_DIM)


def test_first_and_second_halves_of_a_row_share_angles(rope):
    """The duplication, not interleaving, is what the half-split pairing needs."""
    cos, sin = rope.tables(np.array([5]))
    half = HEAD_DIM // 2
    np.testing.assert_array_equal(cos[0, :half], cos[0, half:])
    np.testing.assert_array_equal(sin[0, :half], sin[0, half:])


def test_table_grows_on_demand():
    r = RotaryEmbedding(HEAD_DIM, THETA)
    assert r._length == 0
    r.tables(np.array([0]))
    grown = r._length
    assert grown >= 256, "should over-allocate rather than grow one row at a time"
    r.tables(np.array([grown + 5]))
    assert r._length > grown


def test_non_contiguous_positions_work(rope):
    """Phase 4 will ask for a single position in the middle of a cache."""
    x = np.ones((1, 1, HEAD_DIM), dtype=np.float32)
    at_ten = rope.apply(x, np.array([10]))
    full = rope.apply(np.ones((1, 11, HEAD_DIM), dtype=np.float32), np.arange(11))
    np.testing.assert_allclose(at_ten[0, 0], full[0, 10], rtol=1e-6)


def test_apply_rejects_wrong_head_dim(rope):
    with pytest.raises(ValueError, match="head_dim"):
        rope.apply(np.zeros((1, 1, 32), np.float32), np.array([0]))


def test_apply_rejects_position_count_mismatch(rope):
    with pytest.raises(ValueError, match="positions"):
        rope.apply(np.zeros((1, 4, HEAD_DIM), np.float32), np.array([0, 1]))
