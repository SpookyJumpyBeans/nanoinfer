"""Tests for grouped-query self-attention.

The reference here is torch's fused ``scaled_dot_product_attention``, which
shares no code with this implementation -- different language, different
algorithm, different memory layout. Comparing against it catches the class of
bug that a test restating the same steps cannot.

The real model's logits are checked end-to-end in test_model.py; this file
isolates attention so that when that check fails, there is somewhere smaller to
look.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoinfer.attention import merge_heads, self_attention, split_heads
from nanoinfer.config import ModelConfig
from nanoinfer.ops import repeat_kv
from nanoinfer.rope import RotaryEmbedding
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model

torch = pytest.importorskip("torch", reason="reference oracle not installed")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> ModelWeights:
    return ModelWeights.load(build_tiny_model(tmp_path_factory.mktemp("attn")))


@pytest.fixture(scope="module")
def rope(tiny) -> RotaryEmbedding:
    return RotaryEmbedding(tiny.config.head_dim, tiny.config.rope_theta)


def reference_attention(
    hidden: np.ndarray, layer, config: ModelConfig, rope: RotaryEmbedding
) -> np.ndarray:
    """The same block via torch's fused attention kernel.

    Projections and RoPE are shared with the implementation under test -- they
    have their own tests -- but the masking, softmax and value-weighting all go
    through torch's own kernel rather than ours.
    """
    seq = hidden.shape[0]
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim
    positions = np.arange(seq)

    q = split_heads(hidden @ layer.q_proj_weight.T + layer.q_proj_bias, n_heads, head_dim)
    k = split_heads(hidden @ layer.k_proj_weight.T + layer.k_proj_bias, n_kv, head_dim)
    v = split_heads(hidden @ layer.v_proj_weight.T + layer.v_proj_bias, n_kv, head_dim)

    q = rope.apply(q, positions)
    k = rope.apply(k, positions)
    k = repeat_kv(k, config.kv_group_size)
    v = repeat_kv(v, config.kv_group_size)

    context = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(np.ascontiguousarray(q)),
        torch.from_numpy(np.ascontiguousarray(k)),
        torch.from_numpy(np.ascontiguousarray(v)),
        is_causal=True,
    ).numpy()

    return merge_heads(context) @ layer.o_proj_weight.T


# -- head reshaping --------------------------------------------------------


def test_split_heads_slices_the_feature_axis():
    """Head h must take a contiguous slice of the feature vector.

    The wrong reshape -- straight to [heads, seq, dim] -- has the right element
    count and raises nothing, but gives head 0 the whole first token instead of
    the first 64 features of every token.
    """
    seq, heads, dim = 3, 2, 4
    x = np.arange(seq * heads * dim, dtype=np.float32).reshape(seq, heads * dim)
    out = split_heads(x, heads, dim)

    assert out.shape == (heads, seq, dim)
    np.testing.assert_array_equal(out[0, 0], x[0, :dim])
    np.testing.assert_array_equal(out[1, 0], x[0, dim:])
    np.testing.assert_array_equal(out[0, 2], x[2, :dim])


def test_split_heads_is_not_a_bare_reshape():
    seq, heads, dim = 3, 2, 4
    x = np.arange(seq * heads * dim, dtype=np.float32).reshape(seq, heads * dim)
    assert not np.array_equal(split_heads(x, heads, dim), x.reshape(heads, seq, dim))


def test_merge_heads_inverts_split_heads():
    x = np.random.default_rng(0).standard_normal((5, 12)).astype(np.float32)
    np.testing.assert_array_equal(merge_heads(split_heads(x, 3, 4)), x)


def test_merge_heads_shape():
    x = np.zeros((4, 7, 8), dtype=np.float32)
    assert merge_heads(x).shape == (7, 32)


# -- the block against torch -----------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("seq", [1, 2, 5, 17])
def test_matches_torch_fused_attention(tiny, rope, seq):
    hidden = np.random.default_rng(seq).standard_normal(
        (seq, tiny.config.hidden_size)
    ).astype(np.float32)
    layer = tiny.layers[0]

    mine = self_attention(hidden, layer, tiny.config, rope)
    theirs = reference_attention(hidden, layer, tiny.config, rope)

    np.testing.assert_allclose(mine, theirs, rtol=1e-4, atol=1e-5)


def test_output_shape_matches_input(tiny, rope):
    hidden = np.zeros((6, tiny.config.hidden_size), dtype=np.float32)
    assert self_attention(hidden, tiny.layers[0], tiny.config, rope).shape == hidden.shape


# -- causality -------------------------------------------------------------


def test_future_tokens_cannot_influence_earlier_ones(tiny, rope):
    """The property the causal mask exists to guarantee.

    Change the last token's hidden state. Every earlier output must be
    bit-identical. If the mask is off by one, or absent, they will not be.
    """
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((6, tiny.config.hidden_size)).astype(np.float32)

    altered = hidden.copy()
    altered[-1] = rng.standard_normal(tiny.config.hidden_size).astype(np.float32)

    before = self_attention(hidden, tiny.layers[0], tiny.config, rope)
    after = self_attention(altered, tiny.layers[0], tiny.config, rope)

    np.testing.assert_array_equal(before[:-1], after[:-1])
    assert not np.array_equal(before[-1], after[-1])


def test_a_token_attends_to_itself(tiny, rope):
    """With seq=1 the only permitted position is the token itself.

    If the mask forbade the diagonal this row would be fully masked and the
    output would be all zeros.
    """
    hidden = np.random.default_rng(8).standard_normal(
        (1, tiny.config.hidden_size)
    ).astype(np.float32)
    out = self_attention(hidden, tiny.layers[0], tiny.config, rope)
    assert np.any(out != 0.0)
    assert np.all(np.isfinite(out))


def test_prefix_is_stable_as_the_sequence_grows(tiny, rope):
    """Extending a prompt must not change the outputs of its prefix.

    This is the property phase 4's KV cache depends on entirely: if it did not
    hold, caching keys and values from earlier steps would be invalid.
    """
    rng = np.random.default_rng(9)
    hidden = rng.standard_normal((8, tiny.config.hidden_size)).astype(np.float32)

    short = self_attention(hidden[:5], tiny.layers[0], tiny.config, rope)
    full = self_attention(hidden, tiny.layers[0], tiny.config, rope)

    np.testing.assert_allclose(short, full[:5], rtol=1e-5, atol=1e-6)


# -- grouped-query wiring --------------------------------------------------


def test_grouped_query_ratio_is_actually_exercised(tiny):
    """If the tiny model were plain MHA these tests would prove nothing."""
    assert tiny.config.kv_group_size > 1


def test_tiled_kv_heads_would_give_a_different_answer(tiny, rope):
    """Demonstrate the tile/repeat bug is silent, not loud.

    Swapping repeat for tile changes the answer without changing a shape or
    raising anything -- which is why repeat_kv has its own test rather than
    being trusted.
    """
    from nanoinfer.attention import merge_heads as merge
    from nanoinfer.ops import causal_mask, softmax

    cfg = tiny.config
    layer = tiny.layers[0]
    rng = np.random.default_rng(10)
    hidden = rng.standard_normal((5, cfg.hidden_size)).astype(np.float32)
    positions = np.arange(5)

    q = split_heads(hidden @ layer.q_proj_weight.T + layer.q_proj_bias,
                    cfg.num_attention_heads, cfg.head_dim)
    k = split_heads(hidden @ layer.k_proj_weight.T + layer.k_proj_bias,
                    cfg.num_key_value_heads, cfg.head_dim)
    v = split_heads(hidden @ layer.v_proj_weight.T + layer.v_proj_bias,
                    cfg.num_key_value_heads, cfg.head_dim)
    q, k = rope.apply(q, positions), rope.apply(k, positions)

    def finish(k_expanded, v_expanded):
        scores = q @ k_expanded.transpose(0, 2, 1) * np.float32(cfg.head_dim**-0.5)
        scores = scores + causal_mask(5, dtype=scores.dtype)
        return merge(softmax(scores, -1) @ v_expanded) @ layer.o_proj_weight.T

    correct = finish(repeat_kv(k, cfg.kv_group_size), repeat_kv(v, cfg.kv_group_size))
    wrong = finish(
        np.tile(k, (cfg.kv_group_size, 1, 1)), np.tile(v, (cfg.kv_group_size, 1, 1))
    )

    assert np.all(np.isfinite(wrong)), "the bug produces perfectly valid numbers"
    assert not np.allclose(correct, wrong), "and a different answer"


# -- positions -------------------------------------------------------------


def test_explicit_positions_default_to_the_sequence(tiny, rope):
    hidden = np.random.default_rng(11).standard_normal(
        (4, tiny.config.hidden_size)
    ).astype(np.float32)
    implicit = self_attention(hidden, tiny.layers[0], tiny.config, rope)
    explicit = self_attention(hidden, tiny.layers[0], tiny.config, rope, np.arange(4))
    np.testing.assert_array_equal(implicit, explicit)


def test_uniform_position_shift_is_a_no_op(tiny, rope):
    """Translation invariance: RoPE encodes relative position, not absolute.

    Sliding every position along by the same amount leaves all pairwise
    differences intact, so every attention score is unchanged. This is not an
    incidental property, it is the reason RoPE works, and it is the strongest
    single check that the rotation is being applied to q and k the same way.
    """
    hidden = np.random.default_rng(12).standard_normal(
        (4, tiny.config.hidden_size)
    ).astype(np.float32)

    at_zero = self_attention(hidden, tiny.layers[0], tiny.config, rope, np.arange(4))
    at_fifty = self_attention(hidden, tiny.layers[0], tiny.config, rope, np.arange(50, 54))

    np.testing.assert_allclose(at_zero, at_fifty, rtol=1e-4, atol=1e-6)


def test_changing_the_spacing_between_positions_does_change_the_result(tiny, rope):
    """The other half: stretching the gaps is not a no-op.

    Together with the test above this pins RoPE as genuinely wired in. If the
    rotation were being skipped entirely, both tests would pass the first
    assertion and fail this one.
    """
    hidden = np.random.default_rng(12).standard_normal(
        (4, tiny.config.hidden_size)
    ).astype(np.float32)

    adjacent = self_attention(
        hidden, tiny.layers[0], tiny.config, rope, np.array([0, 1, 2, 3])
    )
    spread = self_attention(
        hidden, tiny.layers[0], tiny.config, rope, np.array([0, 10, 20, 30])
    )

    assert not np.allclose(adjacent, spread)
