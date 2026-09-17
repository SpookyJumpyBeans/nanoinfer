"""Grouped-query self-attention, written out one step at a time.

This is the part of the model where a bug is least likely to announce itself.
Every mistake available here -- the wrong RoPE convention, a mask off by one,
KV heads tiled instead of repeated, a transpose that reshapes cleanly but
scrambles head boundaries -- produces correctly-shaped finite numbers and text
that still reads like English. So the code is written to be obviously right
rather than concisely right, and the tests compare against a completely
different implementation (torch's fused attention kernel) rather than against
a restatement of these same steps.

The shape journey, for one layer of Qwen2.5-0.5B with a 10-token prompt:

    hidden          [10, 896]
    q = x @ Wq.T    [10, 896]   -> [14, 10, 64]   14 query heads
    k = x @ Wk.T    [10, 128]   -> [ 2, 10, 64]    2 key/value heads
    v = x @ Wv.T    [10, 128]   -> [ 2, 10, 64]
    rope(q), rope(k)
    k, v repeated   [14, 10, 64]                   each KV head serves 7
    scores          [14, 10, 10]
    + causal mask, softmax
    out = p @ v     [14, 10, 64] -> [10, 896]
    out @ Wo.T      [10, 896]
"""

from __future__ import annotations

import numpy as np

from nanoinfer.config import ModelConfig
from nanoinfer.kvcache import KVCache
from nanoinfer.ops import causal_mask, repeat_kv, softmax
from nanoinfer.rope import RotaryEmbedding
from nanoinfer.weights import LayerWeights


def split_heads(x: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Reshape ``[seq, num_heads * head_dim]`` into ``[num_heads, seq, head_dim]``.

    The reshape must split the *last* axis into (head, dim) before the
    transpose. Reshaping straight to ``[num_heads, seq, head_dim]`` also
    "works" -- the element count matches and nothing raises -- but it slices
    the sequence across heads instead of slicing the feature vector, which
    scrambles every head. The output stays fluent, naturally.
    """
    seq = x.shape[0]
    return x.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """The inverse: ``[num_heads, seq, head_dim]`` back to ``[seq, hidden]``."""
    num_heads, seq, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq, num_heads * head_dim)


def self_attention(
    hidden: np.ndarray,
    layer: LayerWeights,
    config: ModelConfig,
    rope: RotaryEmbedding,
    positions: np.ndarray | None = None,
    cache: KVCache | None = None,
    layer_index: int | None = None,
) -> np.ndarray:
    """One grouped-query self-attention block, with or without a KV cache.

    ``hidden`` is ``[n_new, hidden_size]`` and the result has the same shape.
    Without a cache, ``n_new`` is the whole sequence. With one, it is only the
    tokens not yet seen -- the whole prompt on the first call, then one token
    per decode step.

    Deliberately one function rather than a cached and an uncached variant.
    Two implementations of attention would drift, and the entire claim of
    phase 4 is that the cached path computes the identical function; that is
    far easier to believe when there is only one path to read.

    ``positions`` defaults to the ``n_new`` absolute positions following
    whatever is already cached. Defaulting it to ``0..n_new-1`` instead would
    rotate every decode token as though it were at position zero, which is the
    single most likely way to get a cache subtly wrong: output stays fluent and
    loses all sense of order beyond the prompt.
    """
    seq = hidden.shape[0]
    cached_len = cache.length if cache is not None else 0

    if positions is None:
        positions = np.arange(cached_len, cached_len + seq)

    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim

    # Projections. Qwen2 puts biases on Q, K and V but not on the output
    # projection -- unlike most Llama-style models, which have none at all.
    q = hidden @ layer.q_proj_weight.T + layer.q_proj_bias
    k = hidden @ layer.k_proj_weight.T + layer.k_proj_bias
    v = hidden @ layer.v_proj_weight.T + layer.v_proj_bias

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_kv, head_dim)
    v = split_heads(v, n_kv, head_dim)

    # Rotate queries and keys by position. Values are never rotated: position
    # belongs in the *matching*, not in the content being retrieved.
    q = rope.apply(q, positions)
    k = rope.apply(k, positions)

    # Store the rotated keys and values, and get back everything so far.
    #
    # Rotation happens before the cache, never after. A token's position never
    # changes, so rotating once on insert is both correct and cheaper than
    # re-rotating the whole history each step. Rotating again on read would
    # apply the rotation twice to every cached key -- fluent output, scrambled
    # sense of order.
    if cache is not None:
        if layer_index is None:
            raise ValueError("layer_index is required when a cache is given")
        k, v = cache.extend(layer_index, k, v)

    # Broadcast the narrow KV heads out to match the query heads. Consecutive,
    # not interleaved -- see repeat_kv.
    k = repeat_kv(k, config.kv_group_size)
    v = repeat_kv(v, config.kv_group_size)

    # Scaled dot product. The 1/sqrt(head_dim) keeps the variance of the scores
    # near 1 regardless of head width; without it softmax saturates and the
    # gradient (and, at inference, the attention distribution) collapses onto
    # a single position.
    scores = q @ k.transpose(0, 2, 1) * np.float32(head_dim**-0.5)

    # The mask is [n_new, cached + n_new] and broadcasts across heads. With a
    # full cache and one new token it is all zeros: the new token may attend
    # everywhere, because everything stored is already in its past.
    scores = scores + causal_mask(seq, cached_len, dtype=scores.dtype)

    weights = softmax(scores, axis=-1)
    context = weights @ v

    return merge_heads(context) @ layer.o_proj_weight.T
