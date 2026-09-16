"""The elementwise and normalization primitives the forward pass is built from.

Everything here is float32 NumPy and deliberately literal. The goal of phase 3
is a forward pass whose logits match a reference implementation to 1e-3; that
is only achievable if each primitive matches the reference's *exact* arithmetic,
including where it chooses to compute in higher precision and in what order it
multiplies. Clever rewrites come later, and only once there is a passing test to
protect them.
"""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Root-mean-square layer normalization.

    RMSNorm is LayerNorm with the mean subtraction removed: it rescales by the
    root mean square instead of standardizing. There is no bias and no
    re-centering, which is why the weight file has a single gain vector per
    norm and nothing else.

        y = x / sqrt(mean(x^2) + eps) * weight

    The epsilon goes *inside* the square root, added to the mean square rather
    than to the root. With eps=1e-6 that distinction is small but it is not
    zero, and it compounds across 49 norms.
    """
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"rms_norm: last dim of x is {x.shape[-1]} but weight is {weight.shape[-1]}"
        )

    # Accumulate the mean square in float32 regardless of the input dtype.
    # The reference does the same, and a bf16 accumulation here would drift
    # far more than 1e-3 by the final layer.
    x32 = x.astype(np.float32, copy=False)
    mean_square = np.mean(x32 * x32, axis=-1, keepdims=True)
    normalized = x32 * np.reciprocal(np.sqrt(mean_square + eps))
    return (normalized * weight).astype(np.float32, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, evaluated on whichever branch cannot overflow.

    The textbook ``1 / (1 + exp(-x))`` overflows ``exp`` for large negative x.
    NumPy still returns the right answer there -- ``inf`` in the denominator
    gives 0 -- but it raises a runtime warning on every call, and a numeric
    kernel that cries wolf is one whose warnings get ignored when they matter.

    So each half is computed with the form whose exponent argument is negative:
    ``exp(-x)`` for x >= 0, ``exp(x)`` for x < 0. Both are in (0, 1].
    """
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x)

    positive = x >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU, also called swish: ``x * sigmoid(x)``.

    This is the activation inside Qwen2's SwiGLU feed-forward block, where the
    full expression is ``down(silu(gate(x)) * up(x))``. Only the gate branch
    passes through the activation; the up branch is multiplied in linearly.
    """
    x = x.astype(np.float32, copy=False)
    return x * sigmoid(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row maximum before exponentiating is not optional. Attention
    scores routinely reach several hundred once scaled, and ``exp(800)`` is
    ``inf``, which turns the whole row into NaN.

    A row that is entirely ``-inf`` -- a fully masked attention row -- sums to
    zero after exponentiating, so the normalization is 0/0. Torch returns NaN
    there. This returns zeros instead: "attends to nothing" is a defensible
    answer and a NaN poisons every downstream value silently. The case cannot
    arise under a causal mask, because position i always attends to itself, so
    the two implementations never disagree on any input the model produces.
    """
    x = x.astype(np.float32, copy=False)

    peak = np.max(x, axis=axis, keepdims=True)
    # An all -inf row has an -inf max; shifting by it would give -inf - -inf = NaN.
    peak = np.where(np.isfinite(peak), peak, 0.0)

    shifted = np.exp(x - peak)
    total = np.sum(shifted, axis=axis, keepdims=True)
    return shifted / np.where(total == 0.0, 1.0, total)


def repeat_kv(x: np.ndarray, repeats: int) -> np.ndarray:
    """Expand grouped-query key/value heads to match the query head count.

    Qwen2.5-0.5B has 14 query heads and 2 key-value heads, so each KV head is
    shared by 7 query heads. The expansion must be **consecutive**: KV head 0
    serves query heads 0-6, KV head 1 serves query heads 7-13.

        repeat  -> [0,0,0,0,0,0,0, 1,1,1,1,1,1,1]   correct
        tile    -> [0,1, 0,1, 0,1, ...]              wrong

    Using ``tile`` here is the classic grouped-query bug. The shapes are
    identical, nothing raises, and the model still produces fluent text -- it
    is just pairing every query head with the wrong key. This is precisely the
    failure the phase-3 logit check exists to catch.

    ``x`` is ``[n_kv_heads, seq, head_dim]``; the result is
    ``[n_kv_heads * repeats, seq, head_dim]``.
    """
    if repeats == 1:
        return x
    if x.ndim != 3:
        raise ValueError(f"repeat_kv expects [heads, seq, dim], got shape {x.shape}")
    return np.repeat(x, repeats, axis=0)


def causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """An additive mask that forbids attending to future positions.

    Additive rather than multiplicative: it is added to the scores before the
    softmax, so blocked positions hold ``-inf`` and permitted ones hold 0.

    The boundary is ``j <= i``, inclusive. Writing ``j < i`` -- forbidding a
    token from attending to itself -- is the off-by-one the brief warns about,
    and the model still produces fluent, subtly wrong text if you do.
    """
    mask = np.zeros((seq_len, seq_len), dtype=dtype)
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask
