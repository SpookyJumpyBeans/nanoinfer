"""Rotary position embeddings, in the convention Qwen2.5 was trained with.

RoPE encodes position by *rotating* pairs of components of each query and key
vector by an angle proportional to the token's position. Because a dot product
between two rotated vectors depends only on the difference of their angles,
attention scores end up depending on relative position without any position
vector ever being added to the residual stream.

**The convention is the whole problem.** There are two incompatible ways to
choose which components get paired into a rotating 2-D plane, and both are in
wide use:

* **Half-split (GPT-NeoX, HuggingFace, and Qwen2).** Component ``i`` pairs with
  component ``i + d/2``. For d=64 that is (0,32), (1,33), ... (31,63).
* **Interleaved (GPT-J, llama.cpp's internal layout).** Component ``2i`` pairs
  with ``2i+1``: (0,1), (2,3), ... (62,63).

They produce different numbers from the same weights. Neither errors. A model
run with the wrong one still emits grammatical text -- it has simply been told
a scrambled version of where each token is, so it degrades with distance rather
than failing outright. That is the single most expensive bug available in this
project, which is why :func:`rotate_half` has a test comparing it against the
interleaved alternative rather than only against itself.

Qwen2.5 uses half-split, and ``rope_theta`` is 1,000,000 rather than the more
common 10,000 -- a larger base stretches the wavelength of every frequency,
which is how the model supports a 32k context.
"""

from __future__ import annotations

import numpy as np


def inverse_frequencies(head_dim: int, theta: float) -> np.ndarray:
    """The per-pair rotation rates, one for each of the ``head_dim / 2`` planes.

    ``inv_freq[i] = theta ** (-2i / d)``, so plane 0 rotates once per token and
    the last plane rotates once per ``theta`` tokens. Fast planes encode local
    order; slow planes encode position over long distances.

    Computed in float32 throughout, because the reference does. Doing this in
    float64 and casting at the end produces slightly different angles, and at
    1e-3 tolerance across 24 layers that difference is measurable.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    exponent = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(
        head_dim
    )
    return (np.float32(1.0) / (np.float32(theta) ** exponent)).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate the half-split pairs by 90 degrees: ``(a, b) -> (-b, a)``.

    Given ``x = [x0 ... x31 | x32 ... x63]`` this returns
    ``[-x32 ... -x63 | x0 ... x31]``.

    Combined as ``x * cos + rotate_half(x) * sin`` this is exactly the 2-D
    rotation ``(a cos - b sin, a sin + b cos)`` applied to every pair
    ``(x_i, x_{i + d/2})`` at once.
    """
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def rotate_interleaved(x: np.ndarray) -> np.ndarray:
    """The *other* convention's rotation, pairing ``(x_2i, x_2i+1)``.

    Not used by this model. It exists so the tests can assert that the two
    conventions genuinely differ, making the choice in :func:`rotate_half` a
    checked decision rather than an unexamined one.
    """
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = np.stack((-pairs[..., 1], pairs[..., 0]), axis=-1)
    return rotated.reshape(x.shape)


class RotaryEmbedding:
    """Precomputed cos/sin tables, grown on demand.

    The tables depend only on position, not on the data, so they are computed
    once and reused for every layer and every token. Phase 4 leans on this
    harder: with a KV cache the model only ever rotates the single new token,
    at a position looked up in these tables.
    """

    def __init__(self, head_dim: int, theta: float) -> None:
        self.head_dim = head_dim
        self.theta = theta
        self._inv_freq = inverse_frequencies(head_dim, theta)
        self._length = 0
        self._cos = np.zeros((0, head_dim), dtype=np.float32)
        self._sin = np.zeros((0, head_dim), dtype=np.float32)

    def _ensure(self, length: int) -> None:
        if length <= self._length:
            return

        # Grow generously so a token-at-a-time decode loop does not rebuild
        # the table on every step.
        target = max(length, self._length * 2, 256)

        positions = np.arange(target, dtype=np.float32)
        angles = np.outer(positions, self._inv_freq)          # [target, d/2]

        # Duplicated, not interleaved: entry i and entry i + d/2 share an
        # angle, which is what makes the half-split pairing work.
        doubled = np.concatenate((angles, angles), axis=-1)   # [target, d]

        self._cos = np.cos(doubled).astype(np.float32)
        self._sin = np.sin(doubled).astype(np.float32)
        self._length = target

    def tables(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ``(cos, sin)`` rows for the given positions, shaped ``[n, d]``."""
        positions = np.asarray(positions)
        if positions.size and int(positions.max()) >= self._length:
            self._ensure(int(positions.max()) + 1)
        return self._cos[positions], self._sin[positions]

    def apply(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Rotate ``x`` of shape ``[heads, seq, head_dim]`` by position.

        ``positions`` has one entry per sequence slot. It is passed explicitly
        rather than assumed to be ``0..seq-1`` because with a KV cache the new
        token's position is the length of the cache, not zero.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected head_dim {self.head_dim}, got {x.shape[-1]}"
            )
        if x.shape[-2] != len(positions):
            raise ValueError(
                f"{x.shape[-2]} sequence slots but {len(positions)} positions"
            )

        cos, sin = self.tables(positions)
        # Broadcast [seq, d] against [heads, seq, d].
        return x * cos[None, :, :] + rotate_half(x) * sin[None, :, :]
