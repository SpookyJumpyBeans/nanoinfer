"""The KV cache: the reason inference is not quadratic.

Attention at position ``t`` needs the keys and values of every position up to
``t``. Those keys and values depend only on their own token and its position,
so once computed they never change -- token 3's key is the same whether the
sequence is 4 tokens long or 400. Phase 3 recomputed them all at every step
anyway, which is why it ran at 0.25 tokens per second.

Caching them turns each decode step from "project and attend over t tokens"
into "project and attend over *one* token, against t stored keys". The attention
score matrix shrinks from ``[heads, t, t]`` to ``[heads, 1, t]``, and the three
QKV projections shrink from ``[t, hidden]`` to ``[1, hidden]``.

**Storage is pre-allocated, not appended.** Growing a list and re-concatenating
every step would reintroduce the copying the cache exists to avoid. Instead one
buffer per tensor is allocated up front and a ``length`` watermark tracks how
much of it is live; extending means writing into the next free slots and
returning a view of the filled region.

**Keys are stored after RoPE, not before.** Rotation depends only on a token's
own absolute position, which never changes, so rotating once at insert is both
correct and cheaper. Storing pre-rotation keys would mean re-rotating the whole
cache on every step -- and storing post-rotation keys while *also* rotating
them again on read is a bug that yields fluent, position-scrambled text.

The size is unforgiving and worth knowing before you allocate: for
Qwen2.5-0.5B this is 24 KiB per token across all 24 layers at float32. Grouped
-query attention is what makes that bearable -- with 14 KV heads instead of 2 it
would be 168 KiB per token, and a 32k context would want 5.5 GB.
"""

from __future__ import annotations

import numpy as np

from nanoinfer.config import ModelConfig


class CacheFullError(RuntimeError):
    """The sequence outgrew the cache's pre-allocated capacity."""


class CacheUsageError(RuntimeError):
    """The cache was driven in a way that would corrupt it."""


class KVCache:
    """Per-layer key and value storage for one sequence.

    One sequence, not a batch. Batched serving needs paged or block-based
    allocation so that sequences of different lengths can share memory without
    padding to the longest; that is a real subject and firmly out of scope for
    a single-stream engine.
    """

    def __init__(self, config: ModelConfig, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.config = config
        self.capacity = capacity
        self._length = 0

        shape = (
            config.num_hidden_layers,
            config.num_key_value_heads,
            capacity,
            config.head_dim,
        )
        self._keys = np.zeros(shape, dtype=np.float32)
        self._values = np.zeros(shape, dtype=np.float32)

        # Which layers have written during the step currently in progress.
        # A layer that writes twice would put two tokens in one slot; a layer
        # that never writes would leave stale values that later steps attend
        # over as though they were real. Both are silent, so both are checked.
        self._written: set[int] = set()

    # -- state ------------------------------------------------------------

    @property
    def length(self) -> int:
        """How many tokens are committed to the cache."""
        return self._length

    @property
    def remaining(self) -> int:
        return self.capacity - self._length

    @property
    def nbytes(self) -> int:
        return self._keys.nbytes + self._values.nbytes

    @property
    def bytes_per_token(self) -> int:
        return self.nbytes // self.capacity

    def reset(self) -> None:
        """Forget everything, without reallocating.

        The buffers are deliberately not zeroed. Nothing above the watermark is
        ever read, so clearing it would be pure cost -- and a test that only
        passes because the buffer happens to be zeroed is a test that hides a
        read-past-the-watermark bug.
        """
        self._length = 0
        self._written.clear()

    # -- driving it -------------------------------------------------------

    def extend(
        self, layer_index: int, keys: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Store this layer's new keys and values; return everything so far.

        ``keys`` and ``values`` are ``[n_kv_heads, n_new, head_dim]``. The
        returned arrays are ``[n_kv_heads, length + n_new, head_dim]`` views
        into the cache buffer -- no copy, which is the whole point.

        This does not advance the watermark. Every layer writes the same
        tokens at the same offset, so the length can only move once all of
        them have: see :meth:`commit`.
        """
        if not 0 <= layer_index < self.config.num_hidden_layers:
            raise IndexError(
                f"layer {layer_index} out of range for "
                f"{self.config.num_hidden_layers} layers"
            )
        if layer_index in self._written:
            raise CacheUsageError(
                f"layer {layer_index} wrote twice in one step; the second write "
                "would overwrite the first in the same slots"
            )

        expected = (self.config.num_key_value_heads, keys.shape[1], self.config.head_dim)
        if keys.shape != expected or values.shape != expected:
            raise ValueError(
                f"expected keys and values shaped {expected}, got "
                f"{keys.shape} and {values.shape}"
            )

        n_new = keys.shape[1]
        if n_new > self.remaining:
            raise CacheFullError(
                f"cannot add {n_new} tokens: cache holds {self._length} of "
                f"{self.capacity}. Allocate a larger capacity for this prompt."
            )

        start, stop = self._length, self._length + n_new
        self._keys[layer_index, :, start:stop, :] = keys
        self._values[layer_index, :, start:stop, :] = values
        self._written.add(layer_index)

        return (
            self._keys[layer_index, :, :stop, :],
            self._values[layer_index, :, :stop, :],
        )

    def commit(self, n_new: int) -> None:
        """Advance the watermark once every layer has written this step.

        Called by the model after the layer loop rather than by a layer,
        because the watermark is a property of the sequence and not of any one
        layer. Requiring every layer to have written first turns "a layer
        silently skipped the cache" into an exception instead of a wrong answer
        on the next step.
        """
        if n_new < 0:
            raise ValueError(f"n_new must be non-negative, got {n_new}")
        if n_new == 0:
            self._written.clear()
            return

        expected = set(range(self.config.num_hidden_layers))
        if self._written != expected:
            missing = sorted(expected - self._written)
            raise CacheUsageError(
                f"layers {missing} did not write to the cache this step; "
                "advancing now would leave them holding stale keys"
            )

        self._length += n_new
        self._written.clear()

    # -- inspection -------------------------------------------------------

    def keys(self, layer_index: int) -> np.ndarray:
        """A read-only view of the committed keys for one layer."""
        view = self._keys[layer_index, :, : self._length, :]
        return view

    def values(self, layer_index: int) -> np.ndarray:
        view = self._values[layer_index, :, : self._length, :]
        return view

    def __repr__(self) -> str:
        return (
            f"KVCache({self._length}/{self.capacity} tokens, "
            f"{self.nbytes / 1e6:.1f} MB allocated, "
            f"{self.bytes_per_token / 1024:.1f} KiB per token)"
        )
