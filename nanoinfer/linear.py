"""One place where a linear layer is applied, whatever the weight is stored as.

Up to phase 7 every projection in the forward pass was written ``x @ W.T``,
and ``W`` was always a float32 array. Phase 6's int8 weights could only reach
the forward pass by being dequantized back to float32 first, which kept the
quality measurement honest and threw away the point of storing fewer bytes.

This is the seam that lets them stay int8. :func:`linear` takes either kind of
weight and the forward pass no longer needs to know which it has:

* a float32 ``ndarray`` is ``x @ W.T``, exactly as before -- one BLAS call,
  and the fp32 path is untouched byte for byte;
* an INT8 :class:`QuantizedTensor` goes to the Rust kernel, which widens each
  weight in a register and never materialises a float32 copy.

Without the compiled kernels the int8 path falls back to NumPy's widen-and-
multiply. That is correct and 30-80x slower than fp32 (phase 6 measured it),
so the engine says which path it is on rather than quietly being slow.
"""

from __future__ import annotations

import numpy as np

from nanoinfer import kernels
from nanoinfer.quantization import QuantizedTensor

Weight = np.ndarray | QuantizedTensor


def linear(x: np.ndarray, weight: Weight) -> np.ndarray:
    """``x @ W.T`` for ``x`` shaped ``[tokens, in_features]``."""
    if isinstance(weight, np.ndarray):
        return x @ weight.T

    if not isinstance(weight, QuantizedTensor):
        raise TypeError(f"cannot apply a linear layer with a {type(weight).__name__}")
    if weight.bits != 8:
        # The packed-nibble layout has no kernel. Dequantizing per call would
        # "work" at a cost nobody would choose knowingly; refuse instead.
        raise NotImplementedError(
            f"INT{weight.bits} weights have no kernel; quantize_model(..., "
            "dequantize=True) simulates them in float32"
        )

    x = np.asarray(x, dtype=np.float32)
    if kernels.available():
        return kernels.matmul_i8(weight.values, weight.scales, x)
    return (x @ weight.values.T.astype(np.float32)) * weight.scales


def linear_many(x: np.ndarray, weights: list[Weight]) -> list[np.ndarray]:
    """``[linear(x, w) for w in weights]``, in one kernel call when it can be.

    Q, K and V all read the same activations, and so do gate and up. Called
    one at a time that is seven trips into Rust per layer, and K and V are
    each 128 rows -- less work than the ~6 us a call costs before anything
    happens. When :func:`~nanoinfer.quantization.quantize_model` has stored a
    group's int8 rows back to back in one buffer, the group is one matrix
    already, and this makes a single call over all of its rows and splits
    the result.

    In the Rust kernel every output row is its own dot product, so the one
    call returns exactly the bits the separate calls would. (The NumPy
    fallback is one BLAS matmul instead, equal to float32 tolerance.)
    Anything else -- fp32 weights, int8 weights stored apart -- takes the
    separate calls, unchanged.
    """
    fused = _adjacent_int8(weights)
    if fused is None:
        return [linear(x, w) for w in weights]

    combined = linear(x, fused)
    outputs, start = [], 0
    for weight in weights:
        rows = weight.values.shape[0]
        outputs.append(combined[:, start : start + rows])
        start += rows
    return outputs


def _adjacent_int8(weights: list[Weight]) -> QuantizedTensor | None:
    """One tensor spanning ``weights`` if they are consecutive int8 row blocks.

    True only when every weight is per-row INT8 and both its values and its
    scales sit immediately after the previous weight's, inside one buffer --
    which is how ``quantize_model(dequantize=False)`` lays out each group. A
    view over the span is then exactly the stacked matrix, with no copy.
    """
    if len(weights) < 2 or not all(
        isinstance(w, QuantizedTensor) and w.bits == 8 for w in weights
    ):
        return None

    first = weights[0]
    values_base, scales_base = first.values.base, first.scales.base
    if values_base is None or scales_base is None:
        return None

    in_features = first.values.shape[1]
    for before, after in zip(weights, weights[1:]):
        for name, base in (("values", values_base), ("scales", scales_base)):
            a, b = getattr(before, name), getattr(after, name)
            if b.base is not base or not b.flags.c_contiguous:
                return None
            if b.ctypes.data != a.ctypes.data + a.nbytes:
                return None
        if after.values.shape[1] != in_features:
            return None

    rows = sum(w.values.shape[0] for w in weights)
    values = np.ndarray(
        (rows, in_features), dtype=np.int8, buffer=values_base,
        offset=first.values.ctypes.data - values_base.ctypes.data,
    )
    scales = np.ndarray(
        (rows,), dtype=np.float32, buffer=scales_base,
        offset=first.scales.ctypes.data - scales_base.ctypes.data,
    )
    return QuantizedTensor(values=values, scales=scales, bits=8)


def gather_rows(weight: Weight, ids: np.ndarray) -> np.ndarray:
    """Rows of a weight matrix as float32: the embedding lookup.

    For int8 storage only the gathered rows are widened, so the lookup costs
    ``len(ids) x hidden`` conversions rather than the whole table.
    """
    if isinstance(weight, np.ndarray):
        return weight[ids]
    if weight.bits != 8:
        raise NotImplementedError(f"INT{weight.bits} embeddings have no row lookup")
    return weight.values[ids].astype(np.float32) * weight.scales[ids, None]


def backend() -> str:
    """Which path an int8 weight will take, for the CLI and benchmarks."""
    if not kernels.available():
        return "numpy (rust kernels not built; int8 is slow)"
    info = kernels.describe()
    simd = "avx2" if info["avx2"] else "scalar"
    return f"rust {simd}, {info['threads']} threads"
