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
