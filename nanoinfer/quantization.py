"""Storing weights in fewer bits.

Phase 4 established what this is actually for. Decoding one token reads every
weight in the model -- 1.98 GB at float32 -- to produce a single 896-value
vector, at roughly 24 GB/s. The decode step is bound by memory bandwidth, not
arithmetic, so the lever that matters is *bytes moved*, and quantization is the
only way to move fewer of them.

INT8 is a 4x reduction against float32, INT4 an 8x.

Three choices, all of them the standard ones, and each worth a sentence:

**Symmetric, no zero point.** A quantized value is just ``q * scale``; there is
no offset to add. Weights are close to zero-centered so the asymmetry an offset
would buy is small, and every downstream matmul stays a plain multiply.

**Per output channel, not per tensor.** Each row of a ``[out_features,
in_features]`` weight gets its own scale. One scale for a whole tensor is
dominated by its largest outlier, and a single large row then crushes the
resolution of every other. Per-channel costs one float per row -- 896 floats
against 802,816 weights for a q_proj -- and is worth far more than it costs.

**Round half away from zero, then clip.** ``np.round`` is banker's rounding
(half to even), which biases a symmetric grid; ``floor(x + 0.5)`` on the
magnitude is the convention quantization toolkits use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Symmetric INT8 uses [-127, 127], not the full [-128, 127]. Keeping the range
# symmetric means a weight and its negation quantize to exact opposites; using
# -128 would give the negative side one extra step and skew the reconstruction.
INT8_MAX = 127


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """Integer values plus the per-channel scales that restore them."""

    values: np.ndarray        # int8, shape [out, in]
    scales: np.ndarray        # float32, shape [out]
    bits: int = 8

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def nbytes(self) -> int:
        return self.values.nbytes + self.scales.nbytes

    def dequantize(self) -> np.ndarray:
        """Reconstruct float32 weights. Lossy by construction."""
        return (self.values.astype(np.float32) * self.scales[:, None]).astype(np.float32)


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Round .5 away from zero rather than to even.

    np.round implements banker's rounding, which sends 0.5 to 0 and 1.5 to 2.
    On a symmetric integer grid that biases small magnitudes toward zero, and
    small magnitudes are most of a weight matrix.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_int8(weights: np.ndarray) -> QuantizedTensor:
    """Quantize a 2-D ``[out, in]`` weight matrix to symmetric per-row INT8.

    The scale for a row is ``max(|row|) / 127``, so the largest magnitude in
    the row maps exactly to +-127 and everything else lands proportionally
    inside. A row that is entirely zero would divide by zero, so its scale is
    forced to 1 -- the reconstruction is zeros either way.
    """
    if weights.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {weights.shape}")

    weights = weights.astype(np.float32, copy=False)
    magnitudes = np.max(np.abs(weights), axis=1)

    scales = magnitudes / INT8_MAX
    scales[scales == 0] = 1.0          # all-zero row: any scale reconstructs zeros

    quantized = round_half_away_from_zero(weights / scales[:, None])
    quantized = np.clip(quantized, -INT8_MAX, INT8_MAX).astype(np.int8)

    return QuantizedTensor(values=quantized, scales=scales.astype(np.float32), bits=8)


def quantization_error(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    """How far a round trip moved the weights.

    Relative Frobenius error is the headline: the norm of the difference over
    the norm of the original, which is scale-free and comparable across tensors
    of different magnitudes.
    """
    difference = original.astype(np.float64) - restored.astype(np.float64)
    original_norm = np.linalg.norm(original.astype(np.float64))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rel_frobenius": float(np.linalg.norm(difference) / original_norm)
        if original_norm
        else 0.0,
    }
