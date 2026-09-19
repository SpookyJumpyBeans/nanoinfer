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

# Signed 4-bit spans [-8, 7]. Using [-7, 7] keeps the grid symmetric for the
# same reason INT8 uses [-127, 127], at the cost of one of the sixteen levels.
INT4_MAX = 7

# INT4 scales are per group of this many inputs rather than per row. Fifteen
# levels is far too coarse to share one scale across a whole row: a group of
# 128 keeps the scale close to the local magnitude, and costs one float per 128
# weights -- about 3% overhead, against a 4x saving on the values.
INT4_GROUP_SIZE = 128


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """Integer values plus the scales that restore them.

    INT8 stores one scale per output row and keeps values as int8. INT4 stores
    one scale per group of ``group_size`` inputs and packs two values per byte,
    so ``values`` is uint8 with half the width of the original.
    """

    values: np.ndarray          # int8 [out, in], or packed uint8 [out, in/2]
    scales: np.ndarray          # float32 [out] or [out, n_groups]
    bits: int = 8
    group_size: int = 0         # 0 means per-row
    in_features: int = 0        # original width, before padding and packing

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.values.shape[0], self.in_features or self.values.shape[1])

    @property
    def nbytes(self) -> int:
        return self.values.nbytes + self.scales.nbytes

    def dequantize(self) -> np.ndarray:
        """Reconstruct float32 weights. Lossy by construction."""
        if self.bits == 8:
            return (self.values.astype(np.float32) * self.scales[:, None]).astype(
                np.float32
            )

        unpacked = unpack_nibbles(self.values).astype(np.float32)
        out, padded = unpacked.shape
        grouped = unpacked.reshape(out, -1, self.group_size)
        restored = (grouped * self.scales[:, :, None]).reshape(out, padded)
        return restored[:, : self.in_features].astype(np.float32)


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


# -- INT4 ------------------------------------------------------------------


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack signed values in [-7, 7] two-per-byte.

    Offset by 8 so the range becomes [1, 15] and fits in four bits, then the
    even column goes in the low nibble and the odd column in the high one.
    """
    shifted = (values.astype(np.int16) + 8).astype(np.uint8)
    low, high = shifted[:, 0::2], shifted[:, 1::2]
    return (low | (high << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`, restoring the signed values."""
    low = (packed & 0x0F).astype(np.int16) - 8
    high = ((packed >> 4) & 0x0F).astype(np.int16) - 8

    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int16)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def quantize_int4(
    weights: np.ndarray, group_size: int = INT4_GROUP_SIZE
) -> QuantizedTensor:
    """Quantize to symmetric group-wise INT4, packed two values per byte.

    Scales are per group of ``group_size`` consecutive inputs rather than per
    row. With only fifteen levels a single row-wide scale is far too coarse --
    one large weight at the end of a row would flatten everything before it --
    and grouping keeps the scale near the local magnitude.

    The input width is padded up to a multiple of the group size, and of two,
    so the packing is exact. The original width is recorded so the padding can
    be stripped on the way back out.
    """
    if weights.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {weights.shape}")
    if group_size < 2 or group_size % 2:
        raise ValueError(f"group_size must be even and at least 2, got {group_size}")

    weights = weights.astype(np.float32, copy=False)
    out_features, in_features = weights.shape

    padding = (-in_features) % group_size
    padded = np.pad(weights, ((0, 0), (0, padding))) if padding else weights

    grouped = padded.reshape(out_features, -1, group_size)
    magnitudes = np.max(np.abs(grouped), axis=2)

    scales = magnitudes / INT4_MAX
    scales[scales == 0] = 1.0

    quantized = round_half_away_from_zero(grouped / scales[:, :, None])
    quantized = np.clip(quantized, -INT4_MAX, INT4_MAX)
    quantized = quantized.reshape(out_features, -1).astype(np.int8)

    return QuantizedTensor(
        values=pack_nibbles(quantized),
        scales=scales.astype(np.float32),
        bits=4,
        group_size=group_size,
        in_features=in_features,
    )
