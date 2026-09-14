"""A from-scratch reader for the safetensors file format.

The format is deliberately simple enough to parse in one sitting, which is why
this file exists instead of a dependency:

    +--------+-------------------------+-----------------------------------+
    | 8 byte | N bytes                 | rest of file                      |
    | u64 LE | UTF-8 JSON header       | raw tensor bytes, back to back    |
    | = N    |                         |                                   |
    +--------+-------------------------+-----------------------------------+

The JSON header maps each tensor name to its dtype, its shape, and a
``data_offsets`` pair ``[begin, end)``. Those offsets are relative to the start
of the data buffer (byte ``8 + N``), not to the start of the file -- getting
that wrong is the classic first bug.

A reserved ``__metadata__`` key may also appear in the header. Its value is a
flat string->string map, and it is *not* a tensor.

Everything is little-endian. There is no compression and no per-tensor padding,
so a tensor's byte length is always ``prod(shape) * itemsize``; we check that,
because a header that disagrees with the buffer means a truncated download.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# The dtype names the format defines, mapped to how we hold them in NumPy.
#
# BF16 and the FP8 types have no NumPy equivalent, so we keep their raw bits in
# an unsigned integer of the same width and convert on demand. Holding the raw
# bits is not a workaround -- it is what we want, because it lets us memory-map
# the weights without converting a gigabyte up front.
_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "F8_E5M2": np.dtype(np.uint8),   # raw bits
    "F8_E4M3": np.dtype(np.uint8),   # raw bits
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),         # raw bits
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
}

# dtypes whose NumPy container holds raw bits rather than the logical value.
_RAW_BITS = frozenset({"BF16", "F8_E5M2", "F8_E4M3"})

# Guard against a corrupt or hostile header claiming an absurd length before we
# allocate anything. Real headers for multi-billion-parameter models are well
# under a megabyte.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen bfloat16 stored as raw uint16 bits into float32.

    bfloat16 is just float32 with the low 16 mantissa bits chopped off: same
    8-bit exponent, same bias, 7 explicit mantissa bits instead of 23. So the
    conversion is not arithmetic at all -- shift the bits back into the high
    half of a uint32 and reinterpret. This is exact for every value including
    the infinities and NaNs, and it never rounds, because we are only ever
    appending zero bits.
    """
    if raw.dtype != np.dtype("<u2"):
        raise TypeError(f"expected raw uint16 bf16 bits, got {raw.dtype}")
    widened = raw.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32).reshape(raw.shape)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Narrow float32 to bfloat16 bits, rounding half to even.

    The inverse of :func:`bf16_to_f32`, used by the tests to prove the widening
    round-trips. Truncating would be one line; we round properly so that the
    round-trip error matches what a real bf16 cast produces.
    """
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    # Round-half-to-even on the bit pattern: add 0x7FFF plus the low bit of the
    # surviving mantissa, then drop the low 16 bits.
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype("<u2").reshape(values.shape)


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """One entry from the JSON header."""

    name: str
    dtype: str          # the format's own name, e.g. "BF16"
    shape: tuple[int, ...]
    begin: int          # byte offset into the data buffer, inclusive
    end: int            # byte offset into the data buffer, exclusive

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def is_raw_bits(self) -> bool:
        """True when NumPy has no native dtype and we hold the bit pattern."""
        return self.dtype in _RAW_BITS


class SafeTensors:
    """Memory-mapped, read-only access to a .safetensors file.

    Opening the file reads only the header. Tensor data is mapped, not copied,
    so constructing this is fast and cheap regardless of file size; the OS pages
    weights in as they are actually touched.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = open(self.path, "rb")
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

        self._buf = np.frombuffer(self._mmap, dtype=np.uint8)
        self.header, self.metadata, self._data_start = self._parse_header(self._buf)
        self._tensors = self._parse_tensors(
            self.header, len(self._buf) - self._data_start
        )

    # -- parsing ---------------------------------------------------------

    @staticmethod
    def _parse_header(buf: np.ndarray) -> tuple[dict, dict[str, str], int]:
        if buf.size < 8:
            raise ValueError("file is shorter than the 8-byte header length prefix")

        # The one and only fixed-position field: how long the JSON header is.
        header_len = int(buf[:8].view("<u8")[0])
        if header_len == 0:
            raise ValueError("header length is zero")
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"header claims {header_len} bytes, refusing to read")
        if 8 + header_len > buf.size:
            raise ValueError(
                f"header claims {header_len} bytes but only {buf.size - 8} follow "
                "(file is truncated)"
            )

        raw = buf[8 : 8 + header_len].tobytes()
        try:
            header = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"header is not valid JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError("header JSON must be an object")

        metadata = header.pop("__metadata__", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError("__metadata__ must be an object")

        return header, {str(k): str(v) for k, v in metadata.items()}, 8 + header_len

    @staticmethod
    def _parse_tensors(header: dict, data_len: int) -> dict[str, TensorInfo]:
        tensors: dict[str, TensorInfo] = {}
        for name, entry in header.items():
            try:
                dtype = entry["dtype"]
                shape = tuple(int(d) for d in entry["shape"])
                begin, end = (int(x) for x in entry["data_offsets"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"tensor {name!r} has a malformed header entry: {exc}"
                ) from exc

            if dtype not in _DTYPES:
                raise ValueError(f"tensor {name!r} has unsupported dtype {dtype!r}")
            if not 0 <= begin <= end <= data_len:
                raise ValueError(
                    f"tensor {name!r} offsets [{begin}, {end}) fall outside the "
                    f"{data_len}-byte data buffer"
                )

            info = TensorInfo(name, dtype, shape, begin, end)
            expected = info.numel * _DTYPES[dtype].itemsize
            if expected != info.nbytes:
                raise ValueError(
                    f"tensor {name!r} shape {shape} needs {expected} bytes but the "
                    f"header reserves {info.nbytes}"
                )
            tensors[name] = info
        return tensors

    # -- access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    @property
    def names(self) -> list[str]:
        return list(self._tensors)

    def info(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in {self.path.name}") from None

    def raw(self, name: str) -> np.ndarray:
        """A zero-copy, read-only view of the tensor exactly as stored.

        For BF16 and the FP8 types this is the bit pattern in an unsigned
        integer array, not a float array. Use :meth:`f32` if you want numbers.
        """
        info = self.info(name)
        start = self._data_start + info.begin
        flat = self._buf[start : start + info.nbytes]
        arr = flat.view(_DTYPES[info.dtype]).reshape(info.shape)
        arr.flags.writeable = False
        return arr

    def f32(self, name: str) -> np.ndarray:
        """The tensor as a freshly allocated float32 array.

        This copies, because it has to: widening bf16 doubles the size. Call it
        deliberately, not in a loop.
        """
        info = self.info(name)
        raw = self.raw(name)
        if info.dtype == "BF16":
            return bf16_to_f32(raw)
        if info.dtype in _RAW_BITS:
            raise NotImplementedError(
                f"no float conversion implemented for {info.dtype}"
            )
        return raw.astype(np.float32)

    @property
    def total_params(self) -> int:
        return sum(t.numel for t in self._tensors.values())

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self._tensors.values())

    @property
    def data_start(self) -> int:
        """Byte offset where the tensor data buffer begins."""
        return self._data_start

    @property
    def header_bytes(self) -> int:
        """Length of the JSON header, excluding the 8-byte length prefix."""
        return self._data_start - 8

    def close(self) -> None:
        # Drop our own view first; NumPy holds a buffer export on the mapping
        # and mmap.close() refuses while one is outstanding.
        self._buf = np.empty(0, dtype=np.uint8)
        try:
            self._mmap.close()
        except BufferError:
            # Views returned by raw() are still alive and point into the
            # mapping. Unmapping now would leave them dangling, so we let
            # CPython release it when the last view is collected. The file
            # handle below is safe to drop either way.
            pass
        finally:
            self._file.close()

    def __enter__(self) -> "SafeTensors":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
