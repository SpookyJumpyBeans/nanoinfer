"""Calling the Rust kernels from the NumPy engine.

The engine is NumPy and stays NumPy. This module is the seam: it loads the
compiled ``nanoinfer_kernels`` cdylib if one has been built and exposes the
int8 matvec through it, falling back to NumPy when it has not.

**ctypes, not PyO3.** The whole surface is four pointers and two lengths.
PyO3 would mean a build step, a compiled extension per Python version, and a
wheel to distribute; ctypes needs the library and nothing else, and the crate
already emits a cdylib for it. The cost is that nothing is checked at the
boundary, so everything is checked on this side -- see :func:`_as_kernel_input`.

Build the library with::

    cd rust && cargo build --release

and it is picked up automatically. Nothing in the engine requires it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_LIBRARY_STEM = "nanoinfer_kernels"


def _candidate_paths() -> list[Path]:
    """Where a built library might be, most specific first."""
    override = os.environ.get("NANOINFER_KERNELS")
    if override:
        return [Path(override)]

    if sys.platform == "win32":
        names = [f"{_LIBRARY_STEM}.dll"]
    elif sys.platform == "darwin":
        names = [f"lib{_LIBRARY_STEM}.dylib"]
    else:
        names = [f"lib{_LIBRARY_STEM}.so"]

    root = Path(__file__).resolve().parent.parent / "rust" / "target"
    return [root / profile / name for profile in ("release", "debug") for name in names]


def _load() -> ctypes.CDLL | None:
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            library = ctypes.CDLL(str(path))
        except OSError:
            continue

        # Declaring these is not optional. Without argtypes ctypes passes
        # Python ints as C int, which truncates a 64-bit pointer and reads
        # whatever happens to be at the low half -- the same failure that cost
        # phase 3 an afternoon on GetCurrentProcess.
        matvec_args = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        for symbol in ("nanoinfer_matvec_i8", "nanoinfer_matvec_i8_single"):
            function = getattr(library, symbol)
            function.argtypes = matvec_args
            function.restype = None

        library.nanoinfer_has_avx2.argtypes = []
        library.nanoinfer_has_avx2.restype = ctypes.c_int
        library.nanoinfer_num_threads.argtypes = []
        library.nanoinfer_num_threads.restype = ctypes.c_int
        return library
    return None


_LIBRARY = _load()


def available() -> bool:
    """Whether the compiled kernels were found and loaded."""
    return _LIBRARY is not None


def describe() -> dict[str, object]:
    """What was loaded, for benchmarks and for the engine to report."""
    if _LIBRARY is None:
        return {"available": False, "path": None, "avx2": False, "threads": 0}
    return {
        "available": True,
        "path": str(next(p for p in _candidate_paths() if p.exists())),
        "avx2": bool(_LIBRARY.nanoinfer_has_avx2()),
        "threads": int(_LIBRARY.nanoinfer_num_threads()),
    }


def _as_kernel_input(array: np.ndarray, dtype, name: str) -> np.ndarray:
    """Coerce to a C-contiguous array of ``dtype``, loudly.

    Rust reads ``out_features * in_features`` elements straight off the
    pointer. A non-contiguous view or a wrong dtype would be read as if it
    were contiguous and correctly typed, which is a segfault or, worse,
    plausible garbage. So this is strict, and it copies rather than failing
    only when it can do so without changing the values.
    """
    coerced = np.ascontiguousarray(array, dtype=dtype)
    if coerced.dtype != dtype:
        raise TypeError(f"{name}: expected {np.dtype(dtype)}, got {array.dtype}")
    return coerced


def matvec_i8(
    quantized: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    *,
    threaded: bool = True,
) -> np.ndarray:
    """``(quantized * scales[:, None]) @ x``, computed in Rust.

    ``quantized`` is ``[out_features, in_features]`` int8, ``scales`` is one
    float32 per output row, ``x`` is ``[in_features]`` float32. Returns
    ``[out_features]`` float32.

    ``threaded=False`` runs the single-threaded SIMD kernel, which exists so
    the benchmark can separate the SIMD win from the threading win.
    """
    if _LIBRARY is None:
        raise RuntimeError(
            "the Rust kernels are not built; run `cargo build --release` in rust/"
        )
    if quantized.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {quantized.shape}")

    out_features, in_features = quantized.shape
    if scales.shape != (out_features,):
        raise ValueError(
            f"expected one scale per output row: {scales.shape} against {out_features}"
        )
    if x.shape != (in_features,):
        raise ValueError(f"x has width {x.shape}, weights expect {in_features}")

    quantized = _as_kernel_input(quantized, np.int8, "quantized")
    scales = _as_kernel_input(scales, np.float32, "scales")
    x = _as_kernel_input(x, np.float32, "x")
    out = np.empty(out_features, dtype=np.float32)

    function = (
        _LIBRARY.nanoinfer_matvec_i8 if threaded else _LIBRARY.nanoinfer_matvec_i8_single
    )
    function(
        quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(out_features),
        ctypes.c_size_t(in_features),
    )
    return out


def matvec_i8_numpy(
    quantized: np.ndarray, scales: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """The same computation in NumPy, as the thing to check Rust against.

    This is deliberately the slow path phase 6 measured -- widen, matmul,
    scale -- because its job is to be obviously correct, not fast.
    """
    return (quantized.astype(np.float32) @ x.astype(np.float32)) * scales
