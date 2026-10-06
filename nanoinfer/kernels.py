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

# Worker threads for the Rust kernels, unless NANOINFER_THREADS says
# otherwise. See the comment where the pool is sized in _load().
DEFAULT_THREADS = 4


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

        # The engine's entry point takes raw addresses (c_void_p) rather than
        # typed pointers. Building a POINTER object per array with data_as()
        # was ~40% of an otherwise empty call -- 169 calls a token, ~3 ms --
        # and the typing it buys is already enforced by _as_kernel_input.
        for symbol in ("nanoinfer_matmul_i8", "nanoinfer_matmul_i8_vnni"):
            function = getattr(library, symbol)
            function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 3
            function.restype = None
        library.nanoinfer_matmul_i8_vnni_blocked.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 4
        )
        library.nanoinfer_matmul_i8_vnni_blocked.restype = None

        library.nanoinfer_has_avx2.argtypes = []
        library.nanoinfer_has_avx2.restype = ctypes.c_int
        library.nanoinfer_has_vnni.argtypes = []
        library.nanoinfer_has_vnni.restype = ctypes.c_int
        library.nanoinfer_num_threads.argtypes = []
        library.nanoinfer_num_threads.restype = ctypes.c_int
        library.nanoinfer_set_threads.argtypes = [ctypes.c_size_t]
        library.nanoinfer_set_threads.restype = ctypes.c_int

        # Size the worker pool before anything can touch it. Rayon builds its
        # global pool once, on first use, at one worker per logical CPU -- 20
        # here, and the slowest setting measured: the even row split leaves the
        # six P-cores waiting on the eight E-cores, 97 times per token. Four
        # was fastest in a sweep on the real model (32.77 ms/token against
        # 55.42 at 20). NANOINFER_THREADS overrides it, since the best count
        # is a property of the machine, not of the code.
        requested = int(os.environ.get("NANOINFER_THREADS", DEFAULT_THREADS))
        library.nanoinfer_set_threads(requested)
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


def matmul_i8(
    quantized: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    allow_vnni: bool = True,
) -> np.ndarray:
    """``x @ (quantized * scales[:, None]).T`` for a batch of tokens, in Rust.

    ``x`` is ``[tokens, in_features]`` and the result ``[tokens,
    out_features]`` -- the shape ``x @ W.T`` has, so the forward pass can call
    this wherever it would have written that.

    One call per projection regardless of the batch size. Prefill would
    otherwise cross the ctypes boundary once per prompt token per projection,
    and each crossing costs ~8 us before any work is done. A batch of one is
    a decode step and is bitwise the threaded :func:`matvec_i8`.
    """
    if _LIBRARY is None:
        raise RuntimeError(
            "the Rust kernels are not built; run `cargo build --release` in rust/"
        )
    if quantized.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {quantized.shape}")
    if x.ndim != 2:
        raise ValueError(f"expected [tokens, in_features] activations, got {x.shape}")

    out_features, in_features = quantized.shape
    tokens = x.shape[0]
    if scales.shape != (out_features,):
        raise ValueError(
            f"expected one scale per output row: {scales.shape} against {out_features}"
        )
    if x.shape[1] != in_features:
        raise ValueError(f"x has width {x.shape[1]}, weights expect {in_features}")
    if tokens == 0:
        return np.empty((0, out_features), dtype=np.float32)

    quantized = _as_kernel_input(quantized, np.int8, "quantized")
    scales = _as_kernel_input(scales, np.float32, "scales")
    x = _as_kernel_input(x, np.float32, "x")

    # Rust writes [out_features, tokens] so that each worker's rows are one
    # contiguous block; the transpose back is a view, not a copy.
    out_t = np.empty((out_features, tokens), dtype=np.float32)
    use_vnni_here = _USE_VNNI and allow_vnni
    if use_vnni_here and _VNNI_BLOCK:
        _LIBRARY.nanoinfer_matmul_i8_vnni_blocked(
            quantized.ctypes.data,
            scales.ctypes.data,
            x.ctypes.data,
            out_t.ctypes.data,
            out_features,
            in_features,
            tokens,
            _VNNI_BLOCK,
        )
        return out_t.T

    entry = (
        _LIBRARY.nanoinfer_matmul_i8_vnni
        if use_vnni_here
        else _LIBRARY.nanoinfer_matmul_i8
    )
    entry(
        quantized.ctypes.data,
        scales.ctypes.data,
        x.ctypes.data,
        out_t.ctypes.data,
        out_features,
        in_features,
        tokens,
    )
    return out_t.T


# Off by default, and deliberately not a silent upgrade. The VNNI kernel
# quantizes activations to int8 so VPDPBUSD can take them, which costs about
# one part in a hundred against the float-activation path. Leaving it off keeps
# the engine bit-comparable with the simulated quantization the perplexity
# numbers were measured on; this is opted into for timing, and for use once
# that quality cost has been measured end to end rather than assumed.
_USE_VNNI = False

# Activation values per scale. 32 is what llama.cpp's Q8_0 and Q8_1 use, and
# where this model's error stops improving: one scale per row costs +5.42%
# perplexity against an fp32 baseline, per 128 costs +1.37%, per 32 costs
# +0.46%, and per 16 measured no better than per 32. Zero means one scale for
# the whole row, which is what the first VNNI kernel did.
_VNNI_BLOCK = 32


def vnni_available() -> bool:
    """Whether the loaded library can run the VNNI kernel on this CPU."""
    return _LIBRARY is not None and bool(_LIBRARY.nanoinfer_has_vnni())


# Whether the LM head may use VNNI. Off by default: it is the one projection
# whose activation error lands straight on the logits with no hidden dimension
# to average it over, and it is memory-bound at 136M weights, so VNNI buys it
# only about 1.06x. Keeping it on float activations is close to free in speed
# and removes the error that matters most.
_VNNI_LM_HEAD = False


def vnni_lm_head(enabled: bool) -> bool:
    """Let the LM head use VNNI too. Returns what took effect."""
    global _VNNI_LM_HEAD
    _VNNI_LM_HEAD = bool(enabled)
    return _VNNI_LM_HEAD


def lm_head_allows_vnni() -> bool:
    return _VNNI_LM_HEAD


def vnni_block(values_per_scale: int) -> int:
    """Set how many activation values share a scale; returns what took effect.

    Zero or a value at least as wide as a row gives one scale per row, which is
    measurably worse on this model: these activations run to 28.9x the median on
    the 896-wide inputs and 68.3x on the 4864-wide ones, so a single scale is
    set by the outlier and crushes everything beside it.
    """
    global _VNNI_BLOCK
    _VNNI_BLOCK = max(int(values_per_scale), 0)
    return _VNNI_BLOCK


def use_vnni(enabled: bool) -> bool:
    """Route :func:`matmul_i8` through the VNNI kernel; returns what took effect.

    Asking for it on a CPU without AVX-VNNI leaves it off rather than
    pretending: the caller gets ``False`` back and keeps the float-activation
    kernel, which is the more accurate of the two.
    """
    global _USE_VNNI
    _USE_VNNI = bool(enabled) and vnni_available()
    return _USE_VNNI


def matvec_i8_numpy(
    quantized: np.ndarray, scales: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """The same computation in NumPy, as the thing to check Rust against.

    This is deliberately the slow path phase 6 measured -- widen, matmul,
    scale -- because its job is to be obviously correct, not fast.
    """
    return (quantized.astype(np.float32) @ x.astype(np.float32)) * scales
