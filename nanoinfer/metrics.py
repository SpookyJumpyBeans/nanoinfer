"""Timing and memory measurement, with no third-party dependencies.

Peak resident set size is the number that matters for an inference engine --
it is what decides whether a model fits on a device -- and it is the one thing
the standard library exposes differently on every platform. So we query the OS
directly rather than pulling in psutil for one call.
"""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or None if unavailable.

    Peak, not current: transient allocations during weight loading are exactly
    what we want to catch, and a current-usage reading would miss them.
    """
    if sys.platform == "win32":
        return _peak_rss_windows()
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Windows, non-POSIX
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


class _ProcessMemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS from psapi.h."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _peak_rss_windows() -> int | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # The signatures must be declared. GetCurrentProcess returns the
        # pseudo-handle (HANDLE)-1; with ctypes' default c_int restype that
        # becomes 0x00000000FFFFFFFF on 64-bit instead of the all-ones handle
        # the API expects, and the call fails with ERROR_INVALID_HANDLE.
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p

        get_memory_info = kernel32.K32GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_uint32,
        ]
        get_memory_info.restype = ctypes.c_int

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except (OSError, AttributeError):  # pragma: no cover
        return None


@dataclass
class Timer:
    """Accumulates named wall-clock spans."""

    spans: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spans[name] = self.spans.get(name, 0.0) + (time.perf_counter() - start)

    def __getitem__(self, name: str) -> float:
        return self.spans[name]


def machine_fingerprint() -> dict[str, object]:
    """Enough about the host that a benchmark number means something later.

    Numbers recorded without this are not comparable across machines, and a
    README that publishes tokens/sec without saying what it ran on is not
    reporting a result.
    """
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
