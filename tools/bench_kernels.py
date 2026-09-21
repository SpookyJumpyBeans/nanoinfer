"""Rust kernels against NumPy, on the shapes the engine actually decodes with.

    python -m tools.bench_kernels

The Rust crate has its own benchmark, but its fp32 column is the crate's own
scalar loop -- a fair baseline for the SIMD work and a meaningless one for the
project. The opponent that matters is the NumPy the engine really runs, which
is OpenBLAS with every core. This is that comparison.

Two things it does deliberately:

**Alternating A/B, not one block each.** Phase 4 produced a 15x "cliff" on this
machine that moved to a different matrix size on every rerun and turned out to
be nothing at all. Running A then B measures the machine's mood as much as the
code; interleaving them means a stall lands on both.

**Minima, not means.** This laptop stalls for seconds at a time under load. A
mean measures the stalls and a minimum measures the code.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from nanoinfer import kernels  # noqa: E402
from nanoinfer.quantization import quantize_int8  # noqa: E402

# Every linear projection Qwen2.5-0.5B runs per decoded token.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("q_proj  896x896", 896, 896),
    ("k_proj  128x896", 128, 896),
    ("v_proj  128x896", 128, 896),
    ("o_proj  896x896", 896, 896),
    ("gate   4864x896", 4864, 896),
    ("up     4864x896", 4864, 896),
    ("down   896x4864", 896, 4864),
)


def alternating(contenders: dict, runs: int) -> dict[str, float]:
    """Time several callables by interleaving them, and return the minima."""
    for call in contenders.values():
        call()                                  # warm caches and page in

    best = {name: float("inf") for name in contenders}
    for _ in range(runs):
        for name, call in contenders.items():
            started = time.perf_counter()
            call()
            best[name] = min(best[name], time.perf_counter() - started)
    return {name: seconds * 1e3 for name, seconds in best.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--layers", type=int, default=24, help="to scale to one token")
    parser.add_argument("--out", type=Path, default=Path("bench/kernels.jsonl"))
    args = parser.parse_args(argv)

    report = kernels.describe()
    if not report["available"]:
        print("the Rust kernels are not built.\n  cd rust && cargo build --release")
        return 1

    print(f"{platform.processor() or platform.machine()}")
    print(f"avx2 {report['avx2']}, {report['threads']} threads, best of {args.runs}\n")
    print(f"{'shape':<18} {'numpy fp32':>11} {'rust 1thr':>10} {'rust mt':>9} {'speedup':>9}")
    print("-" * 61)

    rng = np.random.default_rng(3)
    totals = {"numpy": 0.0, "single": 0.0, "threaded": 0.0}
    rows = []

    for name, out_features, in_features in SHAPES:
        weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
        x = rng.standard_normal(in_features).astype(np.float32)
        q = quantize_int8(weights)

        timings = alternating(
            {
                "numpy": lambda: weights @ x,
                "single": lambda: kernels.matvec_i8(
                    q.values, q.scales, x, threaded=False
                ),
                "threaded": lambda: kernels.matvec_i8(q.values, q.scales, x),
            },
            args.runs,
        )

        for key in totals:
            totals[key] += timings[key]
        speedup = timings["numpy"] / timings["threaded"]

        print(
            f"{name:<18} {timings['numpy']:>11.3f} {timings['single']:>10.3f} "
            f"{timings['threaded']:>9.3f} {speedup:>8.2f}x"
        )
        rows.append({"shape": name, "out": out_features, "in": in_features, **timings})

    print("-" * 61)
    print(
        f"{'one layer':<18} {totals['numpy']:>11.3f} {totals['single']:>10.3f} "
        f"{totals['threaded']:>9.3f} {totals['numpy'] / totals['threaded']:>8.2f}x"
    )
    print(
        f"\n{args.layers} layers, per decoded token: "
        f"numpy {totals['numpy'] * args.layers:.1f} ms, "
        f"rust {totals['threaded'] * args.layers:.1f} ms"
    )
    print(
        "\nProjections only. Attention, norms and the LM head are still NumPy,\n"
        "so this is the ceiling on what phase 7 can move, not a token rate."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "runs": args.runs,
                    "avx2": report["avx2"],
                    "threads": report["threads"],
                    "layers": args.layers,
                    "shapes": rows,
                    "layer_total_ms": totals,
                }
            )
            + "\n"
        )
    print(f"\nrecorded -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
