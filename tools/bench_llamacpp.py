"""nanoinfer against llama.cpp, same weights, same machine.

    python -m tools.bench_llamacpp --llama-cpp C:/Users/daih0/llama.cpp

Run tools.compare_llamacpp first. This times two engines; that one establishes
they are computing the same thing, without which a timing is meaningless.

**Decode and prefill are reported separately, and load is excluded.** Both
engines report the three phases independently, and they behave completely
differently: prefill is one matmul over the whole prompt and is compute-bound,
decode is one matvec per token and is memory-bound. Averaging them hides the
only interesting number. llama.cpp also reloads its 2 GB model on every
invocation while this engine keeps it resident, so any total-time comparison
would mostly measure that.

**Alternating A/B, minima reported.** Phase 4 produced a 15x "cliff" on this
laptop that moved to a different matrix size on every rerun and turned out to
be nothing. Running all of A then all of B measures the machine's mood as much
as the code; interleaving means a stall lands on both. Minima rather than
means, for the same reason.

**Every engine runs at its own best thread count, found by sweeping.** This is
not a detail. The CPU is Alder Lake: 6 fast P-cores and 8 slow E-cores behind
20 logical threads, and splitting a memory-bound matvec evenly across them
leaves the fast cores waiting on the slow ones. Handing everything "-t 20"
made llama.cpp Q8_0 report 264 ms/token; at "-t 1" the same build reports 47.
Sweeping only the opponent would have been worse than not sweeping at all, so
both sides get it -- this engine's own default of all 20 BLAS threads was also
costing it 50%, 181 ms/token against 121 at six.

Defaults measured, per token, decode only:

    llama.cpp Q8_0    t=1   47.23    t=4   80.12    t=20  264.53
    llama.cpp f32     t=3  127.30    t=4  166.95    t=20  278.12
    nanoinfer f32     t=6  121.44    t=4  124.14    t=20  181.09

**Q8_0 is included but is not a like-for-like row.** llama.cpp decodes it with
real integer kernels. This engine cannot: phase 6 measured NumPy's int8 path at
30-80x *slower* than float32, because widening costs more than the matmul it
feeds, and phase 7's Rust kernels are not wired into the forward pass. The row
is there to show the size of the prize, not to claim a matching capability.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoinfer.generate import greedy  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.tokenizer import Tokenizer  # noqa: E402
from nanoinfer.weights import ModelWeights  # noqa: E402

PROMPT = "The capital of France is"

# "prompt eval time =  1367.93 ms /   5 tokens (  273.59 ms per token, 3.66 tokens per second)"
PREFILL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
DECODE = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs")


def llama_run(binary: Path, gguf: Path, prompt: str, n: int, threads: int) -> dict:
    """One llama.cpp generation, returning its own reported phase timings."""
    result = subprocess.run(
        [
            str(binary), "-m", str(gguf), "-p", prompt, "-n", str(n),
            "--temp", "0", "--top-k", "1", "--top-p", "1.0", "--min-p", "0.0",
            "--repeat-penalty", "1.0",      # the GGUF metadata says 1.1; see compare_llamacpp
            "--seed", "0", "-t", str(threads),
            "-no-cnv", "--no-display-prompt", "--log-colors", "off",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    # llama.cpp reports its own timings; using them rather than wall clock keeps
    # its 4 s model load out of the comparison, which is the fair thing to do.
    prefill = PREFILL.search(result.stderr)
    decode = DECODE.search(result.stderr)
    if not (prefill and decode):
        raise RuntimeError(f"could not parse llama.cpp timings:\n{result.stderr[-800:]}")
    return {
        "prefill_ms": float(prefill.group(1)),
        "prefill_tokens": int(prefill.group(2)),
        "decode_ms": float(decode.group(1)),
        "decode_tokens": int(decode.group(2)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument(
        "--gguf", type=Path, default=Path("models/qwen2.5-0.5b-instruct-f32.gguf")
    )
    parser.add_argument(
        "--gguf-q8", type=Path, default=Path("models/qwen2.5-0.5b-instruct-q8_0.gguf")
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--reps", type=int, default=3)
    # Each engine's sweep optimum; see the module docstring for the sweep.
    parser.add_argument("--threads-f32", type=int, default=3)
    parser.add_argument("--threads-q8", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("bench/llamacpp_timing.jsonl"))
    args = parser.parse_args(argv)

    completion = args.llama_cpp / "build" / "bin" / "llama-completion.exe"
    if not completion.exists():
        print(f"missing: {completion}")
        return 1

    contenders = [("llama.cpp f32", args.gguf, args.threads_f32)]
    if args.gguf_q8.exists():
        contenders.append(("llama.cpp Q8_0", args.gguf_q8, args.threads_q8))
    else:
        print(f"note: {args.gguf_q8} not found, skipping the Q8_0 row\n")

    # This engine's thread count is OpenBLAS's, fixed when numpy loaded, so it
    # is reported rather than set -- run the whole tool under
    # OPENBLAS_NUM_THREADS to change it.
    blas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "default (all)")

    tokenizer = Tokenizer.from_model_dir(args.model)
    prompt_ids = list(tokenizer.encode(PROMPT))
    print("loading weights...", flush=True)
    model = Qwen2(ModelWeights.load(args.model))

    # Minimum ms-per-token seen, which is the cleanest read on the code.
    best: dict[str, dict[str, float]] = {}

    def record(name: str, prefill_ms: float, prefill_n: int, decode_ms: float, decode_n: int):
        slot = best.setdefault(name, {"prefill": float("inf"), "decode": float("inf")})
        slot["prefill"] = min(slot["prefill"], prefill_ms / max(prefill_n, 1))
        slot["decode"] = min(slot["decode"], decode_ms / max(decode_n, 1))

    print(f"alternating {args.reps} reps, {args.tokens} tokens each\n")
    for rep in range(args.reps):
        result = greedy(model, prompt_ids, max_new_tokens=args.tokens)
        generated = len(result.generated_ids)
        record(
            "nanoinfer f32",
            result.prefill_s * 1e3, len(prompt_ids),
            result.decode_s * 1e3, max(generated - 1, 1),
        )
        for name, gguf, threads in contenders:
            timing = llama_run(completion, gguf, PROMPT, args.tokens, threads)
            record(
                name,
                timing["prefill_ms"], timing["prefill_tokens"],
                timing["decode_ms"], timing["decode_tokens"],
            )
        print(f"  rep {rep + 1}/{args.reps} done", flush=True)

    print(f"\n{platform.processor() or platform.machine()}")
    print(
        f"\n{'engine':<18} {'threads':>8} {'prefill ms/tok':>15} "
        f"{'decode ms/tok':>14} {'decode tok/s':>13}"
    )
    print("-" * 73)
    baseline = best["nanoinfer f32"]["decode"]
    rows = [("nanoinfer f32", str(blas_threads))]
    rows += [(name, str(threads)) for name, _, threads in contenders]
    for name, threads in rows:
        slot = best[name]
        print(
            f"{name:<18} {threads:>8} {slot['prefill']:>15.2f} "
            f"{slot['decode']:>14.2f} {1e3 / slot['decode']:>13.2f}"
        )
    print("-" * 73)
    for name, _, _ in contenders:
        ratio = baseline / best[name]["decode"]
        if ratio >= 1:
            print(f"{name} decodes {ratio:.2f}x faster than nanoinfer")
        else:
            print(f"{name} decodes {1 / ratio:.2f}x SLOWER than nanoinfer")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "tokens": args.tokens,
                    "reps": args.reps,
                    "llama_threads_f32": args.threads_f32,
                    "llama_threads_q8": args.threads_q8,
                    "nanoinfer_blas_threads": blas_threads,
                    "ms_per_token": best,
                }
            )
            + "\n"
        )
    print(f"\nrecorded -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
