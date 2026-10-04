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

from nanoinfer import kernels  # noqa: E402
from nanoinfer.generate import greedy  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.quantization import quantize_model  # noqa: E402
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
    fp32_weights = ModelWeights.load(args.model)
    model = Qwen2(fp32_weights)

    # The int8 engine belongs in this comparison, not in a separate tool. It is
    # what llama.cpp's Q8_0 should be measured against -- both hold integer
    # weights and read a quarter of the bytes -- and comparing numbers taken
    # from two different runs on two different days is how a 2.3x spread gets
    # mistaken for a 1.1x win.
    int8_weights, _ = quantize_model(fp32_weights, bits=8, dequantize=False)
    int8_model = Qwen2(int8_weights)

    # Every observation, so the spread is visible. A minimum alone cannot say
    # whether it is the floor of a tight distribution or the lucky end of a
    # wide one, and that distinction is the whole question here.
    seen: dict[str, dict[str, list[float]]] = {}

    def record(name: str, prefill_ms: float, prefill_n: int, decode_ms: float, decode_n: int):
        slot = seen.setdefault(name, {"prefill": [], "decode": []})
        slot["prefill"].append(prefill_ms / max(prefill_n, 1))
        slot["decode"].append(decode_ms / max(decode_n, 1))

    def record_engine(name: str, engine: Qwen2, vnni: bool = False):
        # The VNNI kernel is global state in the bridge, so it is turned on
        # only around its own timing and off again immediately -- otherwise
        # whichever contender ran next would silently inherit it.
        if vnni and not kernels.use_vnni(True):
            raise RuntimeError("asked for VNNI, but this CPU has none")
        try:
            result = greedy(engine, prompt_ids, max_new_tokens=args.tokens)
        finally:
            kernels.use_vnni(False)
        generated = len(result.generated_ids)
        record(
            name,
            result.prefill_s * 1e3, len(prompt_ids),
            result.decode_s * 1e3, max(generated - 1, 1),
        )

    # Warm up at the measured size before timing anything, so no contender
    # pays thread-pool spin-up inside a timed rep.
    for engine in (model, int8_model):
        greedy(engine, prompt_ids, max_new_tokens=2)
    for name, gguf, threads in contenders:
        llama_run(completion, gguf, PROMPT, 2, threads)

    print(f"alternating {args.reps} reps, {args.tokens} tokens each\n")
    for rep in range(args.reps):
        record_engine("nanoinfer f32", model)
        record_engine("nanoinfer int8", int8_model)
        if kernels.vnni_available():
            record_engine("nanoinfer vnni", int8_model, vnni=True)
        for name, gguf, threads in contenders:
            timing = llama_run(completion, gguf, PROMPT, args.tokens, threads)
            record(
                name,
                timing["prefill_ms"], timing["prefill_tokens"],
                timing["decode_ms"], timing["decode_tokens"],
            )
        print(f"  rep {rep + 1}/{args.reps} done", flush=True)

    best = {
        name: {phase: min(values) for phase, values in phases.items()}
        for name, phases in seen.items()
    }

    print(f"\n{platform.processor() or platform.machine()}")
    print(
        f"\n{'engine':<18} {'threads':>8} {'prefill ms/tok':>15} "
        f"{'decode ms/tok':>14} {'spread':>8} {'decode tok/s':>13}"
    )
    print("-" * 82)
    rows = [("nanoinfer f32", str(blas_threads)),
            ("nanoinfer int8", str(blas_threads))]
    if kernels.vnni_available():
        rows.append(("nanoinfer vnni", str(blas_threads)))
    rows += [(name, str(threads)) for name, _, threads in contenders]
    spreads = {}
    for name, threads in rows:
        slot = best[name]
        observed = seen[name]["decode"]
        spreads[name] = max(observed) / min(observed)
        print(
            f"{name:<18} {threads:>8} {slot['prefill']:>15.2f} "
            f"{slot['decode']:>14.2f} {spreads[name]:>7.2f}x "
            f"{1e3 / slot['decode']:>13.2f}"
        )
    print("-" * 82)

    # The comparison this tool exists for, stated only as strongly as the
    # measurement supports it. A margin inside the spread of either contender
    # is not a result, however much one would prefer it to be.
    mine = "nanoinfer vnni" if "nanoinfer vnni" in best else "nanoinfer int8"
    ours = best[mine]["decode"]
    noisiest = max(spreads[mine], spreads.get("llama.cpp Q8_0", 1.0))
    for name, _, _ in contenders:
        if name != "llama.cpp Q8_0":
            continue
        theirs = best[name]["decode"]
        margin = theirs / ours if ours < theirs else ours / theirs
        leader = mine if ours < theirs else name
        print(f"{mine} {ours:.2f} ms/tok against {name} {theirs:.2f} ms/tok")
        if margin < noisiest:
            print(f"  {leader} leads by {margin:.2f}x, but the widest spread "
                  f"seen was {noisiest:.2f}x -- inside the noise, so this is "
                  f"NOT yet a result. Quiet the machine and raise --reps.")
        else:
            print(f"  {leader} leads by {margin:.2f}x, wider than the "
                  f"{noisiest:.2f}x spread -- this one holds.")

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
