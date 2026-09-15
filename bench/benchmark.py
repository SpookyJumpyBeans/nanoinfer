"""The benchmark harness, stood up in phase 1 so every later phase feeds it.

The point of building this now, before there is anything to benchmark, is that
a performance project with no measurement history is just anecdote. Every phase
appends one JSON line to ``bench/results.jsonl``; ``--compare`` renders the
whole history as a table. Nothing is ever overwritten, so a regression is
visible rather than forgotten.

Metrics, and why each one:

  load_s          Wall time from cold file to weights usable. Matters on
                  device, and it is the only timing phase 1 can produce.
  tokenizer_tok_s Tokenizer throughput, with the Rust reference alongside
                  it. Published even though we lose badly; that gap is the
                  thing phase 7 has to close.
  ttft_ms         Time to first token: how long the prefill pass takes.
  decode_tok_s    Steady-state tokens per second after the first token. The
                  headline number, and the one the KV cache moves.
  peak_rss_mb     Peak resident memory. Decides whether the model fits.

Run:
  python -m bench.benchmark load --model models/Qwen2.5-0.5B-Instruct
  python -m bench.benchmark --compare
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoinfer.config import ModelConfig  # noqa: E402
from nanoinfer.metrics import machine_fingerprint, peak_rss_bytes  # noqa: E402
from nanoinfer.safetensors import SafeTensors  # noqa: E402
from nanoinfer.tokenizer import Tokenizer  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"


@dataclass
class Result:
    """One benchmark run. Written as a single JSON line, append-only."""

    phase: str
    scenario: str
    timestamp: str
    git_commit: str
    notes: str = ""
    load_s: float | None = None
    ttft_ms: float | None = None
    decode_tok_s: float | None = None
    tokenizer_tok_s: float | None = None
    reference_tok_s: float | None = None
    tokens_generated: int | None = None
    peak_rss_mb: float | None = None
    quantization: str = "none"
    machine: dict[str, Any] = field(default_factory=machine_fingerprint)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=10,
        )
        return out.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def record(result: Result) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(result)) + "\n")
    print(f"\nrecorded -> {RESULTS_PATH.name}")


# -- scenarios -------------------------------------------------------------
#
# A scenario is a callable that runs one measurement and returns a Result.
# Phases 3 onward register generation scenarios here; phase 1 has only the
# loader, because there is no forward pass yet.

Scenario = Callable[[argparse.Namespace], Result]


def scenario_load(args: argparse.Namespace) -> Result:
    """Measure cold-start weight loading: mmap, then materialize to float32.

    Two numbers hide in here. Mapping the file is near-instant because nothing
    is read; the cost is in widening bfloat16 to float32, which both touches
    every page and doubles the memory. Phase 6 exists to attack this.
    """
    model_dir = Path(args.model)
    cfg = ModelConfig.from_model_dir(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors in {model_dir}")

    t0 = time.perf_counter()
    stores = [SafeTensors(p) for p in shards]
    map_s = time.perf_counter() - t0
    n_tensors = sum(len(s) for s in stores)

    t1 = time.perf_counter()
    total_elems = 0
    checksum = 0.0
    for store in stores:
        for name in store.names:
            arr = store.f32(name)
            total_elems += arr.size
            # Touch the data so the OS actually pages it in; without this we
            # would be timing a lazy mapping and reporting a fantasy.
            checksum += float(arr[(0,) * arr.ndim])
    widen_s = time.perf_counter() - t1

    peak = peak_rss_bytes()
    print(f"  tensors           {n_tensors}")
    print(f"  parameters        {total_elems:,}")
    print(f"  mmap open         {map_s * 1000:8.2f} ms")
    print(f"  widen to f32      {widen_s:8.3f} s")
    print(f"  throughput        {total_elems * 2 / widen_s / 1e9:8.2f} GB/s read from bf16")
    print(f"  peak RSS          {peak / 1e6:8.1f} MB" if peak else "  peak RSS          n/a")
    print(f"  (checksum {checksum:.6f} -- forces the read, ignore the value)")

    return Result(
        phase=args.phase,
        scenario="load",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        notes=f"{cfg.architecture} {total_elems:,} params, bf16 -> f32",
        load_s=round(map_s + widen_s, 4),
        peak_rss_mb=round(peak / 1e6, 1) if peak else None,
    )


def scenario_tokenize(args: argparse.Namespace) -> Result:
    """Measure tokenizer throughput, and how far behind the reference we are.

    Publishing the losing number is the point. This is pure Python against a
    Rust implementation, so the gap is large and expected; what matters is
    that it is measured rather than hand-waved, and that it shrinks in phase 7.

    The corpus is the same 10,000 strings the correctness test uses, so speed
    and correctness are being reported over identical input.
    """
    from tests.corpus import generate

    model_dir = Path(args.model)
    corpus = generate()
    total_chars = sum(len(s) for s in corpus)

    t0 = time.perf_counter()
    tok = Tokenizer.from_model_dir(model_dir)
    load_s = time.perf_counter() - t0

    # Warm the merge cache the way real use would, then measure steady state.
    for text in corpus[:200]:
        tok.encode(text)

    t1 = time.perf_counter()
    total_tokens = sum(len(tok.encode(text)) for text in corpus)
    encode_s = time.perf_counter() - t1

    ours = total_tokens / encode_s

    ref_rate: float | None = None
    try:
        from tokenizers import Tokenizer as RefTokenizer

        ref = RefTokenizer.from_file(str(model_dir / "tokenizer.json"))
        t2 = time.perf_counter()
        ref_tokens = sum(len(ref.encode(text).ids) for text in corpus)
        ref_s = time.perf_counter() - t2
        ref_rate = ref_tokens / ref_s
        assert ref_tokens == total_tokens, "token counts diverged from the reference"
    except ImportError:
        pass

    peak = peak_rss_bytes()
    print(f"  strings           {len(corpus):,}")
    print(f"  characters        {total_chars:,}")
    print(f"  tokens            {total_tokens:,}")
    print(f"  chars per token   {total_chars / total_tokens:8.2f}")
    print(f"  tokenizer load    {load_s:8.3f} s")
    print(f"  encode            {encode_s:8.3f} s")
    print(f"  throughput        {ours:12,.0f} tok/s  (pure Python)")
    if ref_rate:
        print(f"  reference         {ref_rate:12,.0f} tok/s  (Rust)")
        print(f"  ratio             {ref_rate / ours:8.1f}x slower than reference")
    print(f"  peak RSS          {peak / 1e6:8.1f} MB" if peak else "  peak RSS          n/a")

    return Result(
        phase=args.phase,
        scenario="tokenize",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        notes=(
            f"{len(corpus):,} strings, {total_tokens:,} tokens, "
            f"{'%.1fx slower than reference' % (ref_rate / ours) if ref_rate else 'no reference'}"
        ),
        load_s=round(load_s, 4),
        tokenizer_tok_s=round(ours, 1),
        reference_tok_s=round(ref_rate, 1) if ref_rate else None,
        tokens_generated=total_tokens,
        peak_rss_mb=round(peak / 1e6, 1) if peak else None,
    )


def scenario_generate(args: argparse.Namespace) -> Result:
    """Measure prefill and decode throughput.

    Not implemented until phase 3 produces a forward pass. It is declared now
    so the harness shape is fixed and later phases only fill in the body.
    """
    raise SystemExit(
        "scenario 'generate' needs a forward pass (phase 3). "
        "Run 'load' until then."
    )


SCENARIOS: dict[str, Scenario] = {
    "load": scenario_load,
    "tokenize": scenario_tokenize,
    "generate": scenario_generate,
}


# -- comparison table ------------------------------------------------------


def compare() -> int:
    if not RESULTS_PATH.exists():
        print("no results recorded yet")
        return 1

    rows = [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("no results recorded yet")
        return 1

    headers = ["phase", "scenario", "quant", "load_s", "ttft_ms", "gen tok/s",
               "tokenizer tok/s", "ref tok/s", "peak MB", "commit", "when"]
    table = []
    for r in rows:
        table.append([
            r.get("phase", ""),
            r.get("scenario", ""),
            r.get("quantization", ""),
            _fmt(r.get("load_s"), "{:.2f}"),
            _fmt(r.get("ttft_ms"), "{:.1f}"),
            _fmt(r.get("decode_tok_s"), "{:.2f}"),
            _fmt(r.get("tokenizer_tok_s"), "{:,.0f}"),
            _fmt(r.get("reference_tok_s"), "{:,.0f}"),
            _fmt(r.get("peak_rss_mb"), "{:.0f}"),
            r.get("git_commit", ""),
            (r.get("timestamp") or "")[:16].replace("T", " "),
        ])

    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return 0


def _fmt(value: object, spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return spec.format(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", choices=sorted(SCENARIOS), help="what to measure")
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct", help="model directory")
    parser.add_argument("--phase", default="1", help="phase label recorded with the result")
    parser.add_argument("--prompt", default="The capital of France is", help="prompt for generation scenarios")
    parser.add_argument("--tokens", type=int, default=64, help="tokens to generate")
    parser.add_argument("--compare", action="store_true", help="print the recorded history and exit")
    parser.add_argument("--dry-run", action="store_true", help="measure but do not record")
    args = parser.parse_args(argv)

    if args.compare:
        return compare()
    if not args.scenario:
        parser.error("give a scenario, or --compare")

    print(f"scenario: {args.scenario}   model: {args.model}   phase: {args.phase}")
    print("-" * 60)
    result = SCENARIOS[args.scenario](args)
    if args.dry_run:
        print("\n--dry-run: not recorded")
    else:
        record(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
