"""Measure what quantization costs: perplexity, footprint, error.

    python -m tools.measure_quantization fp32
    python -m tools.measure_quantization int8
    python -m tools.measure_quantization int4 --group-size 32
    python -m tools.measure_quantization int8 --quantize-embeddings

One level per invocation, deliberately. Holding the float32 weights and a
quantized copy at once is about 4 GB, and running the sweep in a single process
would page rather than measure.

The perplexity figure comes from simulated quantization: weights are quantized
and immediately dequantized, so the model computes on exactly the values the
integer format can represent while the matmuls stay float32. That reproduces
the arithmetic of true integer storage exactly. It does not reproduce its
speed, which NumPy cannot deliver -- see the module docstring in
nanoinfer/quantization.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from nanoinfer.metrics import peak_rss_bytes  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.perplexity import evaluate, held_out_tokens  # noqa: E402
from nanoinfer.quantization import (  # noqa: E402
    quantization_error,
    quantize_model,
)
from nanoinfer.tokenizer import Tokenizer  # noqa: E402
from nanoinfer.weights import ModelWeights  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("level", choices=["fp32", "int8", "int4"])
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--group-size", type=int, default=128, help="INT4 only")
    parser.add_argument("--quantize-embeddings", action="store_true")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("bench/quantization.jsonl"))
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.from_model_dir(args.model)
    token_ids = held_out_tokens(tokenizer, limit=args.tokens)

    weights = ModelWeights.load(args.model)
    fp32_bytes = weights.nbytes

    report = None
    error = None
    if args.level != "fp32":
        bits = 8 if args.level == "int8" else 4
        # Keep one tensor's original values to report the round-trip error on.
        probe_before = weights.layers[0].gate_proj_weight.copy()

        started = time.perf_counter()
        weights, report = quantize_model(
            weights,
            bits=bits,
            quantize_embeddings=args.quantize_embeddings,
            group_size=args.group_size,
        )
        quantize_s = time.perf_counter() - started

        error = quantization_error(probe_before, weights.layers[0].gate_proj_weight)
        print(f"  quantized in     {quantize_s:8.2f} s")
        print(f"  {report}")
        print(f"  layer0 gate_proj round-trip: "
              f"rel Frobenius {error['rel_frobenius']:.5f}, "
              f"max abs {error['max_abs']:.6f}")

    model = Qwen2(weights)

    started = time.perf_counter()
    result = evaluate(model, token_ids, chunk_size=args.chunk_size)
    evaluate_s = time.perf_counter() - started

    peak = peak_rss_bytes()
    print(f"  {result}")
    print(f"  scored in        {evaluate_s:8.2f} s")
    print(f"  peak RSS         {peak / 1e6:8.1f} MB" if peak else "")

    row = {
        "level": args.level,
        "group_size": args.group_size if args.level == "int4" else None,
        "quantize_embeddings": args.quantize_embeddings,
        "perplexity": result.perplexity,
        "mean_nll": result.mean_nll,
        "tokens_scored": result.tokens_scored,
        "chunk_size": args.chunk_size,
        "fp32_bytes": fp32_bytes,
        "stored_bytes": (fp32_bytes - report.original_bytes + report.quantized_bytes)
        if report
        else fp32_bytes,
        "rel_frobenius": error["rel_frobenius"] if error else 0.0,
        "peak_rss_mb": round(peak / 1e6, 1) if peak else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"\nrecorded -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
