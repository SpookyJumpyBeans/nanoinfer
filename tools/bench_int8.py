"""Whole-model decode, fp32 against int8 weights held as int8.

    python -m tools.bench_int8                       # the real model
    python -m tools.bench_int8 --synthetic           # random weights, real shapes

Phase 8 put a number on the job left over from phase 7: llama.cpp decodes Q8_0
at ~86 ms/token where this engine does ~152 at f32, and the int8 kernels that
should close that gap were written, tested and not wired in. This measures
whether wiring them in did.

Three models, from the same weights, timed by alternating them one round at a
time (the phase 4 lesson -- a stall must land on every contender, not one):

  fp32          the engine as it was: OpenBLAS on float32
  int8          every linear projection int8, through the Rust kernel;
                the embedding matrix and LM head stay fp32
  int8+embed    the embedding matrix too, which is also the LM head -- and at
                151,936 x 896 the single biggest read in a decode step

Decode is reported as the median milliseconds per step, prefill as the median
of whole-prompt passes. Medians, because this machine stalls.

``--synthetic`` builds random weights with Qwen2.5-0.5B's exact shapes, for
machines without the download. Timing depends on shapes, not values, so the
speed numbers carry over; nothing about quality does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from nanoinfer.config import ModelConfig  # noqa: E402
from nanoinfer.linear import backend  # noqa: E402
from nanoinfer.metrics import machine_fingerprint  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.quantization import quantize_model  # noqa: E402
from nanoinfer.weights import LayerWeights, ModelWeights  # noqa: E402

QWEN25_05B = {
    "architectures": ["Qwen2ForCausalLM"],
    "vocab_size": 151936,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}


def synthetic_weights(seed: int = 0) -> ModelWeights:
    """Random float32 weights shaped exactly like Qwen2.5-0.5B's."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(QWEN25_05B), encoding="utf-8")
        config = ModelConfig.from_json(path)

    rng = np.random.default_rng(seed)

    def w(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float32)

    h = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    ones = np.ones(h, dtype=np.float32)

    layers = tuple(
        LayerWeights(
            input_layernorm=ones,
            q_proj_weight=w(q, h), q_proj_bias=w(q),
            k_proj_weight=w(kv, h), k_proj_bias=w(kv),
            v_proj_weight=w(kv, h), v_proj_bias=w(kv),
            o_proj_weight=w(h, q),
            post_attention_layernorm=ones,
            gate_proj_weight=w(inter, h),
            up_proj_weight=w(inter, h),
            down_proj_weight=w(h, inter),
        )
        for _ in range(config.num_hidden_layers)
    )
    return ModelWeights(
        config=config,
        embed_tokens=w(config.vocab_size, h),
        layers=layers,
        final_norm=ones,
        _lm_head=None,
    )


def time_one(model: Qwen2, prompt: np.ndarray, steps: int) -> tuple[float, list[float]]:
    """One prefill, then ``steps`` decode steps; seconds for each."""
    cache = model.new_cache(len(prompt) + steps + 1)

    started = time.perf_counter()
    logits = model.next_token_logits(prompt, cache=cache)
    prefill = time.perf_counter() - started

    decode = []
    for _ in range(steps):
        token = np.array([int(np.argmax(logits))])
        started = time.perf_counter()
        logits = model.next_token_logits(token, cache=cache)
        decode.append(time.perf_counter() - started)
    return prefill, decode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--synthetic", action="store_true",
                        help="random weights with the real shapes; no download needed")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=32, help="decode steps per round")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("bench/int8_decode.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="measure but do not record")
    args = parser.parse_args(argv)

    print(f"int8 backend: {backend()}", file=sys.stderr)
    started = time.perf_counter()
    weights = synthetic_weights() if args.synthetic else ModelWeights.load(args.model)
    print(f"loaded in {time.perf_counter() - started:.1f}s", file=sys.stderr)

    int8, _ = quantize_model(weights, bits=8, dequantize=False)
    int8_embed, _ = quantize_model(weights, bits=8, quantize_embeddings=True, dequantize=False)
    contenders = {
        "fp32": Qwen2(weights),
        "int8": Qwen2(int8),
        "int8+embed": Qwen2(int8_embed),
    }
    footprint = {name: model.weights.nbytes for name, model in contenders.items()}

    rng = np.random.default_rng(1)
    prompt = rng.integers(0, weights.config.vocab_size, args.prompt_tokens)

    for model in contenders.values():                 # warm up, page in
        time_one(model, prompt[:4], 2)

    prefill = {name: [] for name in contenders}
    decode = {name: [] for name in contenders}
    for _ in range(args.rounds):
        for name, model in contenders.items():
            p, d = time_one(model, prompt, args.tokens)
            prefill[name].append(p)
            decode[name].extend(d)

    base = statistics.median(decode["fp32"])
    print(f"\n{'':<12} {'weights':>9} {'decode ms/tok':>14} {'vs fp32':>8} "
          f"{'prefill ms/tok':>15}")
    rows = []
    for name in contenders:
        step = statistics.median(decode[name])
        pre = statistics.median(prefill[name]) / args.prompt_tokens
        print(f"{name:<12} {footprint[name] / 1e9:>7.2f}GB {step * 1e3:>14.2f} "
              f"{base / step:>7.2f}x {pre * 1e3:>15.2f}")
        rows.append({
            "contender": name,
            "weights_bytes": footprint[name],
            "decode_ms": step * 1e3,
            "decode_p10_ms": float(np.percentile(decode[name], 10)) * 1e3,
            "decode_p90_ms": float(np.percentile(decode[name], 90)) * 1e3,
            "prefill_ms_per_token": pre * 1e3,
        })

    if args.dry_run:
        return 0
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": "synthetic" if args.synthetic else str(args.model),
        "backend": backend(),
        "prompt_tokens": args.prompt_tokens,
        "decode_steps": args.tokens * args.rounds,
        "machine": machine_fingerprint(),
        "results": rows,
    }
    with args.out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
