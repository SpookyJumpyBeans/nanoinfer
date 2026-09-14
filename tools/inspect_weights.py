"""Phase 1: read the weight file and prove we understand it.

No inference happens here. The goal is to demonstrate three things:

1. We can parse the safetensors container ourselves, byte by byte.
2. Every tensor in the file is one we expected, with the exact shape the
   config implies -- no surprises left to discover during the forward pass.
3. We know which tensors are *missing* and why. For Qwen2.5 the notable
   absence is ``lm_head.weight``, because the model ties its output projection
   to its input embedding matrix.

Run:  python -m tools.inspect_weights models/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pathlib import Path

# Make `python tools/inspect_weights.py` work as well as `-m tools....`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoinfer.config import ModelConfig  # noqa: E402
from nanoinfer.safetensors import SafeTensors  # noqa: E402


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value):,} B"
        value /= 1024
    return f"{value:.1f} TiB"


def expected_tensors(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """The complete tensor manifest implied by the config, built from scratch.

    Writing this out by hand is the point of the exercise. If the file matches
    it exactly, we have understood the architecture before running a single
    matmul.
    """
    h = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    manifest: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, h),
        "model.norm.weight": (h,),
    }

    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        manifest.update(
            {
                # Pre-attention RMSNorm gain (no bias: RMSNorm has none).
                f"{p}.input_layernorm.weight": (h,),
                # Linear layers are stored transposed, [out_features, in_features],
                # because torch.nn.Linear computes x @ W.T -- so a forward pass
                # written with plain matmuls must transpose these.
                f"{p}.self_attn.q_proj.weight": (q_dim, h),
                f"{p}.self_attn.k_proj.weight": (kv_dim, h),
                f"{p}.self_attn.v_proj.weight": (kv_dim, h),
                # Qwen2 keeps biases on Q/K/V but not on the output projection.
                # Most Llama-style models have no attention biases at all; this
                # is the detail that makes Qwen2 its own architecture.
                f"{p}.self_attn.q_proj.bias": (q_dim,),
                f"{p}.self_attn.k_proj.bias": (kv_dim,),
                f"{p}.self_attn.v_proj.bias": (kv_dim,),
                f"{p}.self_attn.o_proj.weight": (h, q_dim),
                f"{p}.post_attention_layernorm.weight": (h,),
                # SwiGLU feed-forward: down(silu(gate(x)) * up(x)).
                f"{p}.mlp.gate_proj.weight": (cfg.intermediate_size, h),
                f"{p}.mlp.up_proj.weight": (cfg.intermediate_size, h),
                f"{p}.mlp.down_proj.weight": (h, cfg.intermediate_size),
            }
        )

    if not cfg.tie_word_embeddings:
        manifest["lm_head.weight"] = (cfg.vocab_size, h)

    return manifest


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="directory holding config.json and *.safetensors")
    parser.add_argument("--all", action="store_true", help="list every tensor, not just layer 0")
    args = parser.parse_args(argv)

    model_dir: Path = args.model_dir
    cfg = ModelConfig.from_model_dir(model_dir)

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        print(f"no .safetensors file in {model_dir}", file=sys.stderr)
        return 1

    section("Container")
    total_file_bytes = 0
    stores: list[SafeTensors] = []
    for shard in shards:
        st = SafeTensors(shard)
        stores.append(st)
        file_bytes = shard.stat().st_size
        total_file_bytes += file_bytes
        print(f"{shard.name}")
        print(f"  file size       {human_bytes(file_bytes)}")
        print(f"  header length   {st.header_bytes:,} B of JSON (+8 B u64 length prefix)")
        print(f"  data buffer     {human_bytes(file_bytes - st.data_start)} starting at byte {st.data_start:,}")
        print(f"  tensors         {len(st)}")
        if st.metadata:
            for k, v in st.metadata.items():
                print(f"  __metadata__    {k} = {v}")
        else:
            print("  __metadata__    (none)")

    found: dict[str, SafeTensors] = {}
    for st in stores:
        for name in st.names:
            found[name] = st

    section("Config")
    print(f"  architecture        {cfg.architecture}")
    print(f"  vocab_size          {cfg.vocab_size:,}")
    print(f"  hidden_size         {cfg.hidden_size}")
    print(f"  layers              {cfg.num_hidden_layers}")
    print(f"  attention heads     {cfg.num_attention_heads} query / {cfg.num_key_value_heads} key-value")
    print(f"  head_dim            {cfg.head_dim}  (derived: hidden_size / query heads)")
    print(f"  GQA group size      {cfg.kv_group_size}  (query heads sharing one KV head)")
    print(f"  intermediate_size   {cfg.intermediate_size}")
    print(f"  rms_norm_eps        {cfg.rms_norm_eps}")
    print(f"  rope_theta          {cfg.rope_theta:,.0f}")
    print(f"  tie_word_embeddings {cfg.tie_word_embeddings}")
    print(f"  stored dtype        {cfg.torch_dtype}")
    print(f"  KV cache cost       {human_bytes(cfg.kv_bytes_per_token)} per token at fp32, all layers")

    section("Shape check against the manifest derived from config.json")
    expected = expected_tensors(cfg)
    missing = [n for n in expected if n not in found]
    unexpected = [n for n in found if n not in expected]
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for name, shape in expected.items():
        st = found.get(name)
        if st is None:
            continue
        actual = st.info(name).shape
        if actual != shape:
            mismatched.append((name, shape, actual))

    print(f"  expected {len(expected)} tensors, file has {len(found)}")
    print(f"  missing      {len(missing)}")
    print(f"  unexpected   {len(unexpected)}")
    print(f"  wrong shape  {len(mismatched)}")
    for name in missing[:10]:
        print(f"    MISSING    {name} {expected[name]}")
    for name in unexpected[:10]:
        print(f"    UNEXPECTED {name} {found[name].info(name).shape}")
    for name, want, got in mismatched[:10]:
        print(f"    SHAPE      {name} expected {want} got {got}")

    if cfg.tie_word_embeddings:
        print()
        print("  Note: lm_head.weight is absent by design. tie_word_embeddings=true")
        print("  means the output projection reuses model.embed_tokens.weight, so the")
        print("  final logits are hidden @ embed_tokens.T. Looking for a separate")
        print("  lm_head tensor and not finding it is the expected outcome, not a bug.")

    section("Tensors" if args.all else "Tensors (layer 0 only; pass --all for every layer)")
    print(f"  {'name':<48} {'dtype':<6} {'shape':<18} {'params':>12}  {'bytes':>12}")
    shown = 0
    for name in found:
        if not args.all and name.startswith("model.layers.") and not name.startswith("model.layers.0."):
            continue
        info = found[name].info(name)
        shape_s = "x".join(str(d) for d in info.shape)
        print(f"  {name:<48} {info.dtype:<6} {shape_s:<18} {info.numel:>12,}  {info.nbytes:>12,}")
        shown += 1
    print(f"  ({shown} tensors shown of {len(found)})")

    section("Parameter budget")
    buckets = {
        "embeddings": 0,
        "attention": 0,
        "mlp": 0,
        "norms": 0,
    }
    for name, st in found.items():
        n = st.info(name).numel
        if "embed_tokens" in name or name.startswith("lm_head"):
            buckets["embeddings"] += n
        elif ".self_attn." in name:
            buckets["attention"] += n
        elif ".mlp." in name:
            buckets["mlp"] += n
        else:
            buckets["norms"] += n

    total = sum(buckets.values())
    for label, n in buckets.items():
        print(f"  {label:<12} {n:>14,}  {100 * n / total:5.1f}%")
    print(f"  {'TOTAL':<12} {total:>14,}  100.0%")
    print()
    print(f"  stored           {human_bytes(sum(st.total_bytes for st in stores))} as {cfg.torch_dtype}")
    print(f"  as float32       {human_bytes(total * 4)}  (what phase 3 will hold in RAM)")
    print(f"  as int8          {human_bytes(total)}      (phase 6 target)")
    print(f"  as int4          {human_bytes(total // 2)}      (phase 6 target)")

    section("bfloat16 spot check")
    probe = "model.layers.0.self_attn.q_proj.bias"
    st = found[probe]
    raw = st.raw(probe)
    vals = st.f32(probe)
    print(f"  {probe}")
    print(f"  raw container  {raw.dtype} (bf16 has no NumPy dtype; we hold the bits)")
    print("  first 4 values, bit pattern -> float32:")
    for i in range(4):
        print(f"    0x{int(raw[i]):04X}  ->  {float(vals[i]): .8f}")
    print(f"  finite: {bool(np.isfinite(vals).all())}, "
          f"min {vals.min(): .4f}, max {vals.max(): .4f}")

    for st in stores:
        st.close()

    ok = not (missing or unexpected or mismatched)
    print()
    print("RESULT: weight file fully accounted for" if ok else "RESULT: discrepancies found (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
