"""A miniature Qwen2 model, built on the fly for tests.

Most of what needs testing about weight loading and the forward pass does not
need 494 million parameters. This builds a structurally identical model with
about 4,000 of them: same architecture, same tensor names, same grouped-query
ratio, same tied embeddings, bfloat16 on disk like the real thing.

That buys three things. Error paths (missing tensor, wrong shape, contradictory
config) can be tested by writing a deliberately broken model rather than
corrupting the real one. Tests run without the 1 GB download, so CI works. And
shape bugs surface as small readable arrays instead of a wall of numbers.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from nanoinfer.safetensors import f32_to_bf16

_DTYPE_NAMES = {
    np.dtype("<f4"): "F32",
    np.dtype("<f8"): "F64",
    np.dtype("<i4"): "I32",
    np.dtype("<i8"): "I64",
    np.dtype("<u2"): "BF16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}


def write_safetensors(
    path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Serialize tensors into a safetensors file, by hand.

    Used to test the reader against files whose exact bytes are known, and to
    build the tiny model below. Deliberately independent of the reader so a
    single shared bug cannot make a round-trip test pass.
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata

    blob = bytearray()
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        begin = len(blob)
        blob.extend(array.tobytes())
        header[name] = {
            "dtype": _DTYPE_NAMES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [begin, len(blob)],
        }

    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(blob))
    return path


TINY_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_act": "silu",
    "hidden_size": 16,
    "initializer_range": 0.02,
    "intermediate_size": 32,
    "max_position_embeddings": 128,
    "model_type": "qwen2",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    # 2 KV heads for 4 query heads: a group size of 2, so grouped-query
    # attention is genuinely exercised rather than degenerating to plain MHA.
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 64,
}


def tiny_tensors(config: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Random bfloat16 weights matching the config's declared shapes."""
    rng = np.random.default_rng(seed)
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = hidden // heads
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    inter = config["intermediate_size"]

    def weights(*shape: int) -> np.ndarray:
        # Small values keep activations in a sane range across layers, so a
        # test failure means a bug rather than an overflow.
        return f32_to_bf16((rng.standard_normal(shape) * 0.05).astype(np.float32))

    def gains(n: int) -> np.ndarray:
        return f32_to_bf16(np.ones(n, dtype=np.float32))

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weights(config["vocab_size"], hidden),
        "model.norm.weight": gains(hidden),
    }

    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}"
        tensors.update(
            {
                f"{p}.input_layernorm.weight": gains(hidden),
                f"{p}.self_attn.q_proj.weight": weights(q_dim, hidden),
                f"{p}.self_attn.q_proj.bias": weights(q_dim),
                f"{p}.self_attn.k_proj.weight": weights(kv_dim, hidden),
                f"{p}.self_attn.k_proj.bias": weights(kv_dim),
                f"{p}.self_attn.v_proj.weight": weights(kv_dim, hidden),
                f"{p}.self_attn.v_proj.bias": weights(kv_dim),
                f"{p}.self_attn.o_proj.weight": weights(hidden, q_dim),
                f"{p}.post_attention_layernorm.weight": gains(hidden),
                f"{p}.mlp.gate_proj.weight": weights(inter, hidden),
                f"{p}.mlp.up_proj.weight": weights(inter, hidden),
                f"{p}.mlp.down_proj.weight": weights(hidden, inter),
            }
        )

    if not config.get("tie_word_embeddings", False):
        tensors["lm_head.weight"] = weights(config["vocab_size"], hidden)

    return tensors


def build_tiny_model(
    directory: Path,
    config_overrides: dict | None = None,
    tensor_overrides: dict[str, np.ndarray | None] | None = None,
    seed: int = 0,
) -> Path:
    """Write a tiny model to ``directory`` and return the path.

    ``tensor_overrides`` replaces or, with a value of ``None``, deletes a
    tensor -- which is how the "missing tensor" and "wrong shape" error paths
    get tested without touching a real model.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    config = dict(TINY_CONFIG)
    config.update(config_overrides or {})

    tensors = tiny_tensors(config, seed=seed)
    for name, value in (tensor_overrides or {}).items():
        if value is None:
            tensors.pop(name, None)
        else:
            tensors[name] = value

    (directory / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    write_safetensors(directory / "model.safetensors", tensors, {"format": "pt"})
    return directory
