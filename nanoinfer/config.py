"""Model hyperparameters, read from the HuggingFace ``config.json``.

Reading the config is not inference, so there is no purity question here: the
file is plain JSON describing shapes and constants. What matters is that we
derive the values the forward pass will need (head dimension, the GQA grouping)
once, in one place, and assert the invariants now rather than discovering them
as a shape mismatch twenty layers deep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_ARCHITECTURES = frozenset({"Qwen2ForCausalLM"})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Everything the forward pass needs to know about the model's shape."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    tie_word_embeddings: bool
    torch_dtype: str
    bos_token_id: int
    eos_token_id: int

    @property
    def head_dim(self) -> int:
        """Width of a single attention head.

        Qwen2 does not store this; it is implied by hidden_size / n_heads. Some
        newer architectures decouple the two, which is why this is a derived
        property with a check rather than a bare division at the call site.
        """
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_group_size(self) -> int:
        """How many query heads share each key/value head.

        1 means plain multi-head attention. Greater than 1 means grouped-query
        attention: the KV cache holds only ``num_key_value_heads`` heads, and
        each is broadcast across this many query heads. For Qwen2.5-0.5B it is
        7, which shrinks the KV cache by 7x -- the single biggest memory win in
        the whole model, and the place where a wrong repeat/reshape silently
        produces fluent but incorrect text.
        """
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache one token costs, at float32, across all layers."""
        per_layer = 2 * self.num_key_value_heads * self.head_dim * 4
        return per_layer * self.num_hidden_layers

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture {self.architecture!r} is not supported yet; "
                f"this engine implements {sorted(SUPPORTED_ARCHITECTURES)}"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not divisible "
                f"by num_key_value_heads {self.num_key_value_heads}"
            )
        if self.hidden_act != "silu":
            raise ValueError(
                f"hidden_act {self.hidden_act!r} is not implemented; expected 'silu'"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        architectures = data.get("architectures") or []
        if len(architectures) != 1:
            raise ValueError(f"expected exactly one architecture, got {architectures}")

        eos = data["eos_token_id"]
        if isinstance(eos, list):  # some configs list several stop tokens
            eos = eos[0]

        return cls(
            architecture=architectures[0],
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            # Absent means no GQA, i.e. one KV head per query head.
            num_key_value_heads=int(
                data.get("num_key_value_heads", data["num_attention_heads"])
            ),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            hidden_act=str(data.get("hidden_act", "silu")),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            torch_dtype=str(data.get("torch_dtype", "float32")),
            bos_token_id=int(data["bos_token_id"]),
            eos_token_id=int(eos),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "ModelConfig":
        return cls.from_json(Path(model_dir) / "config.json")
