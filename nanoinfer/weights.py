"""Loading the weight file into the structure the forward pass wants.

Phase 1 proved every tensor in the file is accounted for. This turns that flat
name-to-tensor mapping into typed objects, checking each shape against the
config as it goes, so the forward pass can index a dataclass field instead of
formatting a string and hoping the tensor exists.

Two decisions worth stating:

**Linear weights stay as stored, in ``[out_features, in_features]``.** That is
PyTorch's layout, because ``nn.Linear`` computes ``x @ W.T``. It is tempting to
transpose once at load time so the forward pass can write a plain ``x @ W``,
but there is nothing to gain: NumPy hands a transposed 2-D view straight to
BLAS with the transpose flag set, so ``x @ W.T`` is already a single ``sgemm``
with no copy. Transposing would cost a gigabyte of churn to save nothing.

**Everything is widened to float32 at load.** The file is bfloat16, which NumPy
cannot compute with, so the conversion has to happen somewhere. Doing it once
up front costs 1.9 GB of RAM and is the correctness-first choice; phase 6 is
where that becomes int8 and int4 and the memory comes back down.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from nanoinfer.config import ModelConfig
from nanoinfer.safetensors import SafeTensors


class WeightShapeError(ValueError):
    """A tensor in the file does not have the shape the config implies."""


@dataclass(frozen=True, slots=True)
class LayerWeights:
    """Everything one transformer block needs.

    Field names follow the file's own naming rather than being prettified, so
    a tensor can be traced from ``config.json`` to here to the forward pass
    without a translation step.
    """

    input_layernorm: np.ndarray          # [hidden]
    q_proj_weight: np.ndarray            # [n_heads * head_dim, hidden]
    q_proj_bias: np.ndarray              # [n_heads * head_dim]
    k_proj_weight: np.ndarray            # [n_kv_heads * head_dim, hidden]
    k_proj_bias: np.ndarray              # [n_kv_heads * head_dim]
    v_proj_weight: np.ndarray            # [n_kv_heads * head_dim, hidden]
    v_proj_bias: np.ndarray              # [n_kv_heads * head_dim]
    o_proj_weight: np.ndarray            # [hidden, n_heads * head_dim]
    post_attention_layernorm: np.ndarray  # [hidden]
    gate_proj_weight: np.ndarray         # [intermediate, hidden]
    up_proj_weight: np.ndarray           # [intermediate, hidden]
    down_proj_weight: np.ndarray         # [hidden, intermediate]

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f.name).nbytes for f in fields(self))


@dataclass(frozen=True, slots=True)
class ModelWeights:
    """The whole model: embeddings, blocks, final norm."""

    config: ModelConfig
    embed_tokens: np.ndarray             # [vocab, hidden]
    layers: tuple[LayerWeights, ...]
    final_norm: np.ndarray               # [hidden]
    _lm_head: np.ndarray | None          # None when tied to embed_tokens

    @property
    def lm_head(self) -> np.ndarray:
        """The output projection, shaped ``[vocab, hidden]`` either way.

        When ``tie_word_embeddings`` is set -- as it is for Qwen2.5 -- there is
        no ``lm_head.weight`` in the file and the embedding matrix does double
        duty: it maps IDs to vectors on the way in, and vectors to logits on
        the way out. Returning it here rather than special-casing at the call
        site keeps the forward pass from having to know.
        """
        return self.embed_tokens if self._lm_head is None else self._lm_head

    @property
    def tied(self) -> bool:
        return self._lm_head is None

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.nbytes + self.final_norm.nbytes
        total += sum(layer.nbytes for layer in self.layers)
        if self._lm_head is not None:
            total += self._lm_head.nbytes
        return total

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelWeights":
        model_dir = Path(model_dir)
        config = ModelConfig.from_model_dir(model_dir)

        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors file in {model_dir}")

        stores = [SafeTensors(path) for path in shards]
        try:
            index: dict[str, SafeTensors] = {}
            for store in stores:
                for name in store.names:
                    index[name] = store

            def take(name: str, expected: tuple[int, ...]) -> np.ndarray:
                store = index.get(name)
                if store is None:
                    raise WeightShapeError(f"missing tensor {name!r}")
                actual = store.info(name).shape
                if actual != expected:
                    raise WeightShapeError(
                        f"{name}: expected shape {expected}, file has {actual}"
                    )
                return store.f32(name)

            hidden = config.hidden_size
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            inter = config.intermediate_size

            layers = []
            for i in range(config.num_hidden_layers):
                p = f"model.layers.{i}"
                layers.append(
                    LayerWeights(
                        input_layernorm=take(f"{p}.input_layernorm.weight", (hidden,)),
                        q_proj_weight=take(f"{p}.self_attn.q_proj.weight", (q_dim, hidden)),
                        q_proj_bias=take(f"{p}.self_attn.q_proj.bias", (q_dim,)),
                        k_proj_weight=take(f"{p}.self_attn.k_proj.weight", (kv_dim, hidden)),
                        k_proj_bias=take(f"{p}.self_attn.k_proj.bias", (kv_dim,)),
                        v_proj_weight=take(f"{p}.self_attn.v_proj.weight", (kv_dim, hidden)),
                        v_proj_bias=take(f"{p}.self_attn.v_proj.bias", (kv_dim,)),
                        o_proj_weight=take(f"{p}.self_attn.o_proj.weight", (hidden, q_dim)),
                        post_attention_layernorm=take(
                            f"{p}.post_attention_layernorm.weight", (hidden,)
                        ),
                        gate_proj_weight=take(f"{p}.mlp.gate_proj.weight", (inter, hidden)),
                        up_proj_weight=take(f"{p}.mlp.up_proj.weight", (inter, hidden)),
                        down_proj_weight=take(f"{p}.mlp.down_proj.weight", (hidden, inter)),
                    )
                )

            embed = take("model.embed_tokens.weight", (config.vocab_size, hidden))
            final_norm = take("model.norm.weight", (hidden,))

            lm_head: np.ndarray | None = None
            if config.tie_word_embeddings:
                if "lm_head.weight" in index:
                    raise WeightShapeError(
                        "config says tie_word_embeddings but the file also "
                        "contains lm_head.weight; refusing to guess which wins"
                    )
            else:
                lm_head = take("lm_head.weight", (config.vocab_size, hidden))

            return cls(
                config=config,
                embed_tokens=embed,
                layers=tuple(layers),
                final_norm=final_norm,
                _lm_head=lm_head,
            )
        finally:
            for store in stores:
                store.close()
