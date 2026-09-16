"""The Qwen2 forward pass: token IDs in, logits out.

Twenty-four blocks, each doing the same two things with a residual connection
around both:

    x = x + attention(rms_norm(x))
    x = x + feed_forward(rms_norm(x))

Two details of that shape are worth naming, because they are architectural
choices rather than arbitrary ones.

**The norm is inside the residual branch, not outside it.** ``x + f(norm(x))``,
not ``norm(x + f(x))``. This is "pre-norm", and it leaves a path from the
embedding to the final norm that no normalization ever touches. That clean
residual highway is what makes deep transformers trainable, and at inference it
means the residual stream accumulates rather than being rescaled at every step.

**There is no cache here at all.** Every call recomputes attention over the
whole sequence from scratch, which makes generating n tokens O(n³) work in
total. That is the point: phase 3 is the correctness baseline that phase 4's
KV cache has to reproduce exactly while being roughly 10x faster. Optimizing
now would mean optimizing something not yet known to be right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nanoinfer.attention import self_attention
from nanoinfer.config import ModelConfig
from nanoinfer.ops import rms_norm, silu
from nanoinfer.rope import RotaryEmbedding
from nanoinfer.weights import LayerWeights, ModelWeights


def feed_forward(hidden: np.ndarray, layer: LayerWeights) -> np.ndarray:
    """SwiGLU: ``down(silu(gate(x)) * up(x))``.

    Two parallel projections up to the intermediate width, one passed through
    SiLU and the other not, multiplied elementwise, then projected back down.
    The un-activated ``up`` branch is the "gate" in gated linear unit -- it
    lets the block modulate its own output multiplicatively, which a plain
    ``down(silu(up(x)))`` cannot.

    This is where two thirds of the model's parameters live.
    """
    gate = hidden @ layer.gate_proj_weight.T
    up = hidden @ layer.up_proj_weight.T
    return (silu(gate) * up) @ layer.down_proj_weight.T


def transformer_block(
    hidden: np.ndarray,
    layer: LayerWeights,
    config: ModelConfig,
    rope: RotaryEmbedding,
    positions: np.ndarray,
) -> np.ndarray:
    """One decoder layer: pre-norm attention, then pre-norm feed-forward."""
    normed = rms_norm(hidden, layer.input_layernorm, config.rms_norm_eps)
    hidden = hidden + self_attention(normed, layer, config, rope, positions)

    normed = rms_norm(hidden, layer.post_attention_layernorm, config.rms_norm_eps)
    hidden = hidden + feed_forward(normed, layer)

    return hidden


class Qwen2:
    """A Qwen2 causal language model, evaluated with NumPy.

    Stateless between calls: :meth:`forward` takes a complete sequence and
    recomputes everything. The rotary tables are the only thing carried
    between calls, and they depend on position alone, not on any input.
    """

    def __init__(self, weights: ModelWeights) -> None:
        self.weights = weights
        self.config = weights.config
        self.rope = RotaryEmbedding(self.config.head_dim, self.config.rope_theta)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen2":
        return cls(ModelWeights.load(model_dir))

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up the embedding vector for each token ID.

        A plain row-gather. It is worth noticing that this is the single
        largest tensor in the model being used as a lookup table, and that the
        same matrix is used again at the end to produce logits.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError(f"expected a 1-D sequence of token IDs, got {token_ids.shape}")
        if token_ids.size == 0:
            raise ValueError("cannot run the model on an empty sequence")

        out_of_range = (token_ids < 0) | (token_ids >= self.config.vocab_size)
        if out_of_range.any():
            bad = token_ids[out_of_range][0]
            raise ValueError(
                f"token id {bad} is outside the embedding matrix "
                f"(0..{self.config.vocab_size - 1})"
            )

        return self.weights.embed_tokens[token_ids]

    def hidden_states(
        self, token_ids: np.ndarray, positions: np.ndarray | None = None
    ) -> np.ndarray:
        """Run the blocks and the final norm, returning ``[seq, hidden]``."""
        hidden = self.embed(token_ids)
        if positions is None:
            positions = np.arange(hidden.shape[0])

        for layer in self.weights.layers:
            hidden = transformer_block(hidden, layer, self.config, self.rope, positions)

        return rms_norm(hidden, self.weights.final_norm, self.config.rms_norm_eps)

    def forward(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray | None = None,
        last_only: bool = False,
    ) -> np.ndarray:
        """Logits for the sequence, shaped ``[seq, vocab]``.

        With ``last_only`` the projection is applied to the final position
        alone and the result is ``[1, vocab]``. That is not a micro-
        optimization: the output projection is a
        ``[seq, 896] @ [896, 151936]`` matmul, which for a long prompt is the
        single most expensive operation in the whole pass, and generation only
        ever needs its last row.
        """
        hidden = self.hidden_states(token_ids, positions)
        if last_only:
            hidden = hidden[-1:]

        # Tied embeddings: this is embed_tokens again, used transposed.
        return hidden @ self.weights.lm_head.T

    def next_token_logits(self, token_ids: np.ndarray) -> np.ndarray:
        """Logits for the token that would come next, shaped ``[vocab]``."""
        return self.forward(token_ids, last_only=True)[0]
