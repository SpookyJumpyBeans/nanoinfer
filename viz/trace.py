"""Generation with the internals kept: candidates, logits and attention.

The engine computes all of this already and throws it away. Attention weights
are a local inside ``self_attention``; the logits for each step are consumed by
the sampler and dropped. This module keeps them, without changing a line of the
engine.

**Attention is captured by wrapping the module's softmax, not by editing the
function.** ``nanoinfer.attention`` calls ``softmax`` exactly once per layer,
on the masked scores, so replacing the name it looks up records every layer's
attention weights in order. The engine keeps a single code path, and the tests
here check that the traced run produces the same tokens as ``greedy()``, so the
visualizer cannot drift into showing a different model from the one tested.

**The rows are assembled from the cached run itself.** Prefill yields
``[heads, prompt, prompt]`` for each layer, and every decode step yields one
more row ``[heads, 1, seen]``. Stacking those gives the full lower-triangular
attention matrix without a second forward pass. The generation loop feeds the
final token back through the model once more, which ``generate_stream``
deliberately skips, so that the last token has a row too.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

import nanoinfer.attention as attention_module
from nanoinfer.sampling import Sampler, SamplingConfig


@contextlib.contextmanager
def capture_attention() -> Iterator[list[np.ndarray]]:
    """Record every attention softmax computed inside the block.

    Yields a list that fills with one ``[heads, n_new, seen]`` array per layer
    per forward pass, in call order.
    """
    original = attention_module.softmax
    captured: list[np.ndarray] = []

    def recording_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        out = original(x, axis=axis)
        captured.append(out.copy())
        return out

    attention_module.softmax = recording_softmax
    try:
        yield captured
    finally:
        attention_module.softmax = original


def top_candidates(
    logits: np.ndarray, sampled_probs: np.ndarray | None, k: int
) -> list[dict]:
    """The ``k`` most likely tokens under the model, with both probabilities.

    ``p_model`` is the model's own distribution (plain softmax of the logits).
    ``p_sampled`` is what the sampler actually drew from after temperature,
    top-k and top-p, which is zero for any token the filters removed. For
    greedy decoding the two are shown as the same distribution.
    """
    shifted = logits - np.max(logits)
    p_model = np.exp(shifted)
    p_model /= p_model.sum()
    order = np.argsort(-p_model)[:k]
    sampled = sampled_probs if sampled_probs is not None else p_model
    return [
        {"id": int(i), "p_model": float(p_model[i]), "p_sampled": float(sampled[i])}
        for i in order
    ]


@dataclass
class Step:
    """One generated token and what the model considered."""

    token_id: int
    candidates: list[dict]
    ms: float


@dataclass
class Trace:
    """A traced generation: tokens, per-step logits and attention."""

    prompt_ids: list[int]
    steps: list[Step] = field(default_factory=list)
    logits: list[np.ndarray] = field(default_factory=list)
    # attention[layer] is [heads, seq, seq], lower triangular.
    attention: list[np.ndarray] = field(default_factory=list)
    prefill_ms: float = 0.0
    stop_reason: str = "max_tokens"

    @property
    def generated_ids(self) -> list[int]:
        return [s.token_id for s in self.steps]

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids


def _assemble(rows_per_layer: list[list[np.ndarray]], seq: int) -> list[np.ndarray]:
    """Stack per-pass attention rows into one ``[heads, seq, seq]`` per layer."""
    layers = []
    for rows in rows_per_layer:
        heads = rows[0].shape[0]
        full = np.zeros((heads, seq, seq), dtype=np.float32)
        r = 0
        for block in rows:
            n_new, seen = block.shape[1], block.shape[2]
            full[:, r : r + n_new, :seen] = block
            r += n_new
        layers.append(full)
    return layers


def traced_generate(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    stop_ids: Sequence[int] = (),
    config: SamplingConfig | None = None,
    top_k: int = 8,
    on_step=None,
) -> Trace:
    """Generate with the KV cache, keeping candidates, logits and attention.

    ``on_step(trace, step)`` is called as each token is chosen, so a caller can
    stream. Attention is assembled once generation finishes.
    """
    if not prompt_ids:
        raise ValueError("cannot generate from an empty prompt")
    config = config or SamplingConfig.greedy()
    sampler = None if config.is_greedy else Sampler(config)
    stop = set(stop_ids)
    n_layers = model.config.num_hidden_layers

    trace = Trace(prompt_ids=list(prompt_ids))
    cache = model.new_cache(len(prompt_ids) + max_new_tokens + 1)
    rows: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

    def run(ids: list[int]) -> np.ndarray:
        with capture_attention() as captured:
            logits = model.next_token_logits(np.array(ids), cache=cache)
        if len(captured) != n_layers:
            raise RuntimeError(
                f"expected {n_layers} attention captures, got {len(captured)}"
            )
        for layer, block in enumerate(captured):
            rows[layer].append(block)
        return logits

    started = time.perf_counter()
    logits = run(list(prompt_ids))
    trace.prefill_ms = (time.perf_counter() - started) * 1000
    # Each step is charged the forward pass that produced its logits: the
    # prefill for the first token, one decode pass for every token after.
    forward_ms = trace.prefill_ms

    for index in range(max_new_tokens):
        if sampler is None:
            chosen = int(np.argmax(logits))
            sampled_probs = None
        else:
            sampled_probs = sampler.probabilities(logits)
            chosen = sampler(logits)

        step = Step(
            token_id=chosen,
            candidates=top_candidates(logits, sampled_probs, top_k),
            ms=forward_ms,
        )
        trace.steps.append(step)
        trace.logits.append(logits.astype(np.float32, copy=True))
        if on_step is not None:
            on_step(trace, step)

        if index + 1 == max_new_tokens or chosen in stop:
            # Unlike generate_stream, the final token is fed back too, so it
            # gets an attention row of its own. Its logits are never used.
            run([chosen])
            if chosen in stop:
                trace.stop_reason = "stop_token"
            break

        forward_started = time.perf_counter()
        logits = run([chosen])
        forward_ms = (time.perf_counter() - forward_started) * 1000

    trace.attention = _assemble(rows, len(trace.all_ids))
    return trace


def resample(
    logits: np.ndarray, config: SamplingConfig, k: int = 10
) -> list[dict]:
    """What a different sampling setting would make of the same step's logits."""
    if config.is_greedy:
        return top_candidates(logits, None, k)
    return top_candidates(logits, Sampler(config).probabilities(logits), k)
