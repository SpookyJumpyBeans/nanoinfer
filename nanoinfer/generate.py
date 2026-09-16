"""Greedy decoding: the loop that turns a forward pass into text.

Autoregressive generation is the forward pass in a loop. Run the model on the
prompt, take the logits for the last position, pick a token, append it, run
again. Phase 5 replaces "pick the argmax" with temperature, top-k and top-p;
phase 4 replaces "run again on everything" with a KV cache.

Right now it really does run again on everything. Each step re-embeds and
re-attends over the entire sequence, so producing n tokens from a prompt of
length p costs roughly ``sum(p+i for i in range(n))`` full forward passes'
worth of attention. The cost of that is measured rather than hidden: the
result object reports prefill and decode time separately, which is exactly the
split phase 4 will improve.

The two timings mean different things and should never be averaged together:

* **Prefill** processes the whole prompt in one pass. It is compute-bound and
  parallel across positions.
* **Decode** produces one token at a time. Even with a cache it is bound by
  memory bandwidth, because every weight in the model is read to produce a
  single token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class GenerationResult:
    """What was generated, and what it cost."""

    prompt_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    stop_reason: str = "max_tokens"

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

    @property
    def prefill_tokens_per_second(self) -> float:
        return len(self.prompt_ids) / self.prefill_s if self.prefill_s else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        return len(self.generated_ids) / self.decode_s if self.decode_s else 0.0

    @property
    def time_to_first_token_ms(self) -> float:
        return self.prefill_s * 1000


def greedy_stream(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    stop_ids: Sequence[int] = (),
) -> Iterator[int]:
    """Yield generated token IDs one at a time, picking the argmax each step.

    Streaming rather than returning a list, because a token-a-second decode
    loop is unbearable to wait on in silence and because it lets a caller stop
    early without the loop knowing why.

    The stop token is yielded before the loop ends. Dropping it silently would
    make the caller unable to tell "the model chose to stop" from "we hit the
    token limit", and those are different outcomes.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    stop = set(stop_ids)
    tokens = list(prompt_ids)

    for _ in range(max_new_tokens):
        logits = model.next_token_logits(np.array(tokens))
        next_id = int(np.argmax(logits))

        tokens.append(next_id)
        yield next_id

        if next_id in stop:
            return


def greedy(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    stop_ids: Sequence[int] = (),
    on_token=None,
) -> GenerationResult:
    """Generate greedily, timing prefill and decode separately.

    ``on_token`` is called with each new ID as it arrives, for callers that
    want to print as they go.

    Prefill is measured by running the prompt through once before the loop
    starts. Without a cache that pass is redundant -- the first decode step
    recomputes it -- but measuring it is the only way to report time-to-first-
    token, and phase 4 turns that redundancy into the cache fill.
    """
    result = GenerationResult(prompt_ids=list(prompt_ids))
    if not prompt_ids:
        raise ValueError("cannot generate from an empty prompt")

    start = time.perf_counter()
    model.next_token_logits(np.array(prompt_ids))
    result.prefill_s = time.perf_counter() - start

    stop = set(stop_ids)
    start = time.perf_counter()
    for token_id in greedy_stream(model, prompt_ids, max_new_tokens, stop_ids):
        result.generated_ids.append(token_id)
        if on_token is not None:
            on_token(token_id)
        if token_id in stop:
            result.stop_reason = "stop_token"
    result.decode_s = time.perf_counter() - start

    return result
