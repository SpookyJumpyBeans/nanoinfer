"""The decode loop that turns a forward pass into text.

Autoregressive generation is the forward pass in a loop. Run the model, take
the logits for the last position, pick a token, append it, run again.

How the token is picked is the only thing that varies: argmax by default, or a
:class:`nanoinfer.sampling.Sampler` for temperature, top-k and top-p. There is
one loop rather than a greedy one and a sampled one, because everything except
that single choice is identical and two copies would drift.

Two strategies live here, and the uncached one is kept deliberately:

* **No cache.** Every step re-embeds and re-attends over the entire sequence,
  so producing n tokens from a prompt of length p costs roughly
  ``sum(p + i for i in range(n))`` forward passes' worth of attention. This is
  not a fallback; it is the reference the cached path is checked against.
* **With a cache.** The prompt is processed once, and each step afterwards
  feeds a single token against the stored keys and values.

The final generated token is deliberately not fed back through the model.
Nothing would read the resulting logits, and skipping it saves an entire
forward pass from every run. The visible consequence is that when generation
ends the cache holds one fewer token than was produced, which matters only to a
caller intending to continue from it.

The two timings mean different things and must never be averaged together:

* **Time to first token** covers the whole prompt. It is compute-bound and
  parallel across positions, and it is what a user experiences as latency.
* **Decode** is the steady-state rate afterwards. Even with a cache it is bound
  by memory bandwidth, because every weight in the model is read to produce a
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
    used_cache: bool = False
    sampled: bool = False
    seed: int | None = None

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

    @property
    def time_to_first_token_ms(self) -> float:
        """Latency from the call to the first token, prompt processing included."""
        return self.prefill_s * 1000

    @property
    def decode_tokens_per_second(self) -> float:
        """Steady-state rate, excluding the first token.

        The first token is excluded because its cost is dominated by the
        prompt. Including it would blend two different regimes into one number
        and would make the rate depend on prompt length, which a steady-state
        throughput figure should not.
        """
        after_first = len(self.generated_ids) - 1
        if after_first < 1 or self.decode_s <= 0:
            return 0.0
        return after_first / self.decode_s

    @property
    def prompt_tokens_per_second(self) -> float:
        return len(self.prompt_ids) / self.prefill_s if self.prefill_s else 0.0


def generate_stream(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    stop_ids: Sequence[int] = (),
    cache=None,
    sampler=None,
) -> Iterator[int]:
    """Yield generated token IDs one at a time.

    ``sampler`` chooses each token from the logits. ``None`` means take the
    argmax, which is what phases 3 and 4 did and what every existing test
    expects; pass a :class:`nanoinfer.sampling.Sampler` to sample instead.
    The loop itself is identical either way -- only the choice of token
    differs, which is why there is one loop here and not two.

    Streaming rather than returning a list, because a slow decode loop is
    unbearable to wait on in silence and because it lets a caller stop early
    without the loop needing to know why.

    The stop token is yielded before the loop ends. Swallowing it would leave
    the caller unable to tell "the model chose to stop" from "we hit the token
    limit", and those are different outcomes.

    ``cache`` selects the strategy. It must be empty on entry; a cache carrying
    another sequence's keys would silently condition this generation on that
    one.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if not prompt_ids:
        raise ValueError("cannot generate from an empty prompt")
    if max_new_tokens == 0:
        return

    stop = set(stop_ids)
    pick = sampler if sampler is not None else (lambda logits: int(np.argmax(logits)))

    if cache is None:
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = int(pick(model.next_token_logits(np.array(tokens))))
            tokens.append(next_id)
            yield next_id
            if next_id in stop:
                return
        return

    if cache.length:
        raise ValueError(
            f"cache already holds {cache.length} tokens; reset it before "
            "starting a new sequence"
        )

    # Prefill: the whole prompt in one pass, which is also what fills the cache.
    logits = model.next_token_logits(np.array(prompt_ids), cache=cache)

    for step in range(max_new_tokens):
        next_id = int(pick(logits))
        yield next_id

        if next_id in stop:
            return
        if step + 1 == max_new_tokens:
            return

        logits = model.next_token_logits(np.array([next_id]), cache=cache)


def greedy_stream(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    stop_ids: Sequence[int] = (),
    cache=None,
) -> Iterator[int]:
    """Always take the argmax. Kept as the name the correctness tests use.

    Greedy decoding is not one sampling strategy among several here -- it is
    the reference the sampled paths are checked against, because it is the only
    one whose output can be compared token for token with another engine.
    """
    yield from generate_stream(model, prompt_ids, max_new_tokens, stop_ids, cache)


def greedy(
    model,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    stop_ids: Sequence[int] = (),
    on_token=None,
    use_cache: bool = True,
    sampler=None,
) -> GenerationResult:
    """Generate, timing the first token separately from the rest.

    ``on_token`` is called with each ID as it arrives, for callers that print
    as they go. ``use_cache=False`` selects the uncached path, which exists so
    the two can be compared. ``sampler`` defaults to None, meaning argmax; pass
    a Sampler to sample instead.

    Timing is taken from the stream rather than by running a separate prefill
    pass. Prefilling twice to get a clean measurement would both distort the
    measurement and double the cost of the thing being measured.
    """
    if not prompt_ids:
        raise ValueError("cannot generate from an empty prompt")

    result = GenerationResult(prompt_ids=list(prompt_ids))
    cache = model.new_cache(len(prompt_ids) + max_new_tokens) if use_cache else None
    result.used_cache = cache is not None
    result.sampled = sampler is not None
    result.seed = getattr(getattr(sampler, "config", None), "seed", None)
    stop = set(stop_ids)

    started = time.perf_counter()
    first_token_at: float | None = None

    for token_id in generate_stream(
        model, prompt_ids, max_new_tokens, stop_ids, cache, sampler
    ):
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
            result.prefill_s = now - started

        result.generated_ids.append(token_id)
        if on_token is not None:
            on_token(token_id)
        if token_id in stop:
            result.stop_reason = "stop_token"

    if first_token_at is not None:
        result.decode_s = time.perf_counter() - first_token_at

    return result
