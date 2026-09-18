"""Turning logits into a token: temperature, top-k, top-p, and a seeded draw.

Phase 3 and 4 always took the argmax. That is the right default for checking
correctness -- it is deterministic, so output can be compared token for token
against a reference -- but it makes the model repetitive, because the single
most likely continuation is very often a word the model has just used.

Sampling trades that determinism for variety, and the three knobs here shape
the distribution it samples from in different ways:

* **Temperature** rescales the logits before the softmax. Below 1 it sharpens
  the distribution toward the argmax; above 1 it flattens it. It changes the
  *shape* of the distribution but never its support -- every token keeps a
  non-zero probability.
* **Top-k** truncates to the k most likely tokens. A fixed-size cut, which is
  blunt: when the model is confident, k=50 admits 49 tokens it has already
  dismissed.
* **Top-p** (nucleus) truncates to the smallest set of tokens whose
  probabilities sum to p. The size of that set adapts to how confident the
  model is, which is why it is usually preferred over top-k.

Order matters and is not arbitrary: temperature, then top-k, then top-p, then
softmax, then draw. Temperature has to come first because it changes the
probabilities the other two threshold against. Applying top-p before
temperature would nucleus-sample a distribution the model never produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The value masked-out logits are set to. -inf is exact: exp(-inf) is zero, so
# a filtered token has probability zero rather than merely a small one.
FILTER_VALUE = -np.inf


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide the logits by ``temperature``.

    ``temperature`` of 1.0 is the identity. Smaller values sharpen the
    distribution; larger values flatten it.

    Zero is a special case rather than a division by zero. The limit as
    temperature approaches zero puts all mass on the argmax, so
    ``temperature=0`` means greedy decoding -- which is how every other
    inference engine spells it, and what a user typing 0 expects. It is
    handled by the caller (see :class:`Sampler`) rather than here, because
    "return the argmax" is a sampling decision, not a rescaling.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if temperature == 0:
        raise ValueError(
            "temperature=0 means greedy decoding; Sampler handles it, "
            "apply_temperature cannot divide by zero"
        )

    return (logits / np.float32(temperature)).astype(np.float32, copy=False)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` highest logits; mask the rest to ``-inf``.

    ``k`` of 0 (or at least the vocabulary size) disables the filter.

    The cut is "strictly below the k-th largest value", not "the first k after
    sorting". Those differ when the k-th and (k+1)-th logits are equal: this
    keeps both, so the surviving set can be *larger* than k. That is what the
    reference does, and it is the defensible choice -- breaking a tie by index
    would make the result depend on vocabulary order, which is arbitrary.
    """
    if k < 0:
        raise ValueError(f"top_k must be non-negative, got {k}")
    if k == 0 or k >= logits.shape[-1]:
        return logits

    # The k-th largest value. argpartition puts it at position -k without
    # sorting the whole vocabulary, which matters when the vocabulary is
    # 151,936 entries and this runs once per generated token.
    kth_value = np.partition(logits, -k)[-k]

    filtered = logits.copy()
    filtered[logits < kth_value] = FILTER_VALUE
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens summing to ``p``.

    ``p`` of 1.0 disables the filter.

    Equivalently: drop the least likely tail whose mass is under ``1 - p``.
    Everything above that cut survives, including the token the boundary falls
    on, which is what makes the kept mass at least ``p`` rather than just under
    it. The most likely token always survives, even when ``p`` is smaller than
    its own probability.

    (Stated as a tail-drop rather than as "accumulate descending until the sum
    reaches p" because that is how it is implemented, and the two are not
    interchangeable in float32 -- see the comment in the body.)

    Unlike top-k, the size of the surviving set is not fixed. When the model is
    confident a single token can carry 90% of the mass and the nucleus is one
    token wide; when it is unsure the same p admits hundreds. That adaptivity
    is the whole point of preferring it to top-k.
    """
    if not 0 < p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits

    # Accumulate ASCENDING and drop the tail whose mass is under 1 - p, rather
    # than accumulating descending and keeping until the total reaches p.
    #
    # The two are algebraically the same cut. They are not the same in float32.
    # For 100 equiprobable tokens with p=0.5 the descending sum reaches
    # 0.4999997913837433 at index 49 -- a hair under the boundary -- so a
    # ">= p" test keeps one extra token where a "<= 1 - p" test on the ascending
    # sum drops it. Every disagreement found between the two formulations was a
    # tie landing on opposite sides like that.
    #
    # Neither is more correct in principle, so this follows the reference: that
    # is what makes top_p=0.8 select the same nucleus here as in any other
    # engine, which is the only thing a user of the parameter can rely on.
    # Stable, not the default quicksort. Ties have to break the same way the
    # reference breaks them or the surviving token differs: for [1, 1, 0, 0]
    # numpy's quicksort orders the tied pair [3, 2, 1, 0] while torch.sort
    # gives [2, 3, 0, 1], so "the most likely token" is a different index and
    # a p small enough to keep only one keeps the wrong one.
    order = np.argsort(logits, kind="stable")  # ascending
    ordered = logits[order]

    shifted = ordered - ordered[-1]           # stable softmax over the sorted run
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)

    discard_sorted = cumulative <= (1.0 - p)
    # Always keep the most likely token, so the nucleus is never empty even
    # when p is smaller than the largest single probability.
    discard_sorted[-1] = False

    discard = np.zeros_like(discard_sorted)
    discard[order] = discard_sorted

    filtered = logits.copy()
    filtered[discard] = FILTER_VALUE
    return filtered


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How to turn logits into a token.

    Defaults are the identity: temperature 1, no truncation. That is sampling
    from the model's own distribution, unmodified.
    """

    temperature: float = 1.0
    top_k: int = 0          # 0 disables
    top_p: float = 1.0      # 1.0 disables
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @classmethod
    def greedy(cls) -> "SamplingConfig":
        return cls(temperature=0.0)

    @classmethod
    def from_model_dir(cls, model_dir, seed: int | None = None) -> "SamplingConfig":
        """Read the sampling defaults the model itself ships.

        Qwen2.5 declares temperature 0.7, top_p 0.8 and top_k 20 in
        ``generation_config.json``. Those are the settings the model was tuned
        to be used with, so defaulting to them makes this engine behave like
        every other one running the same weights.

        ``repetition_penalty`` is deliberately not read. It is a logits
        processor rather than a sampling parameter, phase 5 does not implement
        it, and silently ignoring a declared value would be worse than not
        claiming to support it -- see the phase 3 finding that transformers
        applies it even when do_sample is False.
        """
        import json
        from pathlib import Path

        path = Path(model_dir) / "generation_config.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            top_k=int(data.get("top_k", 0) or 0),
            top_p=float(data.get("top_p", 1.0)),
            seed=seed,
        )


class Sampler:
    """Applies the filters in order and draws one token.

    Order is temperature, then top-k, then top-p, then softmax, then draw.
    Temperature has to come first because it changes the probabilities the
    truncations threshold against; nucleus-sampling a distribution the model
    never produced would make ``top_p`` mean something different at every
    temperature.
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def reset(self) -> None:
        """Restart the RNG from the configured seed.

        Reproducibility is per-sampler, not global: two runs match when they
        start from the same seed and draw the same number of times. Reusing one
        sampler across two generations without resetting gives the second a
        different stream, which is correct but is not a repeat of the first.
        """
        self._rng = np.random.default_rng(self.config.seed)

    def filter(self, logits: np.ndarray) -> np.ndarray:
        """The masked logits this sampler would draw from. Exposed for tests."""
        if self.config.is_greedy:
            return logits

        filtered = apply_temperature(logits, self.config.temperature)
        filtered = top_k_filter(filtered, self.config.top_k)
        return top_p_filter(filtered, self.config.top_p)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        """The distribution actually sampled from, after filtering."""
        filtered = self.filter(logits)
        shifted = filtered - np.max(filtered)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def __call__(self, logits: np.ndarray) -> int:
        """Draw one token id."""
        if self.config.is_greedy:
            return int(np.argmax(logits))

        probabilities = self.probabilities(logits)

        # Inverse-CDF draw. searchsorted on the cumulative distribution is the
        # definition of multinomial sampling, and unlike rng.choice it does not
        # require the probabilities to sum to exactly 1.0 in floating point.
        cumulative = np.cumsum(probabilities)
        drawn = int(np.searchsorted(cumulative, self._rng.random()))
        return min(drawn, len(probabilities) - 1)
