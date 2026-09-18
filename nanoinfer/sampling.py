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
