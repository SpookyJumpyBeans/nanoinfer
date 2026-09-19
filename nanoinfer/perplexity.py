"""Perplexity on held-out text: the instrument phase 6 is measured with.

Quantization is a trade -- fewer bits for some loss of quality -- and the only
honest way to present it is with the loss quantified. Perplexity is the
standard number: the exponentiated average negative log-likelihood the model
assigns to text it has never seen.

    perplexity = exp( -1/N * sum log P(token_i | token_<i) )

Read it as "how many equally-likely options the model felt it was choosing
between, on average". Lower is better; 1.0 would be perfect prediction. What
matters here is not the absolute value but the **delta** between precisions,
measured on identical text with identical tokenization.

Two choices worth stating, because they change the number:

**Non-overlapping chunks, not a sliding window.** The text is split into
independent blocks and each is scored on its own. A sliding window would score
every token with a full context and give a lower (better-looking) perplexity,
at many times the compute. Since every precision level is scored the same way,
the delta -- which is the actual result -- is unaffected.

**The first token of each chunk is not scored.** It has no context to be
predicted from, so including it would fold the model's unconditional guess into
the average and make longer chunks look artificially better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """The score and enough detail to reproduce it."""

    perplexity: float
    mean_nll: float
    tokens_scored: int
    chunks: int
    chunk_size: int

    def __str__(self) -> str:
        return (
            f"perplexity {self.perplexity:.4f} "
            f"(nll {self.mean_nll:.4f}, {self.tokens_scored} tokens "
            f"in {self.chunks} chunks of {self.chunk_size})"
        )


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log of the softmax, computed without forming the softmax.

    ``log(exp(x_i) / sum(exp(x))) = x_i - max - log(sum(exp(x - max)))``.

    Going through the probabilities and taking a log afterwards would round a
    small probability to zero and yield ``-inf``, which would poison the mean.
    This form keeps the whole range representable.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def evaluate(
    model,
    token_ids: np.ndarray,
    chunk_size: int = 128,
    progress=None,
) -> PerplexityResult:
    """Score ``token_ids`` and return its perplexity under ``model``.

    Accumulates the negative log-likelihood in float64. The per-token values
    are float32, and summing a few thousand of them in float32 loses enough
    precision to move the fourth decimal place -- which is exactly the range
    the INT8 delta is expected to live in.
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if token_ids.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {token_ids.shape}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be at least 2, got {chunk_size}")
    if token_ids.size < 2:
        raise ValueError("need at least 2 tokens to score one prediction")

    total_nll = 0.0
    scored = 0
    chunks = 0

    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        if len(chunk) < 2:
            break   # a trailing single token has nothing to predict

        logits = model.forward(chunk)
        log_probs = log_softmax(logits.astype(np.float64))

        # Row i predicts token i+1, so the last row has no target.
        targets = chunk[1:]
        predicted = log_probs[:-1][np.arange(len(targets)), targets]

        total_nll += float(-predicted.sum())
        scored += len(targets)
        chunks += 1

        if progress is not None:
            progress(chunks, scored)

    mean_nll = total_nll / scored
    return PerplexityResult(
        perplexity=float(np.exp(mean_nll)),
        mean_nll=mean_nll,
        tokens_scored=scored,
        chunks=chunks,
        chunk_size=chunk_size,
    )


def held_out_tokens(tokenizer, path: str | Path | None = None, limit: int | None = None):
    """Tokenize the bundled held-out text.

    The text is a slice of Alice's Adventures in Wonderland (Project Gutenberg,
    public domain), committed to the repository so the numbers are reproducible
    without a network fetch. It is ordinary English prose the model was not
    trained on in this exact form, which is all a relative comparison needs.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "heldout.txt"
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    return np.array(ids[:limit] if limit else ids, dtype=np.int64)
