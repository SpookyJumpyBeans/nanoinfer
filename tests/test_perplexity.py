"""Tests for the perplexity harness.

The harness is the instrument phase 6's results are read off, so it is tested
against cases whose answer is known in closed form -- a model that predicts
perfectly should score exactly 1.0, a uniform model exactly the vocabulary
size -- rather than only against itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.perplexity import evaluate, held_out_tokens, log_softmax

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()


class ScriptedModel:
    """A stand-in that returns logits we choose, so the maths is checkable."""

    def __init__(self, vocab: int, logits_for=None):
        self.config = type("cfg", (), {"vocab_size": vocab})()
        self.vocab = vocab
        self._logits_for = logits_for

    def forward(self, token_ids, positions=None, last_only=False, cache=None):
        if self._logits_for is not None:
            return self._logits_for(token_ids)
        return np.zeros((len(token_ids), self.vocab), dtype=np.float32)


# -- log_softmax -----------------------------------------------------------


def test_log_softmax_matches_the_definition():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    expected = logits - np.log(np.exp(logits).sum())
    np.testing.assert_allclose(log_softmax(logits), expected, rtol=1e-12)


def test_log_softmax_rows_exponentiate_to_one():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 50)) * 8
    np.testing.assert_allclose(np.exp(log_softmax(logits)).sum(axis=-1), 1.0, rtol=1e-10)


def test_log_softmax_survives_underflow():
    """Going via probabilities would give -inf here and poison the mean."""
    logits = np.array([[0.0, -900.0]], dtype=np.float64)
    out = log_softmax(logits)
    assert np.all(np.isfinite(out))
    assert out[0, 1] < -800


def test_log_softmax_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal((4, 100)) * 10).astype(np.float64)
    np.testing.assert_allclose(
        log_softmax(logits),
        torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy(),
        rtol=1e-10,
    )


# -- the score itself ------------------------------------------------------


def test_uniform_model_scores_exactly_the_vocabulary_size():
    """Closed form: predicting uniformly over V options gives perplexity V."""
    vocab = 37
    model = ScriptedModel(vocab)          # all-zero logits == uniform
    ids = np.arange(20) % vocab
    result = evaluate(model, ids, chunk_size=10)
    assert result.perplexity == pytest.approx(vocab, rel=1e-9)


def test_perfect_model_scores_one():
    """A model that always puts all its mass on the right token scores 1.0."""
    vocab = 50
    ids = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)

    def oracle(chunk):
        logits = np.full((len(chunk), vocab), -1e4, dtype=np.float64)
        for i in range(len(chunk) - 1):
            logits[i, chunk[i + 1]] = 1e4     # certainty on the true next token
        return logits

    result = evaluate(ScriptedModel(vocab, oracle), ids, chunk_size=8)
    assert result.perplexity == pytest.approx(1.0, abs=1e-6)


def test_first_token_of_each_chunk_is_not_scored():
    """It has no context, so scoring it would reward longer chunks."""
    model = ScriptedModel(10)
    ids = np.arange(20) % 10
    for chunk_size, expected_chunks in ((10, 2), (5, 4)):
        result = evaluate(model, ids, chunk_size=chunk_size)
        assert result.chunks == expected_chunks
        assert result.tokens_scored == len(ids) - expected_chunks


def test_a_trailing_lone_token_is_dropped():
    """A final chunk of one token has nothing to predict."""
    model = ScriptedModel(10)
    result = evaluate(model, np.arange(11) % 10, chunk_size=5)
    assert result.chunks == 2
    assert result.tokens_scored == 8


def test_score_is_deterministic():
    model = ScriptedModel(10)
    ids = np.arange(30) % 10
    first = evaluate(model, ids, chunk_size=8)
    assert evaluate(model, ids, chunk_size=8).perplexity == first.perplexity


def test_perplexity_and_nll_agree():
    model = ScriptedModel(19)
    result = evaluate(model, np.arange(20) % 19, chunk_size=10)
    assert result.perplexity == pytest.approx(np.exp(result.mean_nll))


def test_rejects_a_sequence_too_short_to_score():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        evaluate(ScriptedModel(10), np.array([1]), chunk_size=4)


def test_rejects_a_degenerate_chunk_size():
    with pytest.raises(ValueError, match="at least 2"):
        evaluate(ScriptedModel(10), np.arange(10), chunk_size=1)


def test_rejects_batched_input():
    with pytest.raises(ValueError, match="1-D"):
        evaluate(ScriptedModel(10), np.zeros((2, 5), dtype=np.int64))


# -- the bundled corpus ----------------------------------------------------


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
def test_held_out_text_is_present_and_tokenizes():
    from nanoinfer.tokenizer import Tokenizer

    ids = held_out_tokens(Tokenizer.from_model_dir(MODEL_DIR))
    assert len(ids) > 2000, "corpus should be long enough for a stable score"
    assert ids.dtype == np.int64


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
def test_held_out_text_is_committed_not_fetched():
    """Reproducible forever, with no network dependency."""
    assert (Path(__file__).resolve().parent.parent / "data" / "heldout.txt").exists()
