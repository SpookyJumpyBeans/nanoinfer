"""The visualizer must show the model the tests verify, not a copy of it.

It captures attention by wrapping ``nanoinfer.attention.softmax`` and keeps its
own decode loop so it can record logits. Both are ways to drift from the
engine, so both are pinned: the traced run must choose the same tokens as
``greedy_stream``, and the attention it assembles row by row from the cached
run must equal what a single uncached pass over the whole sequence computes.
"""

from __future__ import annotations

import numpy as np
import pytest

import nanoinfer.attention as attention_module
from nanoinfer.generate import greedy_stream
from nanoinfer.model import Qwen2
from nanoinfer.sampling import SamplingConfig
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model
from viz.trace import capture_attention, resample, traced_generate

PROMPT = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Qwen2:
    return Qwen2(ModelWeights.load(build_tiny_model(tmp_path_factory.mktemp("viz"))))


def test_traced_greedy_matches_the_engine(tiny_model):
    expected = list(greedy_stream(tiny_model, PROMPT, max_new_tokens=6))
    trace = traced_generate(tiny_model, PROMPT, 6)
    assert trace.generated_ids == expected


def test_capture_restores_softmax_even_after_an_error(tiny_model):
    original = attention_module.softmax
    with pytest.raises(RuntimeError):
        with capture_attention():
            raise RuntimeError("boom")
    assert attention_module.softmax is original


def test_one_capture_per_layer_per_pass(tiny_model):
    with capture_attention() as captured:
        tiny_model.forward(np.array(PROMPT))
    assert len(captured) == tiny_model.config.num_hidden_layers


def test_attention_is_square_causal_and_normalised(tiny_model):
    trace = traced_generate(tiny_model, PROMPT, 5)
    seq = len(trace.all_ids)
    assert len(trace.attention) == tiny_model.config.num_hidden_layers
    for layer in trace.attention:
        assert layer.shape == (tiny_model.config.num_attention_heads, seq, seq)
        # Nothing attends to the future...
        assert np.all(np.triu(layer, k=1) == 0)
        # ...and every row is a distribution, including the final token's,
        # which only exists because the loop feeds it back once more.
        np.testing.assert_allclose(layer.sum(axis=-1), 1.0, atol=1e-5)


def test_cached_rows_equal_one_uncached_pass(tiny_model):
    trace = traced_generate(tiny_model, PROMPT, 5)
    with capture_attention() as captured:
        tiny_model.forward(np.array(trace.all_ids))
    for assembled, reference in zip(trace.attention, captured):
        np.testing.assert_allclose(assembled, reference, atol=1e-5)


def test_stop_token_ends_generation(tiny_model):
    first = traced_generate(tiny_model, PROMPT, 6).generated_ids[0]
    trace = traced_generate(tiny_model, PROMPT, 6, stop_ids=[first])
    assert trace.generated_ids == [first]
    assert trace.stop_reason == "stop_token"
    assert trace.attention[0].shape[-1] == len(PROMPT) + 1


def test_candidates_are_ranked_and_include_the_greedy_choice(tiny_model):
    trace = traced_generate(tiny_model, PROMPT, 3)
    for step in trace.steps:
        probs = [c["p_model"] for c in step.candidates]
        assert probs == sorted(probs, reverse=True)
        assert step.candidates[0]["id"] == step.token_id


def test_sampled_choice_has_nonzero_sampled_probability(tiny_model):
    config = SamplingConfig(temperature=1.0, top_k=5, seed=7)
    trace = traced_generate(tiny_model, PROMPT, 6, config=config, top_k=50)
    for step in trace.steps:
        chosen = [c for c in step.candidates if c["id"] == step.token_id]
        assert chosen and chosen[0]["p_sampled"] > 0
        # top-k=5 leaves at most five tokens with any probability to draw
        assert sum(c["p_sampled"] > 0 for c in step.candidates) <= 5


def test_resample_top_k_one_is_greedy(tiny_model):
    trace = traced_generate(tiny_model, PROMPT, 1)
    cands = resample(trace.logits[0], SamplingConfig(temperature=1.0, top_k=1))
    assert cands[0]["id"] == trace.steps[0].token_id
    assert cands[0]["p_sampled"] == pytest.approx(1.0)
    assert all(c["p_sampled"] == 0 for c in cands[1:])
