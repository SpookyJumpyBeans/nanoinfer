"""Tests for the greedy decoding loop.

The headline test compares whole generated sequences against
``transformers.generate(do_sample=False)``, token for token. Matching logits
within 1e-3 does not by itself guarantee matching text: greedy decoding takes
an argmax, so two logit vectors that agree to 1e-5 can still disagree when the
top two candidates are within 1e-5 of each other, and one divergence at step
three changes everything after it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.chat import ChatTemplate
from nanoinfer.generate import greedy, greedy_stream
from nanoinfer.model import Qwen2
from nanoinfer.tokenizer import Tokenizer
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Qwen2:
    return Qwen2(ModelWeights.load(build_tiny_model(tmp_path_factory.mktemp("gen"))))


# -- the loop --------------------------------------------------------------


def test_generates_the_requested_number_of_tokens(tiny_model):
    out = list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=5))
    assert len(out) == 5


def test_zero_tokens_generates_nothing(tiny_model):
    assert list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=0)) == []


def test_negative_token_count_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="non-negative"):
        list(greedy_stream(tiny_model, [1], max_new_tokens=-1))


def test_empty_prompt_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="empty prompt"):
        greedy(tiny_model, [], max_new_tokens=1)


def test_greedy_is_deterministic(tiny_model):
    first = list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=6))
    second = list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=6))
    assert first == second


def test_each_step_takes_the_argmax(tiny_model):
    """Verify the first generated token directly against the logits."""
    prompt = [4, 5, 6]
    expected = int(np.argmax(tiny_model.next_token_logits(np.array(prompt))))
    assert next(iter(greedy_stream(tiny_model, prompt, max_new_tokens=1))) == expected


def test_generation_feeds_its_own_output_back(tiny_model):
    """Step two must condition on step one's token, not just the prompt."""
    prompt = [7, 8]
    produced = list(greedy_stream(tiny_model, prompt, max_new_tokens=2))
    extended = prompt + produced[:1]
    expected_second = int(np.argmax(tiny_model.next_token_logits(np.array(extended))))
    assert produced[1] == expected_second


# -- stopping --------------------------------------------------------------


def test_stops_on_a_stop_token(tiny_model):
    """Force a stop by declaring whatever it produces first as a stop token."""
    first = next(iter(greedy_stream(tiny_model, [1, 2], max_new_tokens=5)))
    out = list(greedy_stream(tiny_model, [1, 2], max_new_tokens=5, stop_ids=[first]))
    assert out == [first]


def test_the_stop_token_is_yielded_not_swallowed(tiny_model):
    """The caller must be able to tell why generation ended."""
    first = next(iter(greedy_stream(tiny_model, [1, 2], max_new_tokens=5)))
    out = list(greedy_stream(tiny_model, [1, 2], max_new_tokens=5, stop_ids=[first]))
    assert out[-1] == first


def test_stop_reason_distinguishes_the_two_endings(tiny_model):
    first = next(iter(greedy_stream(tiny_model, [1, 2], max_new_tokens=5)))

    stopped = greedy(tiny_model, [1, 2], max_new_tokens=5, stop_ids=[first])
    assert stopped.stop_reason == "stop_token"

    exhausted = greedy(tiny_model, [1, 2], max_new_tokens=3)
    assert exhausted.stop_reason == "max_tokens"


# -- the result object -----------------------------------------------------


def test_result_records_prompt_and_output(tiny_model):
    result = greedy(tiny_model, [1, 2, 3], max_new_tokens=4)
    assert result.prompt_ids == [1, 2, 3]
    assert len(result.generated_ids) == 4
    assert result.all_ids == [1, 2, 3] + result.generated_ids


def test_result_reports_both_timings(tiny_model):
    result = greedy(tiny_model, [1, 2, 3], max_new_tokens=3)
    assert result.prefill_s > 0
    assert result.decode_s > 0
    assert result.decode_tokens_per_second > 0
    assert result.time_to_first_token_ms == pytest.approx(result.prefill_s * 1000)


def test_rates_are_zero_rather_than_dividing_by_zero(tiny_model):
    result = greedy(tiny_model, [1], max_new_tokens=0)
    assert result.decode_tokens_per_second == 0.0


def test_on_token_callback_sees_every_token(tiny_model):
    seen: list[int] = []
    result = greedy(tiny_model, [1, 2], max_new_tokens=4, on_token=seen.append)
    assert seen == result.generated_ids


# -- against the reference -------------------------------------------------


@pytest.fixture(scope="module")
def real_model() -> Qwen2:
    return Qwen2.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def real_tokenizer() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def reference_model():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    return model


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize(
    "prompt,tokens",
    [
        ("The capital of France is", 8),
        ("1 2 3 4 5", 6),
        ("def add(a, b):", 8),
    ],
)
def test_greedy_output_matches_reference_token_for_token(
    real_model, real_tokenizer, reference_model, prompt, tokens
):
    """Whole sequences, not just the first token.

    Matching logits is necessary but not sufficient: greedy takes an argmax, so
    near-ties can break differently, and one divergence at step three changes
    every token after it.
    """
    import torch

    ids = real_tokenizer.encode(prompt)
    mine = list(greedy_stream(real_model, ids, max_new_tokens=tokens))

    with torch.no_grad():
        # repetition_penalty=1.0 is required, not optional. See
        # test_reference_generate_is_not_greedy_by_default below.
        out = reference_model.generate(
            torch.tensor([ids]),
            max_new_tokens=tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )
    theirs = out[0][len(ids) :].tolist()

    assert mine == theirs, (
        f"diverged for {prompt!r}\n"
        f"  mine  : {mine} -> {real_tokenizer.decode(mine)!r}\n"
        f"  theirs: {theirs} -> {real_tokenizer.decode(theirs)!r}"
    )


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.reference
@pytest.mark.slow
def test_chat_generation_matches_reference(real_model, real_tokenizer, reference_model):
    """The same check through the ChatML path, including stopping."""
    import torch
    from transformers import AutoTokenizer

    messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    rendered = ChatTemplate.from_model_dir(MODEL_DIR).render(messages)
    ids = real_tokenizer.encode(rendered)

    im_end = real_tokenizer.token_to_id("<|im_end|>")
    mine = list(greedy_stream(real_model, ids, max_new_tokens=16, stop_ids=[im_end]))

    reference_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    reference_ids = reference_tokenizer(rendered, return_tensors="pt").input_ids
    with torch.no_grad():
        out = reference_model.generate(
            reference_ids, max_new_tokens=16, do_sample=False
        )
    theirs = out[0][reference_ids.shape[1] :].tolist()

    assert mine == theirs
    assert mine[-1] == im_end, "generation should have stopped on the turn marker"


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.reference
@pytest.mark.slow
def test_reference_generate_is_not_greedy_by_default(
    real_model, real_tokenizer, reference_model
):
    """A trap worth pinning: do_sample=False is not the same as greedy.

    Qwen2.5 ships a generation_config.json declaring repetition_penalty 1.1,
    and transformers applies it inside generate() regardless of do_sample. So
    the "greedy" baseline quietly downweights tokens already in the context.

    On "The capital of France is" the two agree for three tokens and then part
    company -- ours continues "It is the largest city", the penalised reference
    "It was founded in 7". Chasing that as an attention bug would be days of
    work aimed at the wrong thing, so the difference is asserted here instead
    of being left to be rediscovered.
    """
    import torch

    assert reference_model.generation_config.repetition_penalty == 1.1

    ids = real_tokenizer.encode("The capital of France is")
    with torch.no_grad():
        penalised = reference_model.generate(
            torch.tensor([ids]), max_new_tokens=8, do_sample=False
        )[0][len(ids) :].tolist()
        pure = reference_model.generate(
            torch.tensor([ids]), max_new_tokens=8, do_sample=False, repetition_penalty=1.0
        )[0][len(ids) :].tolist()

    assert penalised != pure, "if these agreed, the trap would not exist"
    assert list(greedy_stream(real_model, ids, max_new_tokens=8)) == pure


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_a_human_readable_completion(real_model, real_tokenizer):
    """Independent of any reference: the output must actually say Paris."""
    ids = real_tokenizer.encode("The capital of France is")
    produced = list(greedy_stream(real_model, ids, max_new_tokens=3))
    assert "Paris" in real_tokenizer.decode(produced)
