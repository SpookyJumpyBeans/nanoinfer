"""The phase 4 gate: the cache must change the cost and nothing else.

A KV cache is an optimization that is trivially easy to get almost right. Every
way of getting it wrong -- an off-by-one in the watermark, decode tokens
rotated at position zero, a stale slot attended over as though it were real --
produces finite numbers and fluent text. None of them raise.

So the gate is equality of *output*, not plausibility of output: the same token
IDs, from the same prompt, with and without the cache, on the real weights.
Logits are compared too, but IDs are the stronger claim, because an argmax can
break a near-tie either way and one divergence at step three changes every
token after it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.generate import greedy, greedy_stream
from nanoinfer.kvcache import CacheUsageError, KVCache
from nanoinfer.model import Qwen2
from nanoinfer.tokenizer import Tokenizer
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()

# The floor established in phase 3: splitting a matmul differently makes BLAS
# reassociate the sum, and float addition is not associative. The cache changes
# every matmul shape in the model, so exact bitwise equality is not available
# and claiming it would be claiming something false.
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Qwen2:
    return Qwen2(ModelWeights.load(build_tiny_model(tmp_path_factory.mktemp("cacheq"))))


# -- logits, on the tiny model ---------------------------------------------


def feed(model: Qwen2, ids: list[int], chunks: list[int], cache: KVCache) -> np.ndarray:
    """Run `ids` through the model in the given chunk sizes, using a cache."""
    outputs = []
    start = 0
    for size in chunks:
        outputs.append(model.forward(np.array(ids[start : start + size]), cache=cache))
        start += size
    return np.concatenate(outputs, axis=0)


@pytest.mark.parametrize(
    "chunks",
    [[8], [1] * 8, [5, 1, 1, 1], [4, 4], [1, 3, 1, 2, 1], [7, 1]],
    ids=["one_shot", "one_at_a_time", "prompt_then_decode", "halves", "ragged", "all_but_one"],
)
def test_chunking_does_not_change_the_logits(tiny_model, chunks):
    """Every way of splitting the same sequence must give the same answer.

    Chunk shape is what exercises the mask offsets and the position arithmetic,
    which is exactly where a cache goes wrong.
    """
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    uncached = tiny_model.forward(np.array(ids))
    cached = feed(tiny_model, ids, chunks, tiny_model.new_cache(16))

    assert cached.shape == uncached.shape
    np.testing.assert_allclose(cached, uncached, rtol=1e-4, atol=1e-5)


def test_cache_holds_every_token_afterwards(tiny_model):
    ids = [1, 2, 3, 4, 5]
    cache = tiny_model.new_cache(16)
    feed(tiny_model, ids, [3, 1, 1], cache)
    assert cache.length == len(ids)


def test_hidden_states_returns_one_row_per_new_token(tiny_model):
    cache = tiny_model.new_cache(16)
    tiny_model.forward(np.array([1, 2, 3]), cache=cache)
    assert tiny_model.forward(np.array([4]), cache=cache).shape[0] == 1


def test_the_watermark_moves_once_per_forward_not_once_per_layer(tiny_model):
    """24 layers writing the same tokens must advance the length by 1, not 24."""
    cache = tiny_model.new_cache(16)
    tiny_model.forward(np.array([1]), cache=cache)
    assert cache.length == 1


def test_reusing_a_dirty_cache_is_rejected(tiny_model):
    """A cache carrying another sequence would silently condition on it."""
    cache = tiny_model.new_cache(16)
    tiny_model.forward(np.array([1, 2, 3]), cache=cache)
    with pytest.raises(ValueError, match="reset it before"):
        list(greedy_stream(tiny_model, [4, 5], max_new_tokens=2, cache=cache))


def test_a_reset_cache_is_reusable(tiny_model):
    cache = tiny_model.new_cache(16)
    first = list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=4, cache=cache))
    cache.reset()
    second = list(greedy_stream(tiny_model, [1, 2, 3], max_new_tokens=4, cache=cache))
    assert first == second


def test_overflowing_the_cache_is_reported(tiny_model):
    cache = tiny_model.new_cache(4)
    with pytest.raises(Exception, match="cannot add"):
        tiny_model.forward(np.array([1, 2, 3, 4, 5]), cache=cache)


def test_a_layer_skipping_the_cache_would_be_caught(tiny_model):
    """The invariant commit() enforces, demonstrated rather than assumed."""
    cache = tiny_model.new_cache(8)
    keys = np.zeros(
        (tiny_model.config.num_key_value_heads, 1, tiny_model.config.head_dim),
        dtype=np.float32,
    )
    for layer in range(tiny_model.config.num_hidden_layers - 1):
        cache.extend(layer, keys, keys)
    with pytest.raises(CacheUsageError, match="did not write"):
        cache.commit(1)


# -- generation, on the tiny model -----------------------------------------


def test_cached_and_uncached_generation_agree(tiny_model):
    prompt = [1, 2, 3]
    uncached = list(greedy_stream(tiny_model, prompt, max_new_tokens=10))
    cached = list(
        greedy_stream(tiny_model, prompt, max_new_tokens=10, cache=tiny_model.new_cache(32))
    )
    assert cached == uncached


def test_greedy_defaults_to_using_a_cache(tiny_model):
    assert greedy(tiny_model, [1, 2, 3], max_new_tokens=3).used_cache


def test_greedy_can_be_told_not_to(tiny_model):
    assert not greedy(tiny_model, [1, 2, 3], max_new_tokens=3, use_cache=False).used_cache


def test_greedy_agrees_with_itself_across_both_paths(tiny_model):
    with_cache = greedy(tiny_model, [1, 2, 3], max_new_tokens=8)
    without = greedy(tiny_model, [1, 2, 3], max_new_tokens=8, use_cache=False)
    assert with_cache.generated_ids == without.generated_ids


def test_decode_rate_excludes_the_first_token(tiny_model):
    """One generated token means no steady state to measure yet."""
    result = greedy(tiny_model, [1, 2, 3], max_new_tokens=1)
    assert len(result.generated_ids) == 1
    assert result.decode_tokens_per_second == 0.0
    assert result.time_to_first_token_ms > 0


# -- the real model --------------------------------------------------------


@pytest.fixture(scope="module")
def real_model() -> Qwen2:
    return Qwen2.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def real_tokenizer() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
@pytest.mark.parametrize(
    "prompt,tokens",
    [
        ("The capital of France is", 16),
        ("1 2 3 4 5", 8),
        ("def add(a, b):", 10),
    ],
)
def test_generated_ids_are_identical_on_the_real_model(
    real_model, real_tokenizer, prompt, tokens
):
    """The gate. Same IDs, cached and uncached, on 494M real parameters."""
    ids = real_tokenizer.encode(prompt)

    uncached = list(greedy_stream(real_model, ids, max_new_tokens=tokens))
    cached = list(
        greedy_stream(
            real_model, ids, max_new_tokens=tokens,
            cache=real_model.new_cache(len(ids) + tokens),
        )
    )

    assert cached == uncached, (
        f"diverged for {prompt!r}\n"
        f"  uncached: {uncached} -> {real_tokenizer.decode(uncached)!r}\n"
        f"  cached  : {cached} -> {real_tokenizer.decode(cached)!r}"
    )


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_logits_agree_within_the_float_floor(real_model, real_tokenizer):
    """Close, but not bitwise equal, and the difference is not a defect.

    The cache changes the shape of every matmul in the model, so BLAS sums the
    same products in a different order. Float addition is not associative, so
    the results differ in the last bits. Asserting bitwise equality here would
    be asserting something false about floating point.
    """
    ids = real_tokenizer.encode("The capital of France is")

    uncached = real_model.next_token_logits(np.array(ids))
    cache = real_model.new_cache(len(ids) + 4)
    real_model.next_token_logits(np.array(ids[:3]), cache=cache)
    cached = real_model.next_token_logits(np.array(ids[3:]), cache=cache)

    difference = np.abs(cached - uncached).max()
    assert difference < TOLERANCE, f"logits differ by {difference:.3e}"
    assert int(cached.argmax()) == int(uncached.argmax())


def count_positions(model, monkeypatch) -> list[int]:
    """Record how many token positions each forward pass actually processes."""
    counts: list[int] = []
    original = model.embed

    def counting(token_ids):
        counts.append(len(np.asarray(token_ids)))
        return original(token_ids)

    monkeypatch.setattr(model, "embed", counting)
    return counts


def test_the_cache_removes_the_quadratic_work(tiny_model, monkeypatch):
    """Assert the work, not the wall clock.

    A timing assertion here would be measuring a thermally-throttling laptop
    and would flake; worse, the ratio it produced would depend on how many
    tokens happened to be generated. The number of token positions pushed
    through the model is exact, deterministic, and is the thing the cache
    actually changes.

    Without a cache, step i re-processes the whole sequence, so generating n
    tokens from a prompt of p costs ``sum(p + i for i in range(n))`` positions
    -- quadratic in n. With one, it costs ``p + (n - 1)``: the prompt once, then
    one position per step, minus the final token that is never fed back.

    Wall-clock numbers belong in bench/results.jsonl, where they are recorded
    with the machine they were measured on.
    """
    prompt = [1, 2, 3, 4, 5]
    n = 10
    p = len(prompt)

    counts = count_positions(tiny_model, monkeypatch)
    list(greedy_stream(tiny_model, prompt, max_new_tokens=n))
    uncached_positions = sum(counts)

    counts.clear()
    list(
        greedy_stream(
            tiny_model, prompt, max_new_tokens=n, cache=tiny_model.new_cache(p + n)
        )
    )
    cached_positions = sum(counts)

    assert uncached_positions == sum(p + i for i in range(n)) == 95
    assert cached_positions == p + (n - 1) == 14
    assert uncached_positions > 6 * cached_positions


def test_the_gap_widens_with_sequence_length(tiny_model, monkeypatch):
    """Quadratic against linear: the advantage is not a constant factor."""
    prompt = [1, 2, 3]
    ratios = []

    for n in (4, 16, 64):
        counts = count_positions(tiny_model, monkeypatch)
        list(greedy_stream(tiny_model, prompt, max_new_tokens=n))
        uncached = sum(counts)

        counts.clear()
        list(
            greedy_stream(
                tiny_model, prompt, max_new_tokens=n,
                cache=tiny_model.new_cache(len(prompt) + n),
            )
        )
        ratios.append(uncached / sum(counts))

    assert ratios == sorted(ratios), f"ratio should grow with n, got {ratios}"
    assert ratios[-1] > 4 * ratios[0]


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_the_real_model_does_the_same_reduced_work(real_model, real_tokenizer, monkeypatch):
    """The same accounting on the real weights, to catch a wiring difference."""
    ids = real_tokenizer.encode("The capital of France is")
    n = 8

    counts = count_positions(real_model, monkeypatch)
    list(greedy_stream(real_model, ids, max_new_tokens=n))
    uncached = sum(counts)

    counts.clear()
    list(
        greedy_stream(
            real_model, ids, max_new_tokens=n, cache=real_model.new_cache(len(ids) + n)
        )
    )
    cached = sum(counts)

    assert uncached == sum(len(ids) + i for i in range(n))
    assert cached == len(ids) + (n - 1)


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_cache_ends_one_token_short_by_design(real_model, real_tokenizer):
    """The last generated token is never fed back, saving a forward pass.

    Documented as a test because it is surprising: a caller planning to
    continue from the cache needs to know it is one behind.
    """
    ids = real_tokenizer.encode("The capital of France is")
    cache = real_model.new_cache(len(ids) + 6)
    produced = list(greedy_stream(real_model, ids, max_new_tokens=6, cache=cache))

    assert len(produced) == 6
    assert cache.length == len(ids) + 5
