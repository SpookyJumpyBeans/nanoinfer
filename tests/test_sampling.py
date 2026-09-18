"""Tests for the sampling stack, against transformers' own logits warpers.

The warpers are a genuine oracle here: they are the code every HuggingFace
generation call runs, so matching them exactly is what makes `temperature=0.7,
top_p=0.8` mean the same thing in this engine as everywhere else.

Differential testing does most of the work. Hand-picked cases catch the obvious
mistakes; random logit vectors compared against the reference catch the
boundary conditions nobody thinks to write down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.sampling import (
    Sampler,
    SamplingConfig,
    apply_temperature,
    top_k_filter,
    top_p_filter,
)

torch = pytest.importorskip("torch", reason="reference oracle not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def reference_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    from transformers.generation.logits_process import TemperatureLogitsWarper

    warper = TemperatureLogitsWarper(temperature)
    scores = torch.from_numpy(logits)[None, :]
    return warper(None, scores)[0].numpy()


# -- temperature -----------------------------------------------------------


def test_temperature_one_is_the_identity(rng):
    logits = rng.standard_normal(64).astype(np.float32) * 5
    np.testing.assert_allclose(apply_temperature(logits, 1.0), logits, rtol=1e-6)


def test_temperature_below_one_sharpens(rng):
    """Lower temperature increases the gap between best and second best."""
    logits = rng.standard_normal(64).astype(np.float32) * 2

    def top_two_gap(x):
        p = np.sort(np.exp(x - x.max()) / np.exp(x - x.max()).sum())[::-1]
        return p[0] - p[1]

    assert top_two_gap(apply_temperature(logits, 0.5)) > top_two_gap(logits)


def test_temperature_above_one_flattens(rng):
    logits = rng.standard_normal(64).astype(np.float32) * 2

    def top_two_gap(x):
        p = np.sort(np.exp(x - x.max()) / np.exp(x - x.max()).sum())[::-1]
        return p[0] - p[1]

    assert top_two_gap(apply_temperature(logits, 2.0)) < top_two_gap(logits)


def test_temperature_never_changes_the_argmax(rng):
    """Rescaling is monotonic, so it reorders nothing."""
    logits = rng.standard_normal(256).astype(np.float32) * 5
    best = int(np.argmax(logits))
    for t in (0.1, 0.5, 0.9, 1.0, 1.5, 10.0):
        assert int(np.argmax(apply_temperature(logits, t))) == best


def test_temperature_never_removes_support(rng):
    """Unlike top-k and top-p, temperature leaves every token reachable."""
    logits = rng.standard_normal(128).astype(np.float32)
    for t in (0.1, 1.0, 5.0):
        assert np.all(np.isfinite(apply_temperature(logits, t)))


@pytest.mark.reference
@pytest.mark.parametrize("temperature", [0.1, 0.5, 0.7, 1.0, 1.3, 2.0, 10.0])
def test_temperature_matches_reference(rng, temperature):
    logits = rng.standard_normal(512).astype(np.float32) * 6
    np.testing.assert_allclose(
        apply_temperature(logits, temperature),
        reference_temperature(logits, temperature),
        rtol=1e-6,
        atol=1e-6,
    )


def test_negative_temperature_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        apply_temperature(np.zeros(4, np.float32), -1.0)


def test_zero_temperature_is_rejected_here(rng):
    """Zero means greedy, which is the Sampler's job, not this function's."""
    with pytest.raises(ValueError, match="greedy"):
        apply_temperature(np.zeros(4, np.float32), 0.0)


def reference_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    from transformers.generation.logits_process import TopKLogitsWarper

    return TopKLogitsWarper(k)(None, torch.from_numpy(logits)[None, :])[0].numpy()


def kept(filtered: np.ndarray) -> np.ndarray:
    """Indices that survived a filter."""
    return np.flatnonzero(np.isfinite(filtered))


# -- top-k -----------------------------------------------------------------


def test_top_k_keeps_exactly_k_when_there_are_no_ties():
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(kept(top_k_filter(logits, 2)), [0, 1])


def test_top_k_masks_with_negative_infinity():
    logits = np.array([5.0, 1.0], dtype=np.float32)
    assert top_k_filter(logits, 1)[1] == -np.inf


def test_top_k_ties_keep_more_than_k():
    """The cut is 'strictly below the k-th value', so ties all survive.

    Breaking the tie by index would make the output depend on vocabulary
    order, which is arbitrary. The reference makes the same choice.
    """
    logits = np.array([5.0, 3.0, 3.0, 3.0, 1.0], dtype=np.float32)
    survivors = kept(top_k_filter(logits, 2))
    assert list(survivors) == [0, 1, 2, 3]
    assert len(survivors) > 2


def test_top_k_zero_disables_the_filter():
    logits = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(top_k_filter(logits, 0), logits)


def test_top_k_larger_than_vocab_is_a_no_op():
    logits = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(top_k_filter(logits, 99), logits)


def test_top_k_one_leaves_only_the_argmax():
    logits = np.array([1.0, 7.0, 3.0], dtype=np.float32)
    assert list(kept(top_k_filter(logits, 1))) == [1]


def test_top_k_does_not_mutate_its_input():
    logits = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    top_k_filter(logits, 1)
    np.testing.assert_array_equal(logits, [3.0, 1.0, 2.0])


def test_negative_top_k_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        top_k_filter(np.zeros(4, np.float32), -1)


@pytest.mark.reference
def test_top_k_matches_reference_on_random_logits(rng):
    """Differential test, including deliberately forced ties."""
    for trial in range(200):
        n = int(rng.integers(5, 200))
        k = int(rng.integers(1, n + 3))
        logits = (rng.standard_normal(n) * rng.uniform(0.1, 8)).astype(np.float32)
        if trial % 4 == 0:
            logits[rng.integers(0, n, size=max(2, n // 5))] = logits[0]

        mine = top_k_filter(logits, k)
        theirs = reference_top_k(logits, k)
        np.testing.assert_array_equal(
            np.isinf(mine), np.isinf(theirs), err_msg=f"n={n} k={k}"
        )
        np.testing.assert_allclose(mine[kept(mine)], theirs[kept(theirs)], rtol=1e-6)


def reference_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    from transformers.generation.logits_process import TopPLogitsWarper

    return TopPLogitsWarper(p)(None, torch.from_numpy(logits)[None, :])[0].numpy()


# -- top-p -----------------------------------------------------------------


def test_top_p_keeps_the_smallest_set_reaching_p():
    # probabilities are roughly 0.64, 0.24, 0.09, 0.03
    logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert list(kept(top_p_filter(logits, 0.6))) == [0]
    assert list(kept(top_p_filter(logits, 0.8))) == [0, 1]


def test_top_p_one_disables_the_filter():
    logits = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(top_p_filter(logits, 1.0), logits)


def test_top_p_always_keeps_at_least_one_token():
    """Even a p below the largest single probability leaves the argmax."""
    logits = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    survivors = kept(top_p_filter(logits, 1e-9))
    assert list(survivors) == [0]


def test_top_p_nucleus_adapts_to_confidence():
    """The point of preferring top-p to top-k: the cut is not a fixed size."""
    confident = np.array([20.0] + [0.0] * 99, dtype=np.float32)
    unsure = np.zeros(100, dtype=np.float32)
    assert len(kept(top_p_filter(confident, 0.9))) == 1
    assert len(kept(top_p_filter(unsure, 0.9))) > 50


def test_top_p_does_not_mutate_its_input():
    logits = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    top_p_filter(logits, 0.5)
    np.testing.assert_array_equal(logits, [3.0, 1.0, 2.0])


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_top_p_out_of_range_is_rejected(p):
    with pytest.raises(ValueError, match="top_p must be in"):
        top_p_filter(np.zeros(4, np.float32), p)


@pytest.mark.reference
def test_top_p_matches_reference_exactly_without_ties(rng):
    """On continuous logits -- what a real model produces -- indices match.

    Note the implementation accumulates ASCENDING and cuts at `<= 1 - p`,
    mirroring the reference, rather than accumulating descending and keeping
    until the sum reaches p. The two are algebraically identical and differ in
    float32: for 100 equiprobable tokens at p=0.5 the descending sum reaches
    0.4999997913837433, so one formulation keeps a token the other drops.
    """
    for _ in range(400):
        n = int(rng.integers(3, 400))
        p = float(rng.uniform(0.01, 0.999))
        logits = (rng.standard_normal(n) * rng.uniform(0.1, 10)).astype(np.float32)

        mine = top_p_filter(logits, p)
        theirs = reference_top_p(logits, p)
        np.testing.assert_array_equal(
            np.isinf(mine), np.isinf(theirs), err_msg=f"n={n} p={p}"
        )


@pytest.mark.reference
def test_top_p_agrees_on_size_when_ties_straddle_the_boundary(rng):
    """With exact ties, which tied token survives is genuinely undefined.

    When the nucleus boundary falls inside a group of equal logits, the
    surviving members depend on sort order within that group -- and the
    reference calls torch.sort with stable=False, so its own answer is
    unspecified and may change between torch versions.

    What *is* well defined, and what this asserts, is that both implementations
    keep the same number of tokens and the same multiset of logit values.

    Exact ties are not synthetic, either. A real forward pass of this model
    produces about 620 exact float32 collisions across its 151,936 logits --
    distinct values that round to the same float. They sit in the
    low-probability tail, so a realistic nucleus boundary does not fall inside
    a tied group, but the ties themselves are real and this is why the limit is
    tested rather than assumed away.
    """
    for _ in range(60):
        n = int(rng.integers(4, 60))
        logits = rng.integers(0, 3, size=n).astype(np.float32)
        p = float(rng.uniform(0.05, 0.95))

        mine = top_p_filter(logits, p)
        theirs = reference_top_p(logits, p)

        assert len(kept(mine)) == len(kept(theirs))
        assert sorted(logits[kept(mine)]) == sorted(logits[kept(theirs)])


def test_top_p_is_deterministic_under_ties():
    """Our own answer must at least be stable run to run, for reproducibility."""
    logits = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    first = kept(top_p_filter(logits, 0.5))
    for _ in range(5):
        np.testing.assert_array_equal(kept(top_p_filter(logits, 0.5)), first)


# -- the sampler -----------------------------------------------------------


def test_greedy_config_takes_the_argmax(rng):
    logits = rng.standard_normal(500).astype(np.float32) * 4
    sampler = Sampler(SamplingConfig.greedy())
    assert sampler(logits) == int(np.argmax(logits))


def test_greedy_ignores_the_seed(rng):
    logits = rng.standard_normal(100).astype(np.float32)
    a = Sampler(SamplingConfig(temperature=0.0, seed=1))(logits)
    b = Sampler(SamplingConfig(temperature=0.0, seed=999))(logits)
    assert a == b == int(np.argmax(logits))


def test_same_seed_gives_the_same_sequence(rng):
    """The reproducibility the phase is graded on."""
    logits = rng.standard_normal(1000).astype(np.float32) * 3
    config = SamplingConfig(temperature=0.8, top_k=40, top_p=0.9, seed=1234)

    first = [Sampler(config)(logits) for _ in range(1)]
    one = Sampler(config)
    two = Sampler(config)
    assert [one(logits) for _ in range(20)] == [two(logits) for _ in range(20)]
    assert first == [Sampler(config)(logits)]


def test_different_seeds_diverge(rng):
    logits = rng.standard_normal(1000).astype(np.float32) * 3
    a = Sampler(SamplingConfig(temperature=1.0, seed=1))
    b = Sampler(SamplingConfig(temperature=1.0, seed=2))
    assert [a(logits) for _ in range(20)] != [b(logits) for _ in range(20)]


def test_reset_replays_the_stream(rng):
    logits = rng.standard_normal(500).astype(np.float32) * 3
    sampler = Sampler(SamplingConfig(temperature=1.0, seed=7))
    first = [sampler(logits) for _ in range(15)]
    sampler.reset()
    assert [sampler(logits) for _ in range(15)] == first


def test_without_reset_the_stream_continues(rng):
    """Reusing a sampler is not a repeat; documented so it cannot surprise."""
    logits = rng.standard_normal(500).astype(np.float32) * 3
    sampler = Sampler(SamplingConfig(temperature=1.0, seed=7))
    first = [sampler(logits) for _ in range(15)]
    second = [sampler(logits) for _ in range(15)]
    assert first != second


def test_filtered_tokens_are_never_drawn(rng):
    """A masked token has probability exactly zero, not merely a small one."""
    logits = rng.standard_normal(200).astype(np.float32) * 2
    config = SamplingConfig(temperature=1.0, top_k=5, seed=3)
    sampler = Sampler(config)

    allowed = set(kept(sampler.filter(logits)))
    assert len(allowed) == 5
    drawn = {sampler(logits) for _ in range(2000)}
    assert drawn <= allowed


def test_draws_follow_the_filtered_distribution():
    """Statistical check: empirical frequencies track the probabilities."""
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))
    sampler = Sampler(SamplingConfig(temperature=1.0, seed=11))

    expected = sampler.probabilities(logits)
    counts = np.zeros(4)
    draws = 40_000
    for _ in range(draws):
        counts[sampler(logits)] += 1

    np.testing.assert_allclose(counts / draws, expected, atol=0.01)


def test_probabilities_sum_to_one(rng):
    logits = rng.standard_normal(300).astype(np.float32) * 3
    sampler = Sampler(SamplingConfig(temperature=0.7, top_k=20, top_p=0.8))
    assert sampler.probabilities(logits).sum() == pytest.approx(1.0)


def test_temperature_is_applied_before_truncation(rng):
    """Order matters: top-p thresholds the temperature-scaled distribution.

    If top-p ran first it would threshold a distribution the model never
    produced, and top_p would mean something different at every temperature.
    """
    logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    hot = Sampler(SamplingConfig(temperature=5.0, top_p=0.8))
    cold = Sampler(SamplingConfig(temperature=0.2, top_p=0.8))
    assert len(kept(hot.filter(logits))) > len(kept(cold.filter(logits)))


def test_default_config_is_the_identity(rng):
    """No temperature change, no truncation: the model's own distribution."""
    logits = rng.standard_normal(100).astype(np.float32) * 2
    sampler = Sampler()
    np.testing.assert_allclose(sampler.filter(logits), logits, rtol=1e-6)


def test_reads_the_models_shipped_defaults():
    from pathlib import Path

    model_dir = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
    if not (model_dir / "generation_config.json").exists():
        pytest.skip("model not downloaded")

    config = SamplingConfig.from_model_dir(model_dir, seed=42)
    assert config.temperature == 0.7
    assert config.top_k == 20
    assert config.top_p == 0.8
    assert config.seed == 42


# -- end to end on the real model ------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()


@pytest.fixture(scope="module")
def real_model():
    from nanoinfer.model import Qwen2

    return Qwen2.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def real_tokenizer():
    from nanoinfer.tokenizer import Tokenizer

    return Tokenizer.from_model_dir(MODEL_DIR)


def run(model, ids, sampler, n=12):
    from nanoinfer.generate import generate_stream

    return list(
        generate_stream(
            model, ids, max_new_tokens=n,
            cache=model.new_cache(len(ids) + n), sampler=sampler,
        )
    )


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_same_seed_reproduces_the_same_text(real_model, real_tokenizer):
    """The phase 5 gate: a fixed seed makes sampled output reproducible."""
    ids = real_tokenizer.encode("The capital of France is")
    config = SamplingConfig(temperature=0.8, top_k=40, top_p=0.9, seed=20260917)

    first = run(real_model, ids, Sampler(config))
    second = run(real_model, ids, Sampler(config))

    assert first == second, (
        f"same seed diverged\n  first : {real_tokenizer.decode(first)!r}"
        f"\n  second: {real_tokenizer.decode(second)!r}"
    )


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_different_seeds_give_different_text(real_model, real_tokenizer):
    ids = real_tokenizer.encode("Once upon a time")
    a = run(real_model, ids, Sampler(SamplingConfig(temperature=1.0, seed=1)))
    b = run(real_model, ids, Sampler(SamplingConfig(temperature=1.0, seed=2)))
    assert a != b


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_temperature_zero_matches_the_greedy_path(real_model, real_tokenizer):
    """A sampler at temperature 0 must reproduce phase 4's output exactly."""
    from nanoinfer.generate import greedy_stream

    ids = real_tokenizer.encode("The capital of France is")
    n = 10
    sampled = run(real_model, ids, Sampler(SamplingConfig.greedy()), n)
    greedy = list(
        greedy_stream(
            real_model, ids, max_new_tokens=n, cache=real_model.new_cache(len(ids) + n)
        )
    )
    assert sampled == greedy


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_sampled_tokens_are_always_decodable(real_model, real_tokenizer):
    """A high temperature must not wander into the untrained padding IDs.

    The embedding matrix has 151,936 rows but the tokenizer knows 151,665
    tokens. Those 271 extra rows are reachable logits with no token, and a hot
    enough sampler could pick one -- which would raise on decode.
    """
    ids = real_tokenizer.encode("Once upon a time")
    produced = run(real_model, ids, Sampler(SamplingConfig(temperature=1.5, seed=5)), 24)

    assert all(0 <= t < real_tokenizer.vocab_size for t in produced), (
        "sampled an ID with no token: "
        f"{[t for t in produced if t >= real_tokenizer.vocab_size]}"
    )
    real_tokenizer.decode(produced)


@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@pytest.mark.slow
def test_untrained_padding_rows_are_negligible_but_reachable(real_model, real_tokenizer):
    """The 271 embedding rows past the vocabulary, quantified.

    config.vocab_size is 151,936 while the tokenizer knows 151,665 tokens. The
    difference is padding to a round number for kernel efficiency, and those
    rows were never trained -- so they are addressable logits with no token
    behind them, and decoding one raises.

    In practice they are harmless: all 271 sit in a band about 6e-3 wide near
    rank 96,000 and together hold under 1e-6 of the probability mass, and any
    top-k or top-p setting removes them outright. This records the measurement
    so the "harmless" claim is a number rather than a hope.
    """
    ids = real_tokenizer.encode("Once upon a time")
    logits = real_model.next_token_logits(np.array(ids))

    padding = logits[real_tokenizer.vocab_size:]
    assert len(padding) == 271
    # Tightly clustered but not identical: untrained rows still differ
    # slightly because the hidden state they are projected against is not zero.
    assert padding.max() - padding.min() < 0.05

    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    assert probabilities[real_tokenizer.vocab_size:].sum() < 1e-6

    # Any truncation removes them entirely.
    filtered = Sampler(SamplingConfig(temperature=1.0, top_k=50)).filter(logits)
    assert np.all(np.isinf(filtered[real_tokenizer.vocab_size:]))
