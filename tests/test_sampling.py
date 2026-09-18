"""Tests for the sampling stack, against transformers' own logits warpers.

The warpers are a genuine oracle here: they are the code every HuggingFace
generation call runs, so matching them exactly is what makes `temperature=0.7,
top_p=0.8` mean the same thing in this engine as everywhere else.

Differential testing does most of the work. Hand-picked cases catch the obvious
mistakes; random logit vectors compared against the reference catch the
boundary conditions nobody thinks to write down.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoinfer.sampling import apply_temperature, top_k_filter, top_p_filter

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
    keep the same number of tokens and the same multiset of logit values. A
    trained model does not produce exact float ties across a tied group, so
    this case is synthetic; it is tested to document the limit of the claim
    rather than because it can arise.
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
