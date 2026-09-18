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

from nanoinfer.sampling import apply_temperature

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
