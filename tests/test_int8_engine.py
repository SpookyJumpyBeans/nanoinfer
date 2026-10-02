"""The forward pass running on int8 weights it actually stores as int8.

Phase 6 measured INT8's quality by rounding every weight to the integer grid
and handing it back as float32. Phase 7 wrote a kernel that consumes the
integers directly. This is the gate between them: the stored-int8 model must
compute the *same function* as the simulated one, because the simulation is
what the perplexity numbers were measured on. If the two disagreed, those
numbers would describe a model nobody runs.

They cannot agree bitwise. Both use the identical integer values and scales,
but the kernel scales once per row after the dot product while the simulation
scales each weight before it, and float addition is not associative. So the
logits are compared to a float32 tolerance, and the generated token IDs --
the claim that actually matters -- are compared exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer import kernels
from nanoinfer.generate import greedy_stream
from nanoinfer.linear import gather_rows, linear
from nanoinfer.model import Qwen2
from nanoinfer.quantization import (
    LINEAR_FIELDS,
    QuantizedTensor,
    quantize_int4,
    quantize_int8,
    quantize_model,
)
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()

needs_kernels = pytest.mark.skipif(
    not kernels.available(), reason="rust kernels not built"
)

# Summation order, and nothing else. Set by float32 over a hidden size of 16
# and two layers, not by what happened to pass.
TOLERANCE = 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory) -> ModelWeights:
    return ModelWeights.load(build_tiny_model(tmp_path_factory.mktemp("int8")))


def both(weights: ModelWeights, quantize_embeddings: bool = False) -> tuple[Qwen2, Qwen2]:
    """The simulated-int8 model and the stored-int8 model, from one source."""
    simulated, _ = quantize_model(weights, bits=8, quantize_embeddings=quantize_embeddings)
    stored, _ = quantize_model(
        weights, bits=8, quantize_embeddings=quantize_embeddings, dequantize=False
    )
    return Qwen2(simulated), Qwen2(stored)


# -- the batched kernel ----------------------------------------------------


@needs_kernels
def test_matmul_agrees_with_numpy(rng):
    for out_features, in_features, tokens in [(4, 8, 1), (64, 896, 7), (896, 4864, 3), (7, 33, 5)]:
        q = quantize_int8(rng.standard_normal((out_features, in_features)).astype(np.float32))
        x = rng.standard_normal((tokens, in_features)).astype(np.float32)

        expected = (x @ q.values.T.astype(np.float32)) * q.scales
        actual = kernels.matmul_i8(q.values, q.scales, x)

        assert actual.shape == (tokens, out_features)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


@needs_kernels
def test_a_batch_is_bitwise_a_loop_of_matvecs(rng):
    """Prefill and decode must not be two slightly different models.

    The batched kernel and the matvec share one per-row dot product, so
    processing a prompt in one call and token by token give identical bits.
    """
    q = quantize_int8(rng.standard_normal((4864, 896)).astype(np.float32))
    x = rng.standard_normal((6, 896)).astype(np.float32)

    batched = kernels.matmul_i8(q.values, q.scales, x)
    for t in range(6):
        np.testing.assert_array_equal(batched[t], kernels.matvec_i8(q.values, q.scales, x[t]))


@needs_kernels
def test_matmul_copies_a_non_contiguous_batch(rng):
    """Every other row of a larger array: right shape, wrong strides."""
    q = quantize_int8(rng.standard_normal((32, 64)).astype(np.float32))
    wide = rng.standard_normal((10, 64)).astype(np.float32)
    strided = wide[::2]
    assert not strided.flags.c_contiguous

    np.testing.assert_array_equal(
        kernels.matmul_i8(q.values, q.scales, strided),
        kernels.matmul_i8(q.values, q.scales, np.ascontiguousarray(strided)),
    )


@needs_kernels
def test_matmul_rejects_mismatched_shapes(rng):
    q = quantize_int8(rng.standard_normal((8, 16)).astype(np.float32))
    with pytest.raises(ValueError, match="width"):
        kernels.matmul_i8(q.values, q.scales, np.ones((2, 15), np.float32))
    with pytest.raises(ValueError, match="tokens, in_features"):
        kernels.matmul_i8(q.values, q.scales, np.ones(16, np.float32))
    with pytest.raises(ValueError, match="one scale per output row"):
        kernels.matmul_i8(q.values, q.scales[:-1], np.ones((2, 16), np.float32))


def test_matmul_without_a_library_says_how_to_build_it(monkeypatch):
    monkeypatch.setattr(kernels, "_LIBRARY", None)
    with pytest.raises(RuntimeError, match="cargo build --release"):
        kernels.matmul_i8(np.zeros((4, 8), np.int8), np.ones(4, np.float32), np.ones((1, 8), np.float32))


# -- linear() --------------------------------------------------------------


def test_float_weights_take_exactly_the_old_path(rng):
    """The fp32 engine must be untouched: bitwise, not approximately."""
    w = rng.standard_normal((24, 16)).astype(np.float32)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    np.testing.assert_array_equal(linear(x, w), x @ w.T)


def test_int8_matches_its_own_dequantization(rng):
    w = rng.standard_normal((48, 32)).astype(np.float32)
    q = quantize_int8(w)
    x = rng.standard_normal((3, 32)).astype(np.float32)
    np.testing.assert_allclose(linear(x, q), x @ q.dequantize().T, rtol=1e-5, atol=1e-5)


def test_numpy_fallback_computes_the_same_thing(rng, monkeypatch):
    """No compiled library must mean slow, never wrong."""
    q = quantize_int8(rng.standard_normal((48, 32)).astype(np.float32))
    x = rng.standard_normal((3, 32)).astype(np.float32)

    monkeypatch.setattr(kernels, "_LIBRARY", None)
    np.testing.assert_allclose(linear(x, q), x @ q.dequantize().T, rtol=1e-5, atol=1e-5)


def test_int4_is_refused_rather_than_silently_dequantized(rng):
    q = quantize_int4(rng.standard_normal((8, 32)).astype(np.float32), group_size=8)
    with pytest.raises(NotImplementedError, match="INT4"):
        linear(np.ones((1, 32), np.float32), q)


def test_gather_rows_widens_only_what_it_returns(rng):
    q = quantize_int8(rng.standard_normal((50, 16)).astype(np.float32))
    ids = np.array([3, 3, 49, 0])
    rows = gather_rows(q, ids)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows, q.dequantize()[ids], rtol=1e-6)


# -- quantize_model(dequantize=False) --------------------------------------


def test_stored_model_keeps_the_integers(tiny_weights):
    stored, report = quantize_model(tiny_weights, bits=8, dequantize=False)
    layer, source = stored.layers[0], tiny_weights.layers[0]

    for name in LINEAR_FIELDS:
        weight = getattr(layer, name)
        assert isinstance(weight, QuantizedTensor), name
        assert weight.values.dtype == np.int8
        reference = quantize_int8(getattr(source, name))
        np.testing.assert_array_equal(weight.values, reference.values)
        np.testing.assert_array_equal(weight.scales, reference.scales)

    # Norms and biases are still float32 arrays, shared with the source.
    assert layer.q_proj_bias is source.q_proj_bias
    assert layer.input_layernorm is source.input_layernorm
    # The footprint the report promises is the footprint actually held.
    assert stored.nbytes < tiny_weights.nbytes
    assert report.quantized_bytes == sum(
        getattr(l, f).nbytes for l in stored.layers for f in LINEAR_FIELDS
    )


def test_int4_cannot_be_stored(tiny_weights):
    with pytest.raises(ValueError, match="INT4 has no kernel"):
        quantize_model(tiny_weights, bits=4, dequantize=False)


# -- the gate --------------------------------------------------------------


@pytest.mark.parametrize("quantize_embeddings", [False, True])
def test_stored_int8_computes_the_simulated_function(tiny_weights, quantize_embeddings):
    simulated, stored = both(tiny_weights, quantize_embeddings)
    ids = np.array([5, 17, 3, 60, 22, 9, 41, 8])

    np.testing.assert_allclose(
        stored.forward(ids), simulated.forward(ids), rtol=0, atol=TOLERANCE
    )


@pytest.mark.parametrize("quantize_embeddings", [False, True])
def test_stored_int8_generates_the_same_tokens(tiny_weights, quantize_embeddings):
    simulated, stored = both(tiny_weights, quantize_embeddings)
    prompt = [5, 17, 3, 60, 22]

    expected = list(greedy_stream(simulated, prompt, max_new_tokens=24))
    actual = list(greedy_stream(stored, prompt, max_new_tokens=24))
    assert actual == expected


def test_cache_and_no_cache_agree_on_stored_int8(tiny_weights):
    """The phase 4 gate, rerun on the int8 path."""
    _, stored = both(tiny_weights)
    ids = np.array([5, 17, 3, 60, 22, 9])

    uncached = stored.forward(ids)
    cache = stored.new_cache(len(ids))
    stepped = np.concatenate(
        [stored.forward(ids[:3], cache=cache)]
        + [stored.forward(ids[i : i + 1], cache=cache) for i in range(3, len(ids))]
    )
    np.testing.assert_allclose(stepped, uncached, rtol=0, atol=TOLERANCE)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")
@needs_kernels
def test_real_model_int8_generates_what_simulation_does():
    from nanoinfer.tokenizer import Tokenizer

    weights = ModelWeights.load(MODEL_DIR)
    simulated, stored = both(weights)
    tokenizer = Tokenizer.from_model_dir(MODEL_DIR)

    for prompt in ["The capital of France is", "def fibonacci(n):"]:
        ids = tokenizer.encode(prompt)
        expected = list(greedy_stream(simulated, ids, max_new_tokens=32))
        actual = list(greedy_stream(stored, ids, max_new_tokens=32))
        assert actual == expected, prompt
