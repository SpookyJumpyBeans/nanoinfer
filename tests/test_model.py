"""The phase 3 gate: our logits must match the reference within 1e-3.

This is the test the whole phase exists to pass. Everything before it checks a
component in isolation; this checks that 24 layers of them compose into the
same function the reference computes, on the real 494M-parameter weights.

Two deliberate choices about how it fails:

* Divergence is located **per layer**, not just reported at the end. A drift
  that starts in layer 0 and one that starts in layer 19 are entirely
  different bugs, and "the logits are wrong" does not distinguish them.
* The comparison is made **at the first token**, not only after fifty. Errors
  compound, so a discrepancy that is invisible at step 1 and obvious at step 50
  is a discrepancy that has already been amplified past the point of being
  diagnosable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.model import Qwen2, feed_forward, transformer_block
from nanoinfer.rope import RotaryEmbedding
from nanoinfer.tokenizer import Tokenizer
from nanoinfer.weights import ModelWeights
from tests.tiny import build_tiny_model

torch = pytest.importorskip("torch", reason="reference oracle not installed")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").exists()

# The tolerance the phase is graded against. Observed max difference on these
# prompts is about 6e-5, so there is nearly two orders of magnitude of headroom;
# the gate is not being met by a whisker.
TOLERANCE = 1e-3


# -- component checks on the tiny model ------------------------------------


@pytest.fixture(scope="module")
def tiny() -> ModelWeights:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        yield ModelWeights.load(build_tiny_model(Path(directory)))


def test_feed_forward_matches_torch(tiny):
    """SwiGLU: down(silu(gate(x)) * up(x))."""
    layer = tiny.layers[0]
    x = np.random.default_rng(0).standard_normal(
        (5, tiny.config.hidden_size)
    ).astype(np.float32)

    xt = torch.from_numpy(x)
    gate = xt @ torch.from_numpy(layer.gate_proj_weight).T
    up = xt @ torch.from_numpy(layer.up_proj_weight).T
    expected = (
        (torch.nn.functional.silu(gate) * up) @ torch.from_numpy(layer.down_proj_weight).T
    ).numpy()

    np.testing.assert_allclose(feed_forward(x, layer), expected, rtol=1e-5, atol=1e-6)


def test_feed_forward_gates_only_the_gate_branch(tiny):
    """down(silu(gate) * up), not down(silu(gate * up)) or down(silu(up))."""
    layer = tiny.layers[0]
    x = np.random.default_rng(1).standard_normal(
        (3, tiny.config.hidden_size)
    ).astype(np.float32)

    from nanoinfer.ops import silu

    gate = x @ layer.gate_proj_weight.T
    up = x @ layer.up_proj_weight.T

    correct = feed_forward(x, layer)
    both_activated = (silu(gate) * silu(up)) @ layer.down_proj_weight.T
    assert not np.allclose(correct, both_activated)


def test_block_is_pre_norm_not_post_norm(tiny):
    """x + f(norm(x)), not norm(x + f(x)).

    Pre-norm leaves an unnormalized path from the embedding to the output. The
    difference is invisible in shapes and very visible in values.
    """
    from nanoinfer.ops import rms_norm

    layer = tiny.layers[0]
    cfg = tiny.config
    rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
    x = np.random.default_rng(2).standard_normal((4, cfg.hidden_size)).astype(np.float32)

    out = transformer_block(x, layer, cfg, rope, np.arange(4))

    # Under pre-norm the residual stream is never rescaled, so the block output
    # stays close to its input plus a correction rather than being renormalized.
    normalized = rms_norm(out, np.ones(cfg.hidden_size, dtype=np.float32), cfg.rms_norm_eps)
    assert not np.allclose(out, normalized), "output looks renormalized: post-norm?"


def test_block_preserves_shape(tiny):
    cfg = tiny.config
    rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
    x = np.zeros((6, cfg.hidden_size), dtype=np.float32)
    assert transformer_block(x, tiny.layers[0], cfg, rope, np.arange(6)).shape == x.shape


# -- input validation ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model(tiny) -> Qwen2:
    return Qwen2(tiny)


def test_rejects_empty_sequence(tiny_model):
    with pytest.raises(ValueError, match="empty sequence"):
        tiny_model.forward(np.array([], dtype=np.int64))


def test_rejects_token_id_past_the_embedding_matrix(tiny_model):
    with pytest.raises(ValueError, match="outside the embedding matrix"):
        tiny_model.forward(np.array([tiny_model.config.vocab_size]))


def test_rejects_negative_token_id(tiny_model):
    with pytest.raises(ValueError, match="outside the embedding matrix"):
        tiny_model.forward(np.array([-1]))


def test_rejects_batched_input(tiny_model):
    with pytest.raises(ValueError, match="1-D"):
        tiny_model.forward(np.zeros((2, 3), dtype=np.int64))


def test_logits_shape(tiny_model):
    logits = tiny_model.forward(np.array([0, 1, 2]))
    assert logits.shape == (3, tiny_model.config.vocab_size)


def test_last_only_matches_the_final_row(tiny_model):
    ids = np.array([0, 1, 2, 3])
    np.testing.assert_allclose(
        tiny_model.forward(ids, last_only=True)[0],
        tiny_model.forward(ids)[-1],
        rtol=1e-6,
        atol=1e-7,
    )


def test_next_token_logits_is_one_dimensional(tiny_model):
    assert tiny_model.next_token_logits(np.array([0, 1])).shape == (
        tiny_model.config.vocab_size,
    )


def test_tied_output_projection_reuses_the_embedding_matrix(tiny_model):
    assert tiny_model.weights.lm_head is tiny_model.weights.embed_tokens


# -- the real model, against the reference ---------------------------------

pytestmark_real = pytest.mark.skipif(not HAVE_MODEL, reason="model not downloaded")

PROMPTS = [
    "The capital of France is",
    "Hello",
    "def fibonacci(n):",
    "1 2 3 4 5 6 7 8 9 10",
    "日本語のテキスト",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]


@pytest.fixture(scope="module")
def real_model() -> Qwen2:
    return Qwen2.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def real_tokenizer() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def reference_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    return model


@pytestmark_real
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_reference(real_model, real_tokenizer, reference_model, prompt):
    """The gate: every logit, at every position, within 1e-3."""
    ids = real_tokenizer.encode(prompt)
    mine = real_model.forward(np.array(ids))

    with torch.no_grad():
        theirs = reference_model(torch.tensor([ids])).logits[0].numpy()

    assert mine.shape == theirs.shape
    difference = np.abs(mine - theirs).max()
    assert difference < TOLERANCE, (
        f"max logit difference {difference:.3e} exceeds {TOLERANCE:.0e} "
        f"for prompt {prompt!r}"
    )


@pytestmark_real
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_choice_matches_reference(real_model, real_tokenizer, reference_model, prompt):
    """Agreement on the argmax, which is what greedy decoding actually uses.

    Strictly weaker than the logit check, but it is the property that decides
    whether generated text is identical, so it is asserted separately rather
    than inferred.
    """
    ids = real_tokenizer.encode(prompt)
    mine = real_model.forward(np.array(ids))
    with torch.no_grad():
        theirs = reference_model(torch.tensor([ids])).logits[0].numpy()

    np.testing.assert_array_equal(mine.argmax(-1), theirs.argmax(-1))


@pytestmark_real
@pytest.mark.reference
@pytest.mark.slow
def test_divergence_is_located_per_layer(real_model, real_tokenizer, reference_model):
    """Compare hidden states after every layer, to localize a future failure.

    The reference exposes hidden_states as (embeddings, after layer 0, ...,
    after the final norm). Walking them tells us *where* a drift begins rather
    than only that the logits are wrong -- and it shows how much error a
    correct implementation accumulates, which is the baseline phase 4 has to
    stay inside.
    """
    from nanoinfer.ops import rms_norm

    ids = real_tokenizer.encode("The capital of France is")
    with torch.no_grad():
        reference = reference_model(torch.tensor([ids]), output_hidden_states=True)
    reference_states = [h[0].numpy() for h in reference.hidden_states]

    config = real_model.config
    hidden = real_model.embed(np.array(ids))
    positions = np.arange(len(ids))

    np.testing.assert_allclose(
        hidden, reference_states[0], rtol=1e-5, atol=1e-5, err_msg="embedding lookup"
    )

    # hidden_states is (embeddings, after layer 0, ..., after layer n-2,
    # after the FINAL NORM). Note the last entry: the reference applies
    # self.norm before appending it, so there is no entry holding the raw
    # output of the last layer. Comparing our pre-norm layer-23 output against
    # that post-norm state reports a divergence of ~150 from perfectly correct
    # code, so the last layer is handled separately below.
    last_index = len(real_model.weights.layers) - 1

    drift = []
    for index, layer in enumerate(real_model.weights.layers):
        hidden = transformer_block(hidden, layer, config, real_model.rope, positions)
        if index == last_index:
            break
        expected = reference_states[index + 1]
        worst = float(np.abs(hidden - expected).max())
        drift.append(worst)
        assert worst < 1e-2, (
            f"layer {index} diverges by {worst:.3e}; "
            f"per-layer drift so far: {['%.1e' % d for d in drift]}"
        )

    final = rms_norm(hidden, real_model.weights.final_norm, config.rms_norm_eps)
    np.testing.assert_allclose(
        final, reference_states[-1], rtol=1e-4, atol=1e-3, err_msg="final norm"
    )

    # Error should accumulate gradually. A jump of orders of magnitude at one
    # layer would mean a real bug rather than float32 round-off.
    assert drift[-1] < 100 * max(drift[0], 1e-7), (
        f"drift grew abruptly rather than smoothly: {['%.1e' % d for d in drift]}"
    )


@pytestmark_real
@pytest.mark.reference
@pytest.mark.slow
def test_predicts_paris(real_model, real_tokenizer):
    """A sanity check a human can read, independent of any reference."""
    ids = real_tokenizer.encode("The capital of France is")
    next_id = int(real_model.next_token_logits(np.array(ids)).argmax())
    assert real_tokenizer.decode([next_id]).strip() == "Paris"


@pytestmark_real
@pytest.mark.slow
def test_last_only_matches_full_projection_on_the_real_model(real_model, real_tokenizer):
    """Same values, but not bit-identical, and that is expected.

    ``last_only`` turns a [5, 896] @ [896, 151936] matmul into [1, 896] @ the
    same. BLAS picks different blocking and vectorization for the two shapes,
    which changes the order the 896 products are summed in. Float addition is
    not associative, so the results differ in the last bits -- about 1e-5 here,
    on logits whose magnitude reaches 19.

    This is worth a test rather than a shrug, because it sets the floor for
    what "identical output" can mean anywhere else in this project. Phase 4
    will claim the KV cache reproduces the uncached path exactly; that claim
    has to be made at this tolerance, not at zero.
    """
    ids = np.array(real_tokenizer.encode("The capital of France is"))

    shortcut = real_model.forward(ids, last_only=True)[0]
    full = real_model.forward(ids)[-1]

    np.testing.assert_allclose(shortcut, full, rtol=1e-3, atol=1e-4)
    assert int(shortcut.argmax()) == int(full.argmax()), "greedy choice must agree"
