"""Tests for loading weights into the structure the forward pass uses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoinfer.weights import ModelWeights, WeightShapeError
from tests.tiny import TINY_CONFIG, build_tiny_model, tiny_tensors

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> ModelWeights:
    directory = build_tiny_model(tmp_path_factory.mktemp("tiny"))
    return ModelWeights.load(directory)


# -- structure -------------------------------------------------------------


def test_loads_every_layer(tiny):
    assert len(tiny.layers) == TINY_CONFIG["num_hidden_layers"]


def test_shapes_follow_the_config(tiny):
    cfg = tiny.config
    hidden = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    layer = tiny.layers[0]
    assert layer.q_proj_weight.shape == (q_dim, hidden)
    assert layer.k_proj_weight.shape == (kv_dim, hidden)
    assert layer.v_proj_weight.shape == (kv_dim, hidden)
    assert layer.o_proj_weight.shape == (hidden, q_dim)
    assert layer.gate_proj_weight.shape == (cfg.intermediate_size, hidden)
    assert layer.down_proj_weight.shape == (hidden, cfg.intermediate_size)
    assert layer.input_layernorm.shape == (hidden,)


def test_kv_projections_are_narrower_than_q(tiny):
    """The visible consequence of grouped-query attention."""
    assert tiny.layers[0].k_proj_weight.shape[0] < tiny.layers[0].q_proj_weight.shape[0]


def test_everything_is_float32(tiny):
    assert tiny.embed_tokens.dtype == np.float32
    assert tiny.final_norm.dtype == np.float32
    for layer in tiny.layers:
        assert layer.q_proj_weight.dtype == np.float32
        assert layer.q_proj_bias.dtype == np.float32


def test_layers_hold_distinct_weights(tiny):
    """A loop bug that loads layer 0 twice would still produce fluent text."""
    assert not np.array_equal(
        tiny.layers[0].q_proj_weight, tiny.layers[1].q_proj_weight
    )


def test_nbytes_accounts_for_everything(tiny):
    counted = tiny.embed_tokens.nbytes + tiny.final_norm.nbytes
    counted += sum(layer.nbytes for layer in tiny.layers)
    assert tiny.nbytes == counted


# -- tied embeddings -------------------------------------------------------


def test_tied_lm_head_is_the_embedding_matrix(tiny):
    assert tiny.tied
    assert tiny.lm_head is tiny.embed_tokens


def test_untied_model_loads_a_separate_lm_head(tmp_path):
    config = dict(TINY_CONFIG, tie_word_embeddings=False)
    tensors = tiny_tensors(config)
    directory = build_tiny_model(
        tmp_path / "untied",
        config_overrides={"tie_word_embeddings": False},
        tensor_overrides={"lm_head.weight": tensors["lm_head.weight"]},
    )
    weights = ModelWeights.load(directory)

    assert not weights.tied
    assert weights.lm_head is not weights.embed_tokens
    assert weights.lm_head.shape == weights.embed_tokens.shape


def test_untied_model_without_lm_head_is_rejected(tmp_path):
    directory = build_tiny_model(
        tmp_path / "broken",
        config_overrides={"tie_word_embeddings": False},
        tensor_overrides={"lm_head.weight": None},
    )
    with pytest.raises(WeightShapeError, match="missing tensor 'lm_head.weight'"):
        ModelWeights.load(directory)


def test_contradictory_tying_is_rejected(tmp_path):
    """tie_word_embeddings=true *and* an lm_head tensor: refuse to guess."""
    extra = tiny_tensors(dict(TINY_CONFIG, tie_word_embeddings=False))["lm_head.weight"]
    directory = build_tiny_model(
        tmp_path / "both", tensor_overrides={"lm_head.weight": extra}
    )
    with pytest.raises(WeightShapeError, match="refusing to guess"):
        ModelWeights.load(directory)


# -- error paths -----------------------------------------------------------


def test_missing_tensor_is_reported_by_name(tmp_path):
    directory = build_tiny_model(
        tmp_path / "missing",
        tensor_overrides={"model.layers.1.mlp.up_proj.weight": None},
    )
    with pytest.raises(WeightShapeError, match="model.layers.1.mlp.up_proj.weight"):
        ModelWeights.load(directory)


def test_wrong_shape_is_reported_with_both_shapes(tmp_path):
    from nanoinfer.safetensors import f32_to_bf16

    directory = build_tiny_model(
        tmp_path / "wrongshape",
        tensor_overrides={
            "model.layers.0.self_attn.q_proj.weight": f32_to_bf16(
                np.zeros((8, 16), dtype=np.float32)
            )
        },
    )
    with pytest.raises(WeightShapeError, match=r"expected shape \(16, 16\), file has \(8, 16\)"):
        ModelWeights.load(directory)


def test_missing_weight_file_is_reported(tmp_path):
    """A valid config with no weights beside it, which is a real failure mode."""
    directory = build_tiny_model(tmp_path / "noweights")
    (directory / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        ModelWeights.load(directory)


def test_weights_survive_the_file_being_closed(tiny):
    """load() closes the mapping, so the arrays must be copies, not views."""
    assert tiny.embed_tokens.flags.owndata or tiny.embed_tokens.base is not None
    # Touching every value would segfault if these were dangling mmap views.
    assert np.isfinite(tiny.embed_tokens).all()
    for layer in tiny.layers:
        assert np.isfinite(layer.down_proj_weight).all()


# -- the real model --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(), reason="model not downloaded"
)
def test_real_model_loads_with_the_expected_budget():
    weights = ModelWeights.load(MODEL_DIR)

    assert len(weights.layers) == 24
    assert weights.tied
    assert weights.embed_tokens.shape == (151_936, 896)
    assert weights.layers[0].q_proj_weight.shape == (896, 896)
    assert weights.layers[0].k_proj_weight.shape == (128, 896)

    # 494M parameters at 4 bytes each.
    assert 1.9e9 < weights.nbytes < 2.1e9
