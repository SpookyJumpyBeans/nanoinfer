"""Tests for the KV cache container, before it is wired into anything.

The cache is where the brief says most bugs live, and the reason is that its
failures are all off-by-one or stale-data problems that produce plausible
output. So the container is tested on its own -- watermark arithmetic, slot
placement, capacity, misuse -- with no attention involved, before any of it is
connected to the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoinfer.config import ModelConfig
from nanoinfer.kvcache import CacheFullError, CacheUsageError, KVCache
from tests.tiny import TINY_CONFIG, build_tiny_model


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> ModelConfig:
    return ModelConfig.from_model_dir(build_tiny_model(tmp_path_factory.mktemp("kv")))


@pytest.fixture
def cache(config) -> KVCache:
    return KVCache(config, capacity=16)


def kv(config, n_new: int, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A distinctly-valued key/value pair, so misplacement is visible."""
    shape = (config.num_key_value_heads, n_new, config.head_dim)
    keys = np.full(shape, fill, dtype=np.float32)
    values = np.full(shape, fill + 100, dtype=np.float32)
    return keys, values


def write_all_layers(cache: KVCache, config: ModelConfig, n_new: int, fill: float):
    """Drive one complete step the way the model does."""
    keys, values = kv(config, n_new, fill)
    for layer in range(config.num_hidden_layers):
        cache.extend(layer, keys, values)
    cache.commit(n_new)


# -- allocation ------------------------------------------------------------


def test_starts_empty(cache):
    assert cache.length == 0
    assert cache.remaining == cache.capacity


def test_capacity_must_be_positive(config):
    with pytest.raises(ValueError, match="positive"):
        KVCache(config, capacity=0)


def test_reports_its_footprint(config):
    cache = KVCache(config, capacity=100)
    expected = (
        2  # keys and values
        * config.num_hidden_layers
        * config.num_key_value_heads
        * 100
        * config.head_dim
        * 4  # float32
    )
    assert cache.nbytes == expected


def test_bytes_per_token_matches_the_config_estimate(config):
    """The same number config.kv_bytes_per_token predicts, independently."""
    cache = KVCache(config, capacity=64)
    assert cache.bytes_per_token == config.kv_bytes_per_token


def test_grouped_query_attention_is_what_makes_this_affordable(config):
    """State the win explicitly: the cache scales with KV heads, not query heads."""
    cache = KVCache(config, capacity=64)
    without_gqa = cache.bytes_per_token * config.kv_group_size
    assert without_gqa > cache.bytes_per_token
    assert config.kv_group_size > 1


# -- the watermark ---------------------------------------------------------


def test_extend_alone_does_not_advance_the_length(cache, config):
    """Every layer writes the same tokens, so only commit() may move it."""
    keys, values = kv(config, 3)
    cache.extend(0, keys, values)
    assert cache.length == 0


def test_commit_advances_by_the_token_count(cache, config):
    write_all_layers(cache, config, 3, fill=1.0)
    assert cache.length == 3
    assert cache.remaining == 13


def test_successive_steps_accumulate(cache, config):
    write_all_layers(cache, config, 4, fill=1.0)
    write_all_layers(cache, config, 1, fill=2.0)
    write_all_layers(cache, config, 1, fill=3.0)
    assert cache.length == 6


def test_committing_zero_tokens_is_a_no_op(cache):
    cache.commit(0)
    assert cache.length == 0


def test_negative_commit_is_rejected(cache):
    with pytest.raises(ValueError, match="non-negative"):
        cache.commit(-1)


# -- where the data lands --------------------------------------------------


def test_new_tokens_land_after_the_existing_ones(cache, config):
    """The off-by-one that would overwrite the prompt with the first output."""
    write_all_layers(cache, config, 3, fill=7.0)
    write_all_layers(cache, config, 1, fill=9.0)

    keys = cache.keys(0)
    assert keys.shape == (config.num_key_value_heads, 4, config.head_dim)
    assert np.all(keys[:, :3, :] == 7.0), "the prompt was overwritten"
    assert np.all(keys[:, 3, :] == 9.0), "the new token is in the wrong slot"


def test_extend_returns_everything_including_the_new_tokens(cache, config):
    write_all_layers(cache, config, 2, fill=5.0)

    keys_new, values_new = kv(config, 1, fill=6.0)
    keys_all, values_all = cache.extend(0, keys_new, values_new)

    assert keys_all.shape[1] == 3
    assert np.all(keys_all[:, :2, :] == 5.0)
    assert np.all(keys_all[:, 2, :] == 6.0)
    assert np.all(values_all[:, 2, :] == 106.0)


def test_returned_arrays_are_views_not_copies(cache, config):
    """Zero-copy is the entire point; a copy here would undo the speedup."""
    keys_new, values_new = kv(config, 1, fill=1.0)
    keys_all, _ = cache.extend(0, keys_new, values_new)
    assert keys_all.base is not None


def test_layers_do_not_share_storage(cache, config):
    """A layer-index bug would have every layer attend over layer 0's keys."""
    for layer in range(config.num_hidden_layers):
        keys, values = kv(config, 1, fill=float(layer))
        cache.extend(layer, keys, values)
    cache.commit(1)

    for layer in range(config.num_hidden_layers):
        assert np.all(cache.keys(layer) == float(layer)), f"layer {layer} sees wrong keys"


def test_keys_and_values_do_not_share_storage(cache, config):
    write_all_layers(cache, config, 2, fill=1.0)
    assert np.all(cache.keys(0) == 1.0)
    assert np.all(cache.values(0) == 101.0)


def test_inspection_views_stop_at_the_watermark(cache, config):
    """Nothing above the watermark may be visible, zeroed or not."""
    write_all_layers(cache, config, 3, fill=1.0)
    assert cache.keys(0).shape[1] == 3


# -- capacity --------------------------------------------------------------


def test_overflow_is_reported_clearly(cache, config):
    write_all_layers(cache, config, 14, fill=1.0)
    keys, values = kv(config, 5)
    with pytest.raises(CacheFullError, match="cannot add 5 tokens"):
        cache.extend(0, keys, values)


def test_filling_exactly_to_capacity_is_allowed(cache, config):
    write_all_layers(cache, config, 16, fill=1.0)
    assert cache.length == 16
    assert cache.remaining == 0


def test_one_past_capacity_fails(cache, config):
    write_all_layers(cache, config, 16, fill=1.0)
    keys, values = kv(config, 1)
    with pytest.raises(CacheFullError):
        cache.extend(0, keys, values)


# -- misuse ----------------------------------------------------------------


def test_a_layer_writing_twice_is_rejected(cache, config):
    """Both writes target the same slots, so the first would be lost."""
    keys, values = kv(config, 1)
    cache.extend(0, keys, values)
    with pytest.raises(CacheUsageError, match="wrote twice"):
        cache.extend(0, keys, values)


def test_committing_with_a_layer_missing_is_rejected(cache, config):
    """A layer that skipped the cache would attend over stale keys next step."""
    keys, values = kv(config, 1)
    for layer in range(config.num_hidden_layers - 1):
        cache.extend(layer, keys, values)

    with pytest.raises(CacheUsageError, match="did not write"):
        cache.commit(1)


def test_the_missing_layer_is_named(cache, config):
    keys, values = kv(config, 1)
    for layer in range(config.num_hidden_layers):
        if layer != 1:
            cache.extend(layer, keys, values)
    with pytest.raises(CacheUsageError, match=r"layers \[1\]"):
        cache.commit(1)


def test_out_of_range_layer_is_rejected(cache, config):
    keys, values = kv(config, 1)
    with pytest.raises(IndexError, match="out of range"):
        cache.extend(config.num_hidden_layers, keys, values)


def test_wrong_head_count_is_rejected(cache, config):
    bad = np.zeros((config.num_attention_heads, 1, config.head_dim), dtype=np.float32)
    with pytest.raises(ValueError, match="expected keys and values shaped"):
        cache.extend(0, bad, bad)


def test_wrong_head_dim_is_rejected(cache, config):
    bad = np.zeros((config.num_key_value_heads, 1, config.head_dim + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="expected keys and values shaped"):
        cache.extend(0, bad, bad)


def test_mismatched_keys_and_values_are_rejected(cache, config):
    keys, _ = kv(config, 2)
    _, values = kv(config, 3)
    with pytest.raises(ValueError):
        cache.extend(0, keys, values)


# -- reset -----------------------------------------------------------------


def test_reset_clears_the_watermark(cache, config):
    write_all_layers(cache, config, 5, fill=1.0)
    cache.reset()
    assert cache.length == 0
    assert cache.remaining == cache.capacity


def test_reset_allows_reuse_without_reallocating(cache, config):
    before = cache.nbytes
    write_all_layers(cache, config, 5, fill=1.0)
    cache.reset()
    write_all_layers(cache, config, 2, fill=9.0)
    assert cache.length == 2
    assert cache.nbytes == before
    assert np.all(cache.keys(0) == 9.0), "stale data survived the reset"


def test_reset_clears_pending_writes(cache, config):
    keys, values = kv(config, 1)
    cache.extend(0, keys, values)
    cache.reset()
    cache.extend(0, keys, values)  # must not raise "wrote twice"


def test_repr_is_informative(cache, config):
    write_all_layers(cache, config, 3, fill=1.0)
    text = repr(cache)
    assert "3/16" in text
    assert "KiB per token" in text
