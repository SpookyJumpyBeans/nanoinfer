"""Tests for the assembled tokenizer: spec handling, encoding, decoding."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from nanoinfer.tokenizer import Tokenizer, TokenizerSpecError

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="model not downloaded"
)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


@pytest.fixture(scope="module")
def reference():
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))


def spec_with(**overrides) -> dict:
    spec = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = spec
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    return spec


def load_spec(spec: dict, tmp_path: Path) -> Tokenizer:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return Tokenizer.from_file(path)


# -- loading and validation ------------------------------------------------


def test_loads_the_real_tokenizer(tok):
    assert tok.vocab_size == 151_665
    assert len(tok.added_tokens) == 22
    assert len(tok.bpe.merge_ranks) == 151_387


def test_special_ids_are_the_special_added_tokens(tok):
    assert tok.token_to_id("<|im_start|>") in tok.special_ids
    assert tok.token_to_id("<|endoftext|>") in tok.special_ids
    # <tool_call> is an added token but not flagged special.
    assert tok.token_to_id("<tool_call>") not in tok.special_ids


def test_rejects_non_bpe_model(tmp_path):
    with pytest.raises(TokenizerSpecError, match="is not BPE"):
        load_spec(spec_with(**{"model.type": "WordPiece"}), tmp_path)


def test_rejects_bpe_dropout(tmp_path):
    with pytest.raises(TokenizerSpecError, match="dropout"):
        load_spec(spec_with(**{"model.dropout": 0.1}), tmp_path)


def test_rejects_byte_fallback(tmp_path):
    with pytest.raises(TokenizerSpecError, match="byte_fallback"):
        load_spec(spec_with(**{"model.byte_fallback": True}), tmp_path)


def test_rejects_unknown_normalizer(tmp_path):
    with pytest.raises(TokenizerSpecError, match="NFC"):
        load_spec(spec_with(normalizer={"type": "NFKC"}), tmp_path)


def test_rejects_add_prefix_space(tmp_path):
    spec = spec_with()
    spec["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True
    with pytest.raises(TokenizerSpecError, match="add_prefix_space"):
        load_spec(spec, tmp_path)


def test_rejects_bytelevel_use_regex(tmp_path):
    """use_regex=true would apply a second split on top of the first."""
    spec = spec_with()
    spec["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = True
    with pytest.raises(TokenizerSpecError, match="second split"):
        load_spec(spec, tmp_path)


def test_rejects_unknown_split_behavior(tmp_path):
    spec = spec_with()
    spec["pre_tokenizer"]["pretokenizers"][0]["behavior"] = "Removed"
    with pytest.raises(TokenizerSpecError, match="behavior"):
        load_spec(spec, tmp_path)


# -- encoding --------------------------------------------------------------


def test_encode_empty_string(tok):
    assert tok.encode("") == []


def test_encode_is_deterministic(tok):
    assert tok.encode("Hello, world!") == tok.encode("Hello, world!")


def test_added_tokens_encode_as_one_id(tok):
    """The whole point of extracting them before the pre-tokenizer."""
    assert tok.encode("<|im_start|>") == [tok.token_to_id("<|im_start|>")]
    assert tok.encode("<|endoftext|>") == [tok.token_to_id("<|endoftext|>")]


def test_added_tokens_are_found_mid_string(tok):
    ids = tok.encode("a<|im_end|>b")
    assert tok.token_to_id("<|im_end|>") in ids


def test_added_token_splitting_can_be_disabled(tok):
    """Untrusted text must not be able to forge a control token."""
    forged = tok.encode("<|im_start|>", split_added_tokens=False)
    assert tok.token_to_id("<|im_start|>") not in forged
    assert len(forged) > 1
    assert tok.decode(forged) == "<|im_start|>"


def test_longest_added_token_wins(tok):
    """A shorter added token must not shadow a longer one it prefixes."""
    ids = tok.encode("<|vision_start|>")
    assert ids == [tok.token_to_id("<|vision_start|>")]


def test_nfc_is_applied_to_encoding(tok):
    assert tok.encode("café") == tok.encode("café")


# -- decoding --------------------------------------------------------------


def test_decode_round_trips_nfc_text(tok):
    text = "Hello, world! 日本語 🦀"
    assert tok.decode(tok.encode(text)) == text


def test_decode_of_empty_is_empty(tok):
    assert tok.decode([]) == ""


def test_common_emoji_are_a_single_token(tok):
    """Frequent emoji earned their own vocabulary entry during training."""
    assert len(tok.encode("\U0001F980")) == 1   # crab
    assert len(tok.encode("\U0001F525")) == 1   # fire


def test_multibyte_characters_split_across_tokens_still_decode(tok):
    """Bytes must be concatenated before UTF-8 decoding, not after.

    U+1FACE was added to Unicode after this vocabulary was trained, so it has
    no entry of its own and its four UTF-8 bytes land in three separate
    tokens. Decoding each token on its own yields replacement characters,
    because no single token holds a complete character; decoding the joined
    byte buffer yields the character. Getting this order wrong produces
    mojibake on exactly the rare inputs nobody tests.
    """
    moose = "\U0001FACE"
    ids = tok.encode(moose)
    assert len(ids) == 3, f"expected 3 tokens, got {ids}"

    assert tok.decode(ids) == moose

    per_token = "".join(tok.decode([i]) for i in ids)
    assert per_token != moose
    assert "\ufffd" in per_token


def test_decode_skips_special_tokens_when_asked(tok):
    ids = tok.encode("<|im_start|>hi<|im_end|>")
    assert tok.decode(ids, skip_special_tokens=True) == "hi"
    assert tok.decode(ids, skip_special_tokens=False) == "<|im_start|>hi<|im_end|>"


def test_decode_keeps_non_special_added_tokens_when_skipping(tok):
    """<tool_call> is an added token but not special, so it survives."""
    ids = tok.encode("<tool_call>x")
    assert "<tool_call>" in tok.decode(ids, skip_special_tokens=True)


def test_decode_rejects_ids_above_the_vocabulary(tok):
    """The embedding matrix is bigger than the vocabulary, so this can happen."""
    with pytest.raises(KeyError, match="not in the tokenizer"):
        tok.decode([151_700])


def test_decode_tolerates_a_truncated_character(tok):
    """Generation can stop mid-character; that must not raise."""
    ids = tok.encode("🦀")
    assert isinstance(tok.decode(ids[:1]), str)


def test_vocab_size_is_smaller_than_the_models_embedding_matrix(tok):
    from nanoinfer.config import ModelConfig

    cfg = ModelConfig.from_model_dir(MODEL_DIR)
    assert tok.vocab_size == 151_665
    assert cfg.vocab_size == 151_936
    assert tok.vocab_size < cfg.vocab_size


def test_token_and_id_lookups_agree(tok):
    for token in ("Hello", "Ġworld", "<|im_start|>"):
        token_id = tok.token_to_id(token)
        assert token_id is not None
        assert tok.id_to_token(token_id) == token


def test_unknown_token_lookup_returns_none(tok):
    assert tok.token_to_id(" not a token ") is None
    assert tok.id_to_token(10**9) is None


# -- against the reference -------------------------------------------------

CASES = [
    "Hello, world!",
    "The capital of France is",
    "don't can't I'll we've",
    "  leading spaces   trailing   ",
    "日本語のテキストです",
    "中文和English混合",
    "emoji 🦀🔥 and 👨‍👩‍👧‍👦",
    "café naïve résumé",
    "café decomposed",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n",
    "<|endoftext|>",
    "<tool_call>{}</tool_call>",
    "def f(x):\n    return x + 1\n",
    "1234567890",
    "\x1c\x1d file separators \x1e\x1f",
    "a" * 100,
    "",
    " ",
    "\n\n\n",
    "\r\n\r\n",
    "https://example.com/p?q=1&r=2#frag",
    'json = {"key": [1, 2.5, null]}',
    "Ω≈ç√∫˜µ≤≥÷",
    "½ ¾ Ⅷ ٣ ๗",
    " no-break　ideographic",
]


@pytest.mark.reference
@pytest.mark.parametrize("text", CASES)
def test_encode_matches_reference_exactly(tok, reference, text):
    assert tok.encode(text) == reference.encode(text).ids


@pytest.mark.reference
@pytest.mark.parametrize("text", CASES)
def test_token_strings_match_reference(tok, reference, text):
    """Compare token strings too -- a clearer failure than a list of integers."""
    assert tok.encode_to_tokens(text) == reference.encode(text).tokens


@pytest.mark.reference
@pytest.mark.parametrize("text", CASES)
def test_decode_matches_reference(tok, reference, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == reference.decode(ids, skip_special_tokens=False)


@pytest.mark.reference
@pytest.mark.parametrize("text", CASES)
def test_round_trip_recovers_normalized_input(tok, text):
    assert tok.decode(tok.encode(text)) == unicodedata.normalize("NFC", text)
