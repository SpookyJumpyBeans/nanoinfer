"""Tests for the pre-tokenizer, including a direct diff against the reference.

BPE merges never cross a pre-token boundary, so if the split is wrong the IDs
are wrong no matter how correct the merge algorithm is. These tests check the
split in isolation, before BPE is in the picture, because a mismatch found here
is obvious and a mismatch found after BPE is a needle in a haystack.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from nanoinfer.pretokenize import (
    PatternTranslationError,
    PreTokenizer,
    translate_pattern,
)
from nanoinfer.bytelevel import bytes_to_unicode

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="model not downloaded"
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_pattern(spec) -> str:
    return spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]


@pytest.fixture(scope="module")
def pretokenizer(raw_pattern) -> PreTokenizer:
    return PreTokenizer.from_pattern(raw_pattern)


@pytest.fixture(scope="module")
def reference():
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))


# Inputs chosen to hit the parts of the pattern that are easy to get wrong:
# the contraction alternation, the optional leading space, CRLF handling, the
# whitespace-before-non-whitespace lookahead, and non-Latin scripts.
CASES = [
    "",
    " ",
    "a",
    "Hello, world!",
    "  leading and   multiple   spaces",
    "trailing spaces   ",
    "don't can't I'll we've they're it's",
    "DON'T CAN'T I'LL",          # the alternation is case-insensitive
    "line1\nline2\r\nline3",
    "\n\n\n",
    "\r\n\r\n",
    "tabs\there\tand\tthere",
    "numbers 12345 and 007 and 3.14",
    "mixed123abc456",
    "1234567890" * 5,
    "snake_case camelCase kebab-case",
    "日本語のテキストです",
    "中文和English混合",
    "emoji 🦀🔥 mixed with text",
    "👨‍👩‍👧‍👦 zwj family",
    "café naïve résumé",
    "Ω≈ç√∫˜µ≤≥÷",
    "½ ¾ Ⅷ ٣ ๗",
    " no-break　ideographic",
    "\x1c\x1d file separators \x1e\x1f",
    "<|im_start|>user\nhi<|im_end|>",
    "a" * 50,
    "!!!???...,,,",
    "   \n   ",
    "x  ",
    "def f(x):\n    return x + 1\n",
    'json = {"key": [1, 2.5, null], "b": true}',
    "https://example.com/path?q=1&r=2#frag",
]


# -- pattern translation ---------------------------------------------------


def test_translate_expands_property_outside_a_class():
    out = translate_pattern(r"\p{L}+")
    assert out.startswith("[") and out.endswith("]+")
    assert re.compile(out).fullmatch("abc")


def test_translate_inlines_property_inside_a_class():
    """Inside [...] the body must be inlined or the brackets would nest."""
    out = translate_pattern(r"[^\r\n\p{L}]")
    assert out.count("[") == 1, out
    pattern = re.compile(out)
    assert pattern.fullmatch("!")
    assert not pattern.fullmatch("a")
    assert not pattern.fullmatch("\n")


def test_translate_rewrites_whitespace_to_the_unicode_property():
    """\\s must become the White_Space class, not stay as Python's \\s."""
    out = translate_pattern(r"\s")
    assert "\\s" not in out
    assert re.compile(out).fullmatch(" ")
    assert not re.compile(out).fullmatch("\x1c")


def test_translate_handles_negated_whitespace():
    out = translate_pattern(r"\S")
    assert re.compile(out).fullmatch("a")
    assert not re.compile(out).fullmatch(" ")


def test_translate_passes_other_escapes_through():
    assert translate_pattern(r"\r\n\.") == r"\r\n\."


def test_translate_preserves_inline_flag_groups():
    out = translate_pattern(r"(?i:'s|'t)")
    assert re.compile(out).fullmatch("'S")


def test_translate_rejects_unknown_property():
    with pytest.raises(PatternTranslationError, match="not implemented"):
        translate_pattern(r"\p{Greek}")


def test_translate_rejects_unterminated_property():
    with pytest.raises(PatternTranslationError, match="unterminated"):
        translate_pattern(r"\p{L")


def test_translate_rejects_unterminated_class():
    with pytest.raises(PatternTranslationError, match="unterminated character class"):
        translate_pattern(r"[abc")


def test_translate_rejects_negated_whitespace_inside_a_class():
    with pytest.raises(PatternTranslationError, match="inside a character class"):
        translate_pattern(r"[\S]")


def test_real_pattern_translates_and_compiles(raw_pattern):
    compiled = re.compile(translate_pattern(raw_pattern))
    assert compiled.search("hello")


# -- splitting behaviour ---------------------------------------------------


def test_leading_space_attaches_to_the_following_word(pretokenizer):
    """' ?[^...]' and the optional prefix are why ' world' is one token."""
    assert pretokenizer.split("Hello world") == ["Hello", " world"]


def test_contractions_split_off(pretokenizer):
    assert pretokenizer.split("don't") == ["don", "'t"]


def test_runs_of_digits_split_individually(pretokenizer):
    """\\p{N} has no '+', so each digit is its own pre-token."""
    assert pretokenizer.split("123") == ["1", "2", "3"]


def test_trailing_whitespace_is_kept(pretokenizer):
    assert pretokenizer.split("hi   ") == ["hi", "   "]


def test_empty_input(pretokenizer):
    assert pretokenizer.split("") == []


def test_nfc_normalization_is_applied(pretokenizer):
    """Decomposed and composed forms must produce identical pre-tokens."""
    composed = "café"
    decomposed = "café"
    assert composed != decomposed
    assert pretokenizer.split(decomposed) == pretokenizer.split(composed)


def test_normalization_can_be_disabled(raw_pattern):
    pt = PreTokenizer.from_pattern(raw_pattern, normalize=False)
    assert pt.split("café") != pt.split("café")


def test_encode_returns_byte_level_strings(pretokenizer):
    assert pretokenizer.encode("Hello world") == ["Hello", "Ġworld"]


def test_encode_handles_non_ascii(pretokenizer):
    """Every piece must be spelled in the byte-level alphabet."""
    allowed = set(bytes_to_unicode().values())
    for piece in pretokenizer.encode("日本語 🦀 café"):
        assert set(piece) <= allowed


@pytest.mark.parametrize("text", CASES)
def test_pattern_covers_input_without_gaps(pretokenizer, text):
    """The pattern is meant to match all input; assert it actually does."""
    assert pretokenizer.covers_without_gaps(text)


@pytest.mark.parametrize("text", CASES)
def test_split_is_lossless(pretokenizer, text):
    """Concatenating the pieces must rebuild the normalized input exactly."""
    assert "".join(pretokenizer.split(text)) == unicodedata.normalize("NFC", text)


# -- against the reference -------------------------------------------------


@pytest.mark.reference
@pytest.mark.parametrize("text", CASES)
def test_matches_reference_pretokenizer(pretokenizer, reference, text):
    """Diff against HuggingFace's Rust pre-tokenizer, piece for piece.

    ``pre_tokenize_str`` runs only the pre-tokenizer stage and skips the
    declared NFC normalizer, so the input is normalized here first to compare
    the same stage on both sides.
    """
    normalized = unicodedata.normalize("NFC", text)
    mine = pretokenizer.encode(normalized)
    theirs = [piece for piece, _ in reference.pre_tokenizer.pre_tokenize_str(normalized)]
    assert mine == theirs
