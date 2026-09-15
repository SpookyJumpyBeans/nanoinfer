"""Tests for the byte-level alphabet.

The alphabet is a fixed table baked into every published GPT-2-family
vocabulary, so these tests are mostly about pinning exact values. If any of
them drift, every token in the vocabulary decodes to the wrong bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoinfer.bytelevel import (
    bytes_to_unicode,
    decode_bytes,
    encode_bytes,
    unicode_to_bytes,
)

TOKENIZER_JSON = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "Qwen2.5-0.5B-Instruct"
    / "tokenizer.json"
)


def test_alphabet_covers_every_byte_exactly_once():
    table = bytes_to_unicode()
    assert len(table) == 256
    assert set(table) == set(range(256))
    assert len(set(table.values())) == 256, "mapping must be injective"


def test_alphabet_is_a_bijection():
    forward = bytes_to_unicode()
    back = unicode_to_bytes()
    assert len(back) == 256
    for byte, ch in forward.items():
        assert back[ch] == byte


def test_printable_ascii_maps_to_itself():
    """ASCII stays readable in the vocab file; that is the whole design goal."""
    table = bytes_to_unicode()
    for byte in range(ord("!"), ord("~") + 1):
        assert table[byte] == chr(byte)


def test_known_escape_values():
    """The escapes everyone recognises from a GPT-2 vocabulary dump."""
    table = bytes_to_unicode()
    assert table[0x20] == "Ġ"   # space -> G with dot above
    assert table[0x0A] == "Ċ"   # newline -> C with dot above
    assert table[0x09] == "ĉ"   # tab
    assert table[0x00] == "Ā"   # first escape, NUL
    assert table[0x0D] == "č"   # carriage return


def test_escapes_occupy_exactly_u0100_to_u0143():
    """68 bytes need escaping, so the escape block is 0x100..0x143 inclusive."""
    table = bytes_to_unicode()
    escaped = sorted(ord(c) for c in table.values() if ord(c) >= 0x100)
    assert len(escaped) == 68
    assert escaped == list(range(0x100, 0x144))


def test_no_alphabet_character_is_whitespace():
    """Nothing in the alphabet may be whitespace.

    If it were, the space-separated merges file would be ambiguous -- which is
    the reason the mapping exists at all.
    """
    for ch in bytes_to_unicode().values():
        assert not ch.isspace(), f"{ch!r} is whitespace"


def test_round_trip_every_single_byte():
    for byte in range(256):
        assert decode_bytes(encode_bytes(bytes([byte]))) == bytes([byte])


def test_round_trip_all_byte_values_at_once():
    data = bytes(range(256))
    assert decode_bytes(encode_bytes(data)) == data


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "  leading spaces",
        "tabs\tand\nnewlines\r\n",
        "héllo wörld",
        "日本語のテキスト",
        "emoji 🦀 and 👨‍👩‍👧‍👦 zwj sequences",
        "mixed ünïcödé \x00\x01\x7f control bytes",
    ],
)
def test_round_trip_utf8_text(text: str):
    data = text.encode("utf-8")
    assert decode_bytes(encode_bytes(data)) == data


def test_encode_of_space_is_the_expected_marker():
    assert encode_bytes(b" hello") == "Ġhello"


def test_empty_input():
    assert encode_bytes(b"") == ""
    assert decode_bytes("") == b""


def test_decode_rejects_characters_outside_the_alphabet():
    with pytest.raises(KeyError, match="not in the byte-level alphabet"):
        decode_bytes("日")


def test_table_is_cached_not_rebuilt():
    assert bytes_to_unicode() is bytes_to_unicode()


@pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="model not downloaded")
def test_every_vocab_token_is_spelled_in_this_alphabet():
    """The strongest available check: run it against the real 151,643 tokens.

    If our alphabet were wrong in even one codepoint, some real vocabulary
    token would contain a character we cannot decode.
    """
    vocab = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))["model"]["vocab"]
    alphabet = set(bytes_to_unicode().values())

    for token in vocab:
        stray = set(token) - alphabet
        assert not stray, f"token {token!r} uses characters outside the alphabet: {stray}"

    # And every token must decode back to bytes without raising.
    for token in vocab:
        decode_bytes(token)
