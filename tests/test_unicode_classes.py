"""Tests for the hand-built Unicode property classes."""

from __future__ import annotations

import re
import sys
import unicodedata

import pytest

from nanoinfer.unicode_classes import (
    WHITESPACE_RANGES,
    category_ranges,
    class_body,
    is_whitespace,
    letter_class_body,
    number_class_body,
    ranges_to_class_body,
    whitespace_class_body,
)


@pytest.fixture(scope="module")
def letter_re():
    return re.compile(f"[{letter_class_body()}]")


@pytest.fixture(scope="module")
def number_re():
    return re.compile(f"[{number_class_body()}]")


@pytest.fixture(scope="module")
def space_re():
    return re.compile(f"[{whitespace_class_body()}]")


# -- letters ---------------------------------------------------------------


@pytest.mark.parametrize(
    "ch",
    ["a", "Z", "é", "ß", "日", "语", "א", "ا", "Ω", "ъ", "ᚠ", "ｱ", "ᵃ", "ʰ"],
)
def test_letters_match(letter_re, ch):
    assert letter_re.fullmatch(ch), f"{ch!r} ({unicodedata.category(ch)}) should be a letter"


@pytest.mark.parametrize("ch", ["1", " ", "!", "\n", "½", "€", "🦀", "́"])
def test_non_letters_do_not_match(letter_re, ch):
    assert not letter_re.fullmatch(ch), f"{ch!r} ({unicodedata.category(ch)}) is not a letter"


def test_letter_class_covers_every_L_category():
    """Lu, Ll, Lt, Lm and Lo must all be included -- \\p{L} is all five."""
    seen = set()
    for start, end in category_ranges("L"):
        for cp in (start, end):
            seen.add(unicodedata.category(chr(cp)))
    assert {"Lu", "Ll", "Lo"} <= seen


# -- numbers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "ch",
    ["0", "9", "٣", "๗", "½", "Ⅷ", "〇", "３"],
)
def test_numbers_match(number_re, ch):
    assert number_re.fullmatch(ch), f"{ch!r} ({unicodedata.category(ch)}) should be a number"


@pytest.mark.parametrize("ch", ["a", " ", "+", "日", "!"])
def test_non_numbers_do_not_match(number_re, ch):
    assert not number_re.fullmatch(ch)


def test_number_class_includes_non_decimal_categories():
    """\\p{N} is Nd plus Nl plus No, not just the ASCII-style digits."""
    body = number_class_body()
    pattern = re.compile(f"[{body}]")
    assert pattern.fullmatch("½")   # No
    assert pattern.fullmatch("Ⅷ")   # Nl
    assert pattern.fullmatch("٣")   # Nd


# -- whitespace, and the Python/Rust divergence ----------------------------


# Written as escapes on purpose: several of these are visually identical to a
# plain space, and a test whose inputs cannot be told apart by eye is a test
# nobody can review.
@pytest.mark.parametrize(
    "ch",
    [
        "\t",       # tab
        "\n",       # line feed
        "\r",       # carriage return
        "\v",       # vertical tab
        "\f",       # form feed
        "\x20",     # space
        "\x85",     # next line
        "\xa0",     # no-break space
        "\u1680",   # ogham space mark
        "\u2000",   # en quad
        "\u200a",   # hair space
        "\u2028",   # line separator
        "\u2029",   # paragraph separator
        "\u202f",   # narrow no-break space
        "\u205f",   # medium mathematical space
        "\u3000",   # ideographic space
    ],
)
def test_whitespace_property_members(space_re, ch):
    assert space_re.fullmatch(ch), f"U+{ord(ch):04X} has White_Space"
    assert is_whitespace(ch)


@pytest.mark.parametrize("ch", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_c0_separators_are_not_unicode_whitespace(space_re, ch):
    """The whole reason this module exists.

    Python's \\s matches these; Rust's regex crate does not. A tokenizer built
    on Python's \\s splits U+001C differently from the reference and produces
    wrong token IDs on any input containing one.
    """
    assert not space_re.fullmatch(ch), f"U+{ord(ch):04X} must not count as whitespace"
    assert not is_whitespace(ch)
    # Confirm the divergence is real and not imagined.
    assert re.fullmatch(r"\s", ch), "Python's \\s does match it -- that is the trap"


def test_zero_width_space_is_not_whitespace(space_re):
    """U+200B looks like a space and is not one, in either engine."""
    assert not space_re.fullmatch("​")
    assert not is_whitespace("​")


def test_every_declared_whitespace_codepoint_matches(space_re):
    """Exhaustive over the property, not just the sample above."""
    for start, end in WHITESPACE_RANGES:
        for cp in range(start, end + 1):
            assert space_re.fullmatch(chr(cp)), f"U+{cp:04X}"
            assert is_whitespace(chr(cp)), f"U+{cp:04X}"


def test_whitespace_set_is_exactly_25_codepoints():
    total = sum(end - start + 1 for start, end in WHITESPACE_RANGES)
    assert total == 25


# -- class construction ----------------------------------------------------


def test_ranges_to_class_body_uses_dashes_for_runs():
    assert ranges_to_class_body(((0x61, 0x7A),)) == "a-z"


def test_ranges_to_class_body_expands_pairs():
    """A two-codepoint range is written out rather than hyphenated."""
    assert ranges_to_class_body(((0x61, 0x62),)) == "ab"


def test_ranges_to_class_body_single_codepoint():
    assert ranges_to_class_body(((0x61, 0x61),)) == "a"


def test_class_body_escapes_regex_metacharacters():
    """Characters that would break a [...] class must be escaped."""
    body = ranges_to_class_body(((0x5D, 0x5D), (0x5E, 0x5E), (0x2D, 0x2D)))
    assert "\\]" in body and "\\^" in body and "\\-" in body
    re.compile(f"[{body}]")  # must still compile


def test_control_characters_are_hex_escaped():
    assert ranges_to_class_body(((0x00, 0x00),)) == "\\x00"


def test_every_generated_class_compiles():
    for body in (letter_class_body(), number_class_body(), whitespace_class_body()):
        re.compile(f"[{body}]")
        re.compile(f"[^{body}]")


def test_class_body_is_cached():
    assert class_body("L") is class_body("L")


def test_category_ranges_are_sorted_and_disjoint():
    ranges = category_ranges("L")
    assert ranges == tuple(sorted(ranges))
    for (_, end), (next_start, _) in zip(ranges, ranges[1:]):
        assert end + 1 < next_start, "adjacent ranges should have been merged"


def test_category_ranges_cover_the_whole_codespace_correctly():
    """Spot-check the ranges against unicodedata across a wide sample."""
    ranges = category_ranges("N")
    in_ranges = lambda cp: any(s <= cp <= e for s, e in ranges)  # noqa: E731
    for cp in range(0, sys.maxunicode + 1, 997):   # prime stride, ~1100 samples
        expected = unicodedata.category(chr(cp)).startswith("N")
        assert in_ranges(cp) == expected, f"U+{cp:04X}"
