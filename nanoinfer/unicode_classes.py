"""Unicode property classes for the standard library's ``re`` module.

The pre-tokenizer pattern shipped in ``tokenizer.json`` is written for Rust's
``regex`` crate and uses Unicode property escapes:

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+...

Python's ``re`` does not support ``\\p{...}``. The usual answer is to install the
third-party ``regex`` package; instead we build the character classes ourselves
from ``unicodedata``, which keeps the engine's runtime dependency list at numpy
and makes the semantics explicit rather than borrowed.

There is a subtlety here that is worth more than it looks, because it is
exactly the kind of thing that produces a tokenizer which is right on 99.9% of
inputs and quietly wrong on the rest:

**Python's ``\\s`` is not Rust's ``\\s``.** For ``str`` patterns CPython matches
whatever ``str.isspace()`` accepts, which includes the C0 separators
U+001C..U+001F. Rust's ``regex`` matches the Unicode ``White_Space`` property,
which does not. Feeding a file-separator byte through a tokenizer built on
Python's ``\\s`` therefore splits differently from the reference. So we do not
use ``\\s`` at all -- :data:`WHITESPACE_RANGES` spells out the ``White_Space``
property directly.
"""

from __future__ import annotations

import sys
import unicodedata
from functools import lru_cache

# The Unicode White_Space property, verbatim from PropList.txt. This is the
# complete set -- 25 codepoints in 11 ranges -- and it is stable across Unicode
# versions, so hardcoding it is safer than deriving it.
#
# Note what is absent: U+001C..U+001F, which Python's \s matches and Rust's
# does not.
WHITESPACE_RANGES: tuple[tuple[int, int], ...] = (
    (0x0009, 0x000D),   # tab, LF, VT, FF, CR
    (0x0020, 0x0020),   # space
    (0x0085, 0x0085),   # next line
    (0x00A0, 0x00A0),   # no-break space
    (0x1680, 0x1680),   # ogham space mark
    (0x2000, 0x200A),   # en quad .. hair space
    (0x2028, 0x2028),   # line separator
    (0x2029, 0x2029),   # paragraph separator
    (0x202F, 0x202F),   # narrow no-break space
    (0x205F, 0x205F),   # medium mathematical space
    (0x3000, 0x3000),   # ideographic space
)


@lru_cache(maxsize=None)
def category_ranges(prefix: str) -> tuple[tuple[int, int], ...]:
    """Every codepoint whose general category starts with ``prefix``, as ranges.

    ``"L"`` gives ``\\p{L}`` (all letters: Lu, Ll, Lt, Lm, Lo) and ``"N"`` gives
    ``\\p{N}`` (Nd, Nl, No). Scanning the whole codespace takes a moment, so the
    result is cached; it is a pure function of the Python build's Unicode
    tables.

    Those tables are the one version-dependent thing here. Python 3.13 ships
    Unicode 15.1; a reference tokenizer built against a different version could
    disagree about codepoints assigned in between. In practice the disagreement
    can only involve characters that were unassigned at the older version, and
    ``tests/test_tokenizer_reference.py`` checks the real corpus for it.
    """
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous = -1

    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)).startswith(prefix):
            if start is None:
                start = cp
            previous = cp
        elif start is not None:
            ranges.append((start, previous))
            start = None

    if start is not None:
        ranges.append((start, previous))
    return tuple(ranges)


def escape_for_class(cp: int) -> str:
    """Render one codepoint safe to place inside a ``[...]`` character class."""
    ch = chr(cp)
    if ch in "\\]^-[":
        return "\\" + ch
    if cp < 0x20 or cp == 0x7F:
        return f"\\x{cp:02x}"
    return ch


def ranges_to_class_body(ranges: tuple[tuple[int, int], ...]) -> str:
    """Turn ``[(start, end), ...]`` into the inside of a regex character class."""
    parts: list[str] = []
    for start, end in ranges:
        if start == end:
            parts.append(escape_for_class(start))
        elif end == start + 1:
            parts.append(escape_for_class(start) + escape_for_class(end))
        else:
            parts.append(f"{escape_for_class(start)}-{escape_for_class(end)}")
    return "".join(parts)


@lru_cache(maxsize=None)
def class_body(*prefixes: str) -> str:
    """The character-class body for the union of one or more categories.

    ``class_body("L", "N")`` is the body of ``[\\p{L}\\p{N}]``.
    """
    body = []
    for prefix in prefixes:
        if prefix == "White_Space":
            body.append(ranges_to_class_body(WHITESPACE_RANGES))
        else:
            body.append(ranges_to_class_body(category_ranges(prefix)))
    return "".join(body)


def letter_class_body() -> str:
    r"""Body of ``\p{L}``."""
    return class_body("L")


def number_class_body() -> str:
    r"""Body of ``\p{N}``."""
    return class_body("N")


def whitespace_class_body() -> str:
    r"""Body of ``\s`` as Rust's regex crate defines it, not as Python does."""
    return class_body("White_Space")


def is_whitespace(ch: str) -> bool:
    """Whether a character has the Unicode ``White_Space`` property."""
    cp = ord(ch)
    return any(start <= cp <= end for start, end in WHITESPACE_RANGES)
