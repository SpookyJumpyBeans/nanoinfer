"""The pre-tokenizer: chopping text into pieces before BPE ever runs.

BPE merges never cross a pre-token boundary. That single rule is what stops
the vocabulary from learning a token that spans " the" and "dog", and it is
also why the pre-tokenizer is where mismatches hide: get the split wrong and
the merge algorithm is still perfectly correct, it is just being fed different
input, so the output looks like plausible text with wrong IDs.

Qwen2.5 declares a two-stage pre-tokenizer in ``tokenizer.json``:

    Sequence[ Split(Regex(...), behavior="Isolated"), ByteLevel(use_regex=False) ]

The Split stage cuts text on a GPT-4-style pattern; the ByteLevel stage maps
each resulting piece's UTF-8 bytes through the printable alphabet in
:mod:`nanoinfer.bytelevel`. ``use_regex=False`` on the ByteLevel stage matters:
it means ByteLevel does *not* apply its own built-in splitting, because the
Split stage already did it.

The pattern is read from the model's own file and translated, rather than
copied into the source. A hardcoded copy silently goes stale the moment the
tokenizer is swapped for another model's.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from nanoinfer.bytelevel import encode_bytes
from nanoinfer.unicode_classes import (
    letter_class_body,
    number_class_body,
    whitespace_class_body,
)


class PatternTranslationError(ValueError):
    """Raised when a pattern uses a construct we have not implemented."""


def translate_pattern(pattern: str) -> str:
    r"""Rewrite a Rust-``regex`` pattern into one Python's ``re`` accepts.

    Only three constructs need rewriting, and all three are Unicode property
    escapes that Python lacks: ``\p{L}``, ``\p{N}``, and ``\s``/``\S``.
    (``\s`` is rewritten not because Python lacks it but because Python's
    definition is wrong here -- see :mod:`nanoinfer.unicode_classes`.)

    Whether a class expands with brackets depends on where it appears:
    ``\p{L}+`` outside a class becomes ``[a-z...]+``, but the same escape
    inside ``[^\r\n\p{L}\p{N}]`` must contribute only its body or the brackets
    would nest and the class would break.
    """
    bodies = {
        "L": letter_class_body,
        "N": number_class_body,
    }

    out: list[str] = []
    i = 0
    in_class = False
    n = len(pattern)

    while i < n:
        ch = pattern[i]

        if ch == "\\" and i + 1 < n:
            nxt = pattern[i + 1]

            if nxt == "p" and i + 2 < n and pattern[i + 2] == "{":
                close = pattern.find("}", i + 3)
                if close == -1:
                    raise PatternTranslationError(f"unterminated \\p{{ at index {i}")
                name = pattern[i + 3 : close]
                if name not in bodies:
                    raise PatternTranslationError(
                        f"Unicode property \\p{{{name}}} is not implemented"
                    )
                body = bodies[name]()
                out.append(body if in_class else f"[{body}]")
                i = close + 1
                continue

            if nxt == "s":
                body = whitespace_class_body()
                out.append(body if in_class else f"[{body}]")
                i += 2
                continue

            if nxt == "S":
                if in_class:
                    raise PatternTranslationError(
                        r"\S inside a character class cannot be expressed as a body"
                    )
                out.append(f"[^{whitespace_class_body()}]")
                i += 2
                continue

            # Any other escape passes through untouched.
            out.append(pattern[i : i + 2])
            i += 2
            continue

        if ch == "[" and not in_class:
            in_class = True
            out.append(ch)
            i += 1
            # A '^' or ']' immediately after '[' is literal, not a delimiter.
            if i < n and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < n and pattern[i] == "]":
                out.append("\\]")
                i += 1
            continue

        if ch == "]" and in_class:
            in_class = False
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    if in_class:
        raise PatternTranslationError("unterminated character class")

    return "".join(out)


@dataclass(frozen=True)
class PreTokenizer:
    """Splits text into pre-tokens and renders each in the byte-level alphabet.

    ``normalize`` applies the NFC normalizer the tokenizer declares. It is not
    cosmetic: NFC composes "e" + combining-acute into "é", which changes the
    byte sequence and therefore the tokens. Skipping it produces different IDs
    for any decomposed input.
    """

    regex: re.Pattern[str]
    normalize: bool = True

    @classmethod
    def from_pattern(cls, pattern: str, normalize: bool = True) -> "PreTokenizer":
        return cls(re.compile(translate_pattern(pattern)), normalize)

    def split(self, text: str) -> list[str]:
        """Cut text into pre-tokens, as raw (not yet byte-encoded) strings.

        The Split stage declares ``behavior="Isolated"``, which keeps every
        matched run as its own piece *and* preserves anything between matches.
        The pattern is written to cover all input, so gaps should never occur;
        we emit them anyway rather than silently dropping characters, and
        :func:`covers_without_gaps` lets the tests assert they do not happen.
        """
        if self.normalize:
            text = unicodedata.normalize("NFC", text)

        pieces: list[str] = []
        cursor = 0
        for match in self.regex.finditer(text):
            if match.start() > cursor:
                pieces.append(text[cursor : match.start()])
            if match.group():
                pieces.append(match.group())
            cursor = match.end()
        if cursor < len(text):
            pieces.append(text[cursor:])
        return pieces

    def encode(self, text: str) -> list[str]:
        """Pre-tokens rendered in the byte-level alphabet, ready for BPE."""
        return [encode_bytes(piece.encode("utf-8")) for piece in self.split(text)]

    def covers_without_gaps(self, text: str) -> bool:
        """Whether the pattern matched the entire input with no leftover runs."""
        if self.normalize:
            text = unicodedata.normalize("NFC", text)
        cursor = 0
        for match in self.regex.finditer(text):
            if match.start() != cursor:
                return False
            cursor = match.end()
        return cursor == len(text)
