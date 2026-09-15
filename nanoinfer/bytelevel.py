"""The byte-level alphabet that byte-level BPE is built on.

BPE was originally defined over characters, which leaves an awkward question:
what happens to a character the vocabulary has never seen? Byte-level BPE
removes the question entirely by running BPE over the 256 possible *bytes*
instead. Every possible input is then representable, and there is no need for
an unknown token -- which is exactly why this tokenizer's ``unk_token`` is
null and ``byte_fallback`` is false.

The complication is cosmetic. BPE implementations, and the vocabulary files
they produce, work with text. Feeding raw bytes through would put control
characters, newlines and spaces directly into token strings, where they are
invisible in a vocab dump and where a whitespace-splitting merge file becomes
ambiguous. So each byte is first mapped to a *printable* Unicode codepoint:

    byte 0x41 ('A')  -> 'A'      already printable, unchanged
    byte 0x20 (' ')  -> 'Ġ'      U+0120, the famous leading-space marker
    byte 0x0A ('\\n') -> 'Ċ'      U+010A

This is a bijection over all 256 bytes, so nothing is lost. The 'Ġ' that shows
up everywhere in GPT-2-family vocabularies is not a special marker anyone
designed; it is simply where 0x20 landed in this mapping.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map each of the 256 byte values to a distinct printable codepoint.

    The construction, which is the one GPT-2 shipped and every byte-level BPE
    since has copied verbatim:

    1. Take the byte ranges that are already safely printable and leave them
       alone -- ``!``..``~``, ``¡``..``¬``, ``®``..``ÿ``. These map to
       themselves, so ASCII text stays readable in the vocabulary file.
    2. Every remaining byte (the C0 and C1 control ranges, space, and the three
       gaps at 0x7F, 0xA0 and 0xAD) is assigned the next free codepoint
       starting at U+0100, in increasing byte order.

    There are 188 bytes in step 1 and 68 in step 2, so the escapes occupy
    U+0100..U+0143. The exact ranges matter: they are baked into every
    published vocabulary, so a different-but-equally-valid mapping would
    decode every token wrong.
    """
    printable: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    byte_values = list(printable)
    codepoints = list(printable)

    next_free = 0
    for byte in range(256):
        if byte not in printable:
            byte_values.append(byte)
            codepoints.append(256 + next_free)
            next_free += 1

    return {b: chr(c) for b, c in zip(byte_values, codepoints)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """The inverse of :func:`bytes_to_unicode`, used when decoding."""
    return {ch: b for b, ch in bytes_to_unicode().items()}


def encode_bytes(data: bytes) -> str:
    """Render raw bytes as the printable string BPE will operate on."""
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)


def decode_bytes(text: str) -> bytes:
    """Recover the raw bytes from a token string.

    Raises ``KeyError`` on a character outside the alphabet, which can only
    happen if a token string did not come from this tokenizer's vocabulary.
    """
    table = unicode_to_bytes()
    try:
        return bytes(table[ch] for ch in text)
    except KeyError as exc:
        raise KeyError(
            f"character {exc.args[0]!r} is not in the byte-level alphabet; "
            "this string did not come from a byte-level BPE vocabulary"
        ) from None
