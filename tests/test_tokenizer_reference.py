"""The phase 2 deliverable: 10,000 strings, exact token-ID equality.

"Close enough" is not a category here. A tokenizer that is 99.9% correct still
produces output that reads like fluent text, so the failure mode is not a crash
but days spent looking for a bug in the attention implementation. The gate is
exact equality on every string, and it runs before any model code exists.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from nanoinfer.tokenizer import Tokenizer
from tests.corpus import category_of, generate

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

CORPUS_SIZE = 10_000

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


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    return generate(CORPUS_SIZE)


def test_corpus_is_the_promised_size(corpus):
    assert len(corpus) == CORPUS_SIZE


def test_corpus_is_deterministic():
    assert generate(100) == generate(100)


def test_corpus_covers_every_category(corpus):
    categories = {category_of(i) for i in range(len(corpus))}
    assert len(categories) >= 14


@pytest.mark.reference
@pytest.mark.slow
def test_ten_thousand_strings_encode_identically(tok, reference, corpus):
    """Exact token-ID equality against the reference on every string.

    Failures are collected rather than raised on the first one: a systematic
    bug produces thousands of mismatches, and knowing which categories are
    affected localises it immediately.
    """
    mismatches: list[tuple[int, str, list[int], list[int]]] = []

    for index, text in enumerate(corpus):
        mine = tok.encode(text)
        theirs = reference.encode(text).ids
        if mine != theirs:
            mismatches.append((index, text, mine, theirs))

    if mismatches:
        from collections import Counter

        by_category = Counter(category_of(i) for i, _, _, _ in mismatches)
        report = [
            f"{len(mismatches)}/{len(corpus)} strings encoded differently",
            f"by category: {dict(by_category)}",
            "",
            "first 5 mismatches:",
        ]
        for index, text, mine, theirs in mismatches[:5]:
            report += [
                f"  [{index}] category={category_of(index)} text={text!r}",
                f"        mine  ({len(mine)}): {mine[:24]}",
                f"        ref   ({len(theirs)}): {theirs[:24]}",
                f"        mine tokens: {tok.encode_to_tokens(text)[:12]}",
                f"        ref  tokens: {reference.encode(text).tokens[:12]}",
            ]
        pytest.fail("\n".join(report))


@pytest.mark.reference
@pytest.mark.slow
def test_ten_thousand_strings_decode_identically(tok, reference, corpus):
    """Decoding must agree too, not just encoding."""
    mismatches = []
    for index, text in enumerate(corpus):
        ids = tok.encode(text)
        mine = tok.decode(ids)
        theirs = reference.decode(ids, skip_special_tokens=False)
        if mine != theirs:
            mismatches.append((index, text, mine, theirs))

    if mismatches:
        lines = [f"{len(mismatches)}/{len(corpus)} strings decoded differently"]
        for index, text, mine, theirs in mismatches[:5]:
            lines += [
                f"  [{index}] category={category_of(index)} input={text!r}",
                f"        mine: {mine!r}",
                f"        ref : {theirs!r}",
            ]
        pytest.fail("\n".join(lines))


@pytest.mark.slow
def test_ten_thousand_strings_round_trip(tok, corpus):
    """encode -> decode must recover the NFC-normalized input.

    Normalized, not original: the tokenizer declares an NFC normalizer, so
    decomposed input is composed on the way in and cannot be recovered. That
    is a property of the tokenizer, not a defect in this implementation.
    """
    failures = []
    for index, text in enumerate(corpus):
        expected = unicodedata.normalize("NFC", text)
        got = tok.decode(tok.encode(text))
        if got != expected:
            failures.append((index, text, expected, got))

    if failures:
        lines = [f"{len(failures)}/{len(corpus)} strings failed to round-trip"]
        for index, text, expected, got in failures[:5]:
            lines += [
                f"  [{index}] category={category_of(index)}",
                f"        input   : {text!r}",
                f"        expected: {expected!r}",
                f"        got     : {got!r}",
            ]
        pytest.fail("\n".join(lines))
