"""Tests for the BPE merge algorithm, on hand-built models.

Toy vocabularies are used deliberately: with a three-rule merge table the
correct answer can be worked out on paper, so a failure points at the
algorithm rather than at 151,387 merge rules.
"""

from __future__ import annotations

import pytest

from nanoinfer.bpe import BPE, UnknownTokenError


def toy(merges: list[str], extra_vocab: list[str] | None = None) -> BPE:
    """Build a BPE over lowercase letters plus whatever the merges produce."""
    symbols = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    for rule in merges:
        symbols.append(rule.replace(" ", ""))
    symbols.extend(extra_vocab or [])

    vocab = {s: i for i, s in enumerate(dict.fromkeys(symbols))}
    return BPE.from_spec(vocab, merges)


# -- merge table construction ---------------------------------------------


def test_accepts_space_separated_merges():
    bpe = BPE.from_spec({}, ["a b", "ab c"])
    assert bpe.merge_ranks == {("a", "b"): 0, ("ab", "c"): 1}


def test_accepts_list_pair_merges():
    """Newer tokenizer.json files store merges as two-element lists."""
    bpe = BPE.from_spec({}, [["a", "b"], ["ab", "c"]])
    assert bpe.merge_ranks == {("a", "b"): 0, ("ab", "c"): 1}


def test_rank_is_the_position_in_the_file():
    bpe = BPE.from_spec({}, ["x y", "a b", "c d"])
    assert bpe.merge_ranks[("x", "y")] == 0
    assert bpe.merge_ranks[("c", "d")] == 2


def test_rejects_malformed_merge_rule():
    with pytest.raises(ValueError, match="not a pair"):
        BPE.from_spec({}, ["a b c"])


# -- merging ---------------------------------------------------------------


def test_no_merges_leaves_characters_alone():
    bpe = toy([])
    assert bpe.merge("abc") == ("a", "b", "c")


def test_single_merge_applies():
    bpe = toy(["a b"])
    assert bpe.merge("ab") == ("ab",)


def test_merges_apply_repeatedly_up_the_table():
    """ab merges first, then the new symbol merges with c."""
    bpe = toy(["a b", "ab c"])
    assert bpe.merge("abc") == ("abc",)


def test_lower_rank_wins_regardless_of_position():
    """(b,c) is rank 0, so it merges before (a,b) at rank 1."""
    bpe = toy(["b c", "a b"])
    assert bpe.merge("abc") == ("a", "bc")


def test_equal_ranks_resolve_leftmost():
    """The tie-break that a small corpus will not catch.

    Both positions in 'aa' + 'a' hold the pair (a,a) at rank 0. The leftmost
    must merge first; choosing the rightmost gives the same answer here but
    diverges on longer runs, so the behaviour is pinned explicitly.
    """
    bpe = toy(["a a"])
    assert bpe.merge("aaa") == ("aa", "a")


def test_repeated_runs_merge_pairwise():
    bpe = toy(["a a"])
    assert bpe.merge("aaaa") == ("aa", "aa")


def test_longer_merge_consumes_the_run():
    bpe = toy(["a a", "aa a"])
    assert bpe.merge("aaa") == ("aaa",)


def test_stops_when_no_pair_is_mergeable():
    bpe = toy(["a b"])
    assert bpe.merge("xyz") == ("x", "y", "z")


def test_single_character_is_returned_unchanged():
    bpe = toy(["a b"])
    assert bpe.merge("a") == ("a",)


def test_empty_piece():
    bpe = toy(["a b"])
    assert bpe.merge("") == ()


def test_merge_is_deterministic():
    bpe = toy(["a b", "ab c", "c d"])
    first = bpe.merge("abcd")
    bpe._cache.clear()
    assert bpe.merge("abcd") == first


# -- caching ---------------------------------------------------------------


def test_result_is_cached():
    bpe = toy(["a b"])
    bpe.merge("ab")
    assert "ab" in bpe._cache


def test_cache_returns_the_same_object():
    bpe = toy(["a b"])
    assert bpe.merge("ab") is bpe.merge("ab")


def test_cache_respects_its_limit():
    bpe = toy([])
    bpe.cache_limit = 3
    for word in ("aa", "bb", "cc", "dd", "ee"):
        bpe.merge(word)
    assert len(bpe._cache) == 3


def test_cache_does_not_change_results():
    bpe = toy(["a b", "ab c"])
    uncached = bpe.merge("abc")
    bpe._cache.clear()
    bpe.cache_limit = 0
    assert bpe.merge("abc") == uncached


# -- vocabulary lookup -----------------------------------------------------


def test_encode_piece_returns_ids():
    bpe = toy(["a b"])
    ids = bpe.encode_piece("ab")
    assert ids == [bpe.vocab["ab"]]


def test_encode_piece_falls_back_to_single_characters():
    bpe = toy(["a b"])
    assert bpe.encode_piece("xy") == [bpe.vocab["x"], bpe.vocab["y"]]


def test_encode_piece_raises_when_vocab_and_merges_disagree():
    """A merge rule producing a symbol the vocab lacks is a corrupt model."""
    bpe = BPE.from_spec({"a": 0, "b": 1}, ["a b"])   # 'ab' deliberately absent
    with pytest.raises(UnknownTokenError, match="not in the vocabulary"):
        bpe.encode_piece("ab")


def test_len_is_the_vocab_size():
    bpe = toy(["a b"])
    assert len(bpe) == len(bpe.vocab)
