"""Byte-pair encoding: turning one pre-token into a sequence of vocabulary IDs.

The algorithm is short. A pre-token starts as a list of single characters (in
the byte-level alphabet, so one character is one byte). Repeatedly find the
adjacent pair with the lowest merge rank, join it into a single symbol, and
continue until no adjacent pair appears in the merge table. Whatever symbols
remain are looked up in the vocabulary.

The merge table is ordered: rank 0 is the pair the trainer merged first, so
"lowest rank" means "learned earliest", which means "most frequent". That
ordering is the entire model -- there is nothing else to BPE.

**Tie-breaking is load-bearing.** When two positions hold pairs of equal rank,
the leftmost one merges first. HuggingFace's implementation gets this from a
priority queue keyed on (rank, position); we get it from taking the first
strict improvement while scanning left to right. Picking the rightmost instead
produces different tokens on repeated-character runs, and it produces them
rarely enough that a small test corpus will not catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class UnknownTokenError(KeyError):
    """A symbol survived merging but is not in the vocabulary.

    With byte-level BPE this should be impossible: every single byte is in the
    vocabulary, so merging can only ever produce symbols that are either in the
    vocabulary or further mergeable. Seeing this means the vocab and merges
    disagree.
    """


@dataclass
class BPE:
    """A trained byte-pair-encoding model: a vocabulary plus ranked merges."""

    vocab: dict[str, int]
    merge_ranks: dict[tuple[str, str], int]
    _cache: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    cache_limit: int = 100_000

    @classmethod
    def from_spec(cls, vocab: dict[str, int], merges: list) -> "BPE":
        """Build from the ``model`` section of a ``tokenizer.json``.

        ``merges`` is a list in rank order. Older files store each entry as one
        space-separated string (``"Ġ t"``); newer ones store a two-element
        list. Both appear in the wild, so both are accepted.

        The space-separated form is only unambiguous because no token in the
        byte-level alphabet can contain a space -- which is exactly why the
        alphabet escapes 0x20 in the first place.
        """
        ranks: dict[tuple[str, str], int] = {}
        for rank, entry in enumerate(merges):
            if isinstance(entry, str):
                parts = entry.split(" ")
            else:
                parts = list(entry)
            if len(parts) != 2:
                raise ValueError(
                    f"merge rule {rank} is not a pair: {entry!r}"
                )
            ranks[(parts[0], parts[1])] = rank
        return cls(vocab=dict(vocab), merge_ranks=ranks)

    # -- the algorithm ----------------------------------------------------

    def merge(self, piece: str) -> tuple[str, ...]:
        """Apply merges to one byte-level pre-token until none apply.

        Results are cached because real text repeats pre-tokens relentlessly --
        the same " the" is merged thousands of times in a document, and the
        merge result depends only on the input string.
        """
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        symbols = list(piece)
        while len(symbols) > 1:
            best_rank: int | None = None
            best_at = -1

            # Scan left to right, keeping only strict improvements, so equal
            # ranks resolve to the leftmost position.
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_at = i

            if best_rank is None:
                break

            symbols[best_at : best_at + 2] = [symbols[best_at] + symbols[best_at + 1]]

        result = tuple(symbols)
        if len(self._cache) < self.cache_limit:
            self._cache[piece] = result
        return result

    def encode_piece(self, piece: str) -> list[int]:
        """Merge one pre-token and look the resulting symbols up in the vocab."""
        ids: list[int] = []
        for symbol in self.merge(piece):
            try:
                ids.append(self.vocab[symbol])
            except KeyError:
                raise UnknownTokenError(
                    f"symbol {symbol!r} is not in the vocabulary; "
                    "the vocab and merge table disagree"
                ) from None
        return ids

    def __len__(self) -> int:
        return len(self.vocab)
