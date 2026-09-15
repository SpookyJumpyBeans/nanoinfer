"""The complete tokenizer: text in, token IDs out, and back again.

This assembles the pieces built in the preceding modules into the pipeline the
model's ``tokenizer.json`` declares:

    raw text
      -> extract added tokens          (matched on raw text, never merged)
      -> NFC normalize                 (the declared normalizer)
      -> split on the pre-token regex  (nanoinfer.pretokenize)
      -> map bytes to the alphabet     (nanoinfer.bytelevel)
      -> merge                         (nanoinfer.bpe)
      -> look up IDs

Added tokens are pulled out *first*, against the raw string. That ordering is
not a detail: if ``<|im_start|>`` went through the pre-tokenizer it would be
shredded into ``<``, ``|``, ``im``, ``_start``, ``|``, ``>`` and BPE would
happily encode it as ordinary text. Every chat-formatted prompt would then be
wrong in a way that still produces fluent output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from nanoinfer.bpe import BPE
from nanoinfer.bytelevel import decode_bytes
from nanoinfer.pretokenize import PreTokenizer


class TokenizerSpecError(ValueError):
    """The tokenizer.json declares something this implementation does not do."""


@dataclass(frozen=True, slots=True)
class AddedToken:
    """A token inserted above the trained vocabulary, matched literally."""

    id: int
    content: str
    special: bool


class Tokenizer:
    """Byte-level BPE tokenizer driven entirely by a ``tokenizer.json``.

    Nothing about Qwen is hardcoded. The pre-token pattern, the merge table,
    the vocabulary and the added tokens all come out of the file, and the
    constructor refuses configurations it has not implemented rather than
    silently ignoring them.
    """

    def __init__(
        self,
        bpe: BPE,
        pretokenizer: PreTokenizer,
        added_tokens: list[AddedToken],
    ) -> None:
        self.bpe = bpe
        self.pretokenizer = pretokenizer
        self.added_tokens = added_tokens

        self._added_by_content = {t.content: t.id for t in added_tokens}
        self._added_by_id = {t.id: t for t in added_tokens}
        self.special_ids = frozenset(t.id for t in added_tokens if t.special)

        # id -> token string, for decoding. Base vocabulary first, then added
        # tokens, which by construction occupy IDs above it.
        self._id_to_token: dict[int, str] = {i: s for s, i in bpe.vocab.items()}
        for token in added_tokens:
            self._id_to_token[token.id] = token.content

        # Longest content first, so <|im_start|> cannot be shadowed by a
        # shorter added token that happens to be a prefix of it.
        if added_tokens:
            alternation = "|".join(
                re.escape(t.content)
                for t in sorted(added_tokens, key=lambda t: -len(t.content))
            )
            self._added_re: re.Pattern[str] | None = re.compile(f"({alternation})")
        else:
            self._added_re = None

    # -- construction -----------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))

        model = spec["model"]
        if model.get("type") != "BPE":
            raise TokenizerSpecError(f"model type {model.get('type')!r} is not BPE")
        for unsupported, why in (
            ("dropout", "BPE dropout is a training-time regularizer"),
            ("continuing_subword_prefix", "not used by byte-level BPE"),
            ("end_of_word_suffix", "not used by byte-level BPE"),
        ):
            value = model.get(unsupported)
            if value:
                raise TokenizerSpecError(f"{unsupported}={value!r} is not implemented ({why})")
        if model.get("byte_fallback"):
            raise TokenizerSpecError(
                "byte_fallback is not implemented; byte-level BPE does not need it"
            )

        normalizer = spec.get("normalizer")
        if normalizer is not None and normalizer.get("type") != "NFC":
            raise TokenizerSpecError(
                f"normalizer {normalizer.get('type')!r} is not implemented; expected NFC"
            )

        pretokenizer = cls._build_pretokenizer(spec, normalize=normalizer is not None)

        added = [
            AddedToken(id=int(t["id"]), content=t["content"], special=bool(t["special"]))
            for t in spec.get("added_tokens", [])
        ]

        return cls(
            bpe=BPE.from_spec(model["vocab"], model["merges"]),
            pretokenizer=pretokenizer,
            added_tokens=added,
        )

    @staticmethod
    def _build_pretokenizer(spec: dict, normalize: bool) -> PreTokenizer:
        node = spec.get("pre_tokenizer")
        if node is None:
            raise TokenizerSpecError("no pre_tokenizer declared")

        stages = node["pretokenizers"] if node.get("type") == "Sequence" else [node]

        pattern: str | None = None
        for stage in stages:
            kind = stage.get("type")
            if kind == "Split":
                if stage.get("invert"):
                    raise TokenizerSpecError("inverted Split is not implemented")
                if stage.get("behavior") != "Isolated":
                    raise TokenizerSpecError(
                        f"Split behavior {stage.get('behavior')!r} is not implemented"
                    )
                pattern = stage["pattern"]["Regex"]
            elif kind == "ByteLevel":
                if stage.get("add_prefix_space"):
                    raise TokenizerSpecError("add_prefix_space is not implemented")
                if stage.get("use_regex"):
                    raise TokenizerSpecError(
                        "ByteLevel use_regex=true would apply a second split"
                    )
            else:
                raise TokenizerSpecError(f"pre-tokenizer stage {kind!r} is not implemented")

        if pattern is None:
            raise TokenizerSpecError("no Split stage found in the pre-tokenizer")
        return PreTokenizer.from_pattern(pattern, normalize=normalize)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Tokenizer":
        return cls.from_file(Path(model_dir) / "tokenizer.json")

    # -- encoding ---------------------------------------------------------

    def encode(self, text: str, split_added_tokens: bool = True) -> list[int]:
        """Encode text into token IDs.

        With ``split_added_tokens=False`` the added tokens are not recognised
        and ``<|im_start|>`` encodes as its literal characters. That is the
        right behaviour for untrusted user text, where a string in the input
        should not be able to forge a control token.
        """
        if not text:
            return []

        ids: list[int] = []
        for segment, is_added in self._segments(text, split_added_tokens):
            if is_added:
                ids.append(self._added_by_content[segment])
            else:
                for piece in self.pretokenizer.encode(segment):
                    ids.extend(self.bpe.encode_piece(piece))
        return ids

    def _segments(self, text: str, split_added: bool):
        """Yield ``(segment, is_added_token)`` over the raw input."""
        if not split_added or self._added_re is None:
            yield text, False
            return

        cursor = 0
        for match in self._added_re.finditer(text):
            if match.start() > cursor:
                yield text[cursor : match.start()], False
            yield match.group(), True
            cursor = match.end()
        if cursor < len(text):
            yield text[cursor:], False

    def encode_to_tokens(self, text: str, split_added_tokens: bool = True) -> list[str]:
        """The token strings rather than IDs. For debugging a mismatch."""
        return [self._id_to_token[i] for i in self.encode(text, split_added_tokens)]

    # -- decoding ---------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Turn IDs back into text.

        Base-vocabulary tokens are concatenated as bytes *before* decoding
        UTF-8, which is the only correct order: a multi-byte character is
        routinely split across several tokens, so decoding each token
        separately would produce a replacement character where a perfectly
        good character belongs.

        ``errors`` is passed to the final UTF-8 decode. It has to be lenient by
        default because a partial generation can legitimately end mid-character.
        """
        out: list[str] = []
        buffer = bytearray()

        for token_id in ids:
            added = self._added_by_id.get(token_id)
            if added is not None:
                if buffer:
                    out.append(buffer.decode("utf-8", errors=errors))
                    buffer.clear()
                if not (skip_special_tokens and added.special):
                    out.append(added.content)
                continue

            token = self._id_to_token.get(token_id)
            if token is None:
                raise KeyError(
                    f"token id {token_id} is not in the tokenizer "
                    f"(valid range 0..{self.vocab_size - 1}). Note that the "
                    "model's embedding matrix is larger than the vocabulary, "
                    "so sampling can produce IDs with no token."
                )
            buffer.extend(decode_bytes(token))

        if buffer:
            out.append(buffer.decode("utf-8", errors=errors))
        return "".join(out)

    # -- introspection ----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Base vocabulary plus added tokens.

        This is smaller than the model's ``vocab_size``: the embedding matrix
        is padded up to a round number for kernel efficiency, leaving IDs that
        are addressable but have no token.
        """
        return len(self.bpe.vocab) + len(self.added_tokens)

    def token_to_id(self, token: str) -> int | None:
        if token in self._added_by_content:
            return self._added_by_content[token]
        return self.bpe.vocab.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)
