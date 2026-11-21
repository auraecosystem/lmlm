"""Dependency-free byte-level BPE tokenizer for local LMLM models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TokenizerError(ValueError):
    """Invalid tokenizer metadata or input."""


def _byte_encoder() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible[:]
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {value: chr(codepoint) for value, codepoint in zip(visible, codepoints)}


_BYTE_ENCODER = _byte_encoder()
_BYTE_DECODER = {char: value for value, char in _BYTE_ENCODER.items()}


class ByteLevelBPETokenizer:
    """GPT-2-style byte alphabet and ranked pair merges.

    The tokenizer is loaded exclusively from local JSON metadata. It never
    downloads vocabulary files or executes tokenizer code from a model bundle.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        bos_token: str | None = None,
        eos_token: str | None = None,
        unk_token: str | None = None,
    ) -> None:
        if not vocab:
            raise TokenizerError("vocabulary cannot be empty")
        ids = list(vocab.values())
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in ids):
            raise TokenizerError("vocabulary IDs must be non-negative integers")
        if len(set(ids)) != len(ids):
            raise TokenizerError("vocabulary IDs must be unique")
        if set(ids) != set(range(len(ids))):
            raise TokenizerError("vocabulary IDs must be contiguous from zero")
        self.vocab = dict(vocab)
        self.inverse_vocab = {token_id: token for token, token_id in vocab.items()}
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if len(pair) != 2 or not all(isinstance(part, str) and part for part in pair):
                raise TokenizerError("each merge must contain two non-empty strings")
            if pair in self.ranks:
                raise TokenizerError("duplicate merge pair")
            self.ranks[pair] = rank
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        for name, token in (("bos_token", bos_token), ("eos_token", eos_token), ("unk_token", unk_token)):
            if token is not None and token not in self.vocab:
                raise TokenizerError(f"{name} is missing from vocabulary")
        self.unk_id = self.vocab.get(unk_token) if unk_token else None

    @classmethod
    def from_file(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        try:
            value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot read tokenizer file: {exc}") from exc
        if not isinstance(value, dict) or value.get("format_version") != 1:
            raise TokenizerError("unsupported tokenizer format")
        vocab = value.get("vocab")
        merges = value.get("merges", [])
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise TokenizerError("tokenizer vocab must be an object and merges must be an array")
        return cls(
            vocab,
            [tuple(pair) if isinstance(pair, list) else pair for pair in merges],
            bos_token=value.get("bos_token"),
            eos_token=value.get("eos_token"),
            unk_token=value.get("unk_token"),
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        symbols = [_BYTE_ENCODER[value] for value in text.encode("utf-8")]
        while len(symbols) > 1:
            candidates = [
                (self.ranks[(symbols[index], symbols[index + 1])], index)
                for index in range(len(symbols) - 1)
                if (symbols[index], symbols[index + 1]) in self.ranks
            ]
            if not candidates:
                break
            _, index = min(candidates)
            symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
        result: list[int] = []
        if add_bos and self.bos_token is not None:
            result.append(self.vocab[self.bos_token])
        for symbol in symbols:
            token_id = self.vocab.get(symbol, self.unk_id)
            if token_id is None:
                raise TokenizerError(f"token not found in vocabulary: {symbol!r}")
            result.append(token_id)
        if add_eos and self.eos_token is not None:
            result.append(self.vocab[self.eos_token])
        return result

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        special = {token for token in (self.bos_token, self.eos_token, self.unk_token) if token}
        encoded = "".join(
            self.inverse_vocab[token_id]
            for token_id in token_ids
            if token_id in self.inverse_vocab
            and not (skip_special_tokens and self.inverse_vocab[token_id] in special)
        )
        try:
            return bytes(_BYTE_DECODER[char] for char in encoded).decode("utf-8", errors="replace")
        except KeyError as exc:
            raise TokenizerError(f"token contains a character outside the byte alphabet: {exc}") from exc
