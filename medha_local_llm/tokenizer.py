"""A dependency-free byte-level BPE tokenizer with explicit chat tokens.

The byte vocabulary guarantees that any UTF-8 input remains representable,
including text outside the tiny research corpus. The merge table is stored as
plain data so it can be safely loaded from a versioned model checkpoint.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping

TOKENIZER_VERSION = 1
BYTE_VOCAB_SIZE = 256
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>")
_PRETOKEN = re.compile(r"\s+| ?\w+| ?[^\w\s]+", re.UNICODE)


class ByteBPETokenizer:
    """Train and apply a small byte-pair encoding vocabulary."""

    def __init__(self, merges: Iterable[tuple[int, int]] = ()) -> None:
        self.merges = [(int(left), int(right)) for left, right in merges]
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._merge_ids = {pair: BYTE_VOCAB_SIZE + rank for rank, pair in enumerate(self.merges)}
        self._token_bytes = [bytes((value,)) for value in range(BYTE_VOCAB_SIZE)]
        for left, right in self.merges:
            if left >= len(self._token_bytes) or right >= len(self._token_bytes):
                raise ValueError("BPE merge references a token that has not been created yet.")
            self._token_bytes.append(self._token_bytes[left] + self._token_bytes[right])
        self._special_ids = {
            token: BYTE_VOCAB_SIZE + len(self.merges) + index
            for index, token in enumerate(SPECIAL_TOKENS)
        }
        self._special_by_id = {index: token for token, index in self._special_ids.items()}
        pattern = "|".join(re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True))
        self._special_pattern = re.compile(f"({pattern})")

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        """Learn frequent adjacent byte-token pairs with deterministic tie breaks."""
        if vocab_size <= BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS):
            return cls()
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one.")
        if isinstance(texts, str):
            corpus = [texts]
        else:
            corpus = [text for text in texts if isinstance(text, str)]

        sequences: list[list[int]] = []
        for text in corpus:
            # BPE merges stay inside simple lexical pieces so merges do not span
            # arbitrary whitespace or punctuation boundaries.
            for piece in _PRETOKEN.findall(text):
                encoded = list(piece.encode("utf-8"))
                if encoded:
                    sequences.append(encoded)

        merges: list[tuple[int, int]] = []
        token_bytes = [bytes((value,)) for value in range(BYTE_VOCAB_SIZE)]
        merge_limit = vocab_size - BYTE_VOCAB_SIZE - len(SPECIAL_TOKENS)
        for _ in range(merge_limit):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            best = min(
                counts,
                key=lambda pair: (-counts[pair], token_bytes[pair[0]] + token_bytes[pair[1]], pair),
            )
            if counts[best] < min_frequency:
                break

            merged_id = len(token_bytes)
            merged_bytes = token_bytes[best[0]] + token_bytes[best[1]]
            updated: list[list[int]] = []
            for sequence in sequences:
                output: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == best:
                        output.append(merged_id)
                        index += 2
                    else:
                        output.append(sequence[index])
                        index += 1
                updated.append(output)
            sequences = updated
            merges.append(best)
            token_bytes.append(merged_bytes)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return len(self._token_bytes) + len(self._special_ids)

    @property
    def pad_token_id(self) -> int:
        return self._special_ids["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self._special_ids["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self._special_ids["<eos>"]

    @property
    def special_token_ids(self) -> Mapping[str, int]:
        return dict(self._special_ids)

    def _encode_piece(self, piece: str) -> list[int]:
        symbols = list(piece.encode("utf-8"))
        while len(symbols) > 1:
            candidates = (
                (self._merge_ranks[pair], index, pair)
                for index, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self._merge_ranks
            )
            best = min(candidates, default=None)
            if best is None:
                break
            _, _, pair = best
            merged_id = self._merge_ids[pair]
            output: list[int] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                    output.append(merged_id)
                    index += 2
                else:
                    output.append(symbols[index])
                    index += 1
            symbols = output
        return symbols

    def encode(self, text: str, allow_special_tokens: bool = False) -> list[int]:
        """Encode text; user content treats special-token spellings as ordinary text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not allow_special_tokens:
            return [token for piece in _PRETOKEN.findall(text) for token in self._encode_piece(piece)]

        result: list[int] = []
        for piece in self._special_pattern.split(text):
            if not piece:
                continue
            special_id = self._special_ids.get(piece)
            if special_id is not None:
                result.append(special_id)
            else:
                result.extend(token for word in _PRETOKEN.findall(piece) for token in self._encode_piece(word))
        return result

    def encode_chat(
        self,
        messages: Iterable[Mapping[str, str]],
        add_assistant_prompt: bool = False,
    ) -> list[int]:
        """Encode role-tagged turns without allowing message text to forge roles."""
        role_tokens = {
            "system": self._special_ids["<system>"],
            "user": self._special_ids["<user>"],
            "assistant": self._special_ids["<assistant>"],
        }
        result = [self.bos_token_id]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in role_tokens or not isinstance(content, str):
                raise ValueError("Each chat message needs a supported role and string content.")
            result.append(role_tokens[role])
            result.extend(self.encode(content))
        if add_assistant_prompt:
            result.append(role_tokens["assistant"])
        else:
            result.append(self.eos_token_id)
        return result

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        chunks: list[bytes] = []
        for value in token_ids:
            token = int(value)
            if token in self._special_by_id:
                if not skip_special_tokens:
                    chunks.append(self._special_by_id[token].encode("utf-8"))
            elif 0 <= token < len(self._token_bytes):
                chunks.append(self._token_bytes[token])
            else:
                raise ValueError(f"Token id {token} is outside this vocabulary.")
        return b"".join(chunks).decode("utf-8", errors="replace")

    def state(self) -> dict[str, object]:
        return {
            "version": TOKENIZER_VERSION,
            "merges": [[left, right] for left, right in self.merges],
        }

    @classmethod
    def from_state(cls, state: object) -> "ByteBPETokenizer":
        if not isinstance(state, dict) or state.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"Expected tokenizer version {TOKENIZER_VERSION}.")
        raw_merges = state.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("Tokenizer checkpoint is missing its BPE merge table.")
        merges: list[tuple[int, int]] = []
        for pair in raw_merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(value, int) or value < 0 for value in pair)
            ):
                raise ValueError("Tokenizer checkpoint contains an invalid BPE merge.")
            merges.append((pair[0], pair[1]))
        return cls(merges)
