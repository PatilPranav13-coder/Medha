"""HTML-to-text extraction, passage chunking, and small-corpus BM25 ranking."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Sequence

WORD_RE = re.compile(r"[\w]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+")
DROP_TAGS = {"script", "style", "nav", "footer", "form", "noscript", "svg"}
BLOCK_TAGS = {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "tr", "blockquote"}


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._drop_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_TAGS:
                self._drop_depth += 1
            return
        if tag in DROP_TAGS:
            self._drop_depth = 1
            return
        if tag == "title":
            self._in_title = True
        if tag in BLOCK_TAGS and not self._in_title:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_TAGS:
                self._drop_depth = max(0, self._drop_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS and not self._in_title:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.text_parts.append(data)


def extract_html(html: str) -> tuple[str, str]:
    parser = _PlainTextParser()
    parser.feed(html)
    parser.close()
    text = re.sub(r"[ \t\xa0]+", " ", "".join(parser.text_parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    title = " ".join("".join(parser.title_parts).split())
    return text, title


def split_passages(text: str, max_chars: int = 1200, overlap_chars: int = 120) -> list[str]:
    if max_chars < 100 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Passage chunk limits are invalid.")
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            boundary = clean.rfind(". ", start + max_chars // 2, end)
            if boundary >= start:
                end = boundary + 1
        passage = clean[start:end].strip()
        if passage:
            chunks.append(passage)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


@dataclass(frozen=True)
class Passage:
    source_number: int
    title: str
    url: str
    text: str


def _terms(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(text)]


def rank_passages(query: str, passages: Sequence[Passage], limit: int = 12) -> list[tuple[float, Passage]]:
    """Rank a small set of retrieved passages with Okapi BM25."""
    query_terms = _terms(query)
    docs = [_terms(passage.text) for passage in passages]
    if not docs or not query_terms:
        return []
    frequencies = [Counter(doc) for doc in docs]
    lengths = [len(doc) for doc in docs]
    average_length = sum(lengths) / len(lengths) or 1.0
    scores: list[tuple[float, Passage]] = []
    k1, b = 1.5, 0.75
    for index, passage in enumerate(passages):
        score = 0.0
        for term in query_terms:
            document_frequency = sum(term in freq for freq in frequencies)
            if document_frequency == 0:
                continue
            idf = math.log(1 + (len(docs) - document_frequency + 0.5) / (document_frequency + 0.5))
            term_frequency = frequencies[index][term]
            denominator = term_frequency + k1 * (1 - b + b * lengths[index] / average_length)
            score += idf * term_frequency * (k1 + 1) / denominator
        scores.append((score, passage))
    scores.sort(key=lambda item: (-item[0], item[1].source_number, item[1].text))
    return scores[:max(0, limit)]


def sentence_candidates(passage: Passage) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(passage.text) if len(sentence.strip()) >= 25]
