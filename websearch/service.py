"""Bounded search orchestration with graceful failure and cited passages."""
from __future__ import annotations

import time
import threading
from collections.abc import Callable, Sequence
from typing import Any

from websearch.answer import SearchAnswer, SearchMode, should_search, synthesize_or_extract
from websearch.fetcher import PageFetchError, PublicPageFetcher
from websearch.providers import SearchProviderChain, SearchProviderError, SearchResult, provider_chain_from_env
from websearch.ranking import Passage, rank_passages, split_passages


class SearchService:
    def __init__(
        self,
        providers: SearchProviderChain | None = None,
        fetcher: PublicPageFetcher | None = None,
        deadline_seconds: float = 12.0,
        clock: Callable[[], float] = time.monotonic,
        synthesis_generate: Callable[[Sequence[dict[str, str]]], str] | None = None,
        synthesis_capable: bool = False,
    ) -> None:
        self.providers = providers or provider_chain_from_env()
        self.fetcher = fetcher or PublicPageFetcher()
        self.deadline_seconds = min(max(0.1, deadline_seconds), 12.0)
        self.clock = clock
        self.synthesis_generate = synthesis_generate
        self.synthesis_capable = synthesis_capable

    def search(self, query: str, mode: SearchMode = "auto", cancel_event: threading.Event | None = None) -> SearchAnswer:
        if mode not in {"auto", "always", "off"}:
            raise ValueError("Search mode must be auto, always, or off.")
        if not should_search(query, mode):
            return SearchAnswer("", [], False)
        if len(query.strip()) > 600:
            return SearchAnswer("This question is too long to send to web search. Try a shorter search query.", [], True, "query_too_long")
        if cancel_event is not None and cancel_event.is_set():
            return SearchAnswer("", [], False)
        deadline = self.clock() + self.deadline_seconds
        results: list[SearchResult] = []
        provider_failed = False
        for provider in self.providers.providers:
            if cancel_event is not None and cancel_event.is_set():
                return SearchAnswer("", [], False)
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                results = provider.search(query, min(remaining, 4.0))
            except SearchProviderError:
                provider_failed = True
                continue
            if results:
                break
        if not results:
            message = "I couldn't verify this with web sources just now. Please try again shortly."
            return SearchAnswer(message, [], True, "search_unavailable" if provider_failed else "no_results")

        passages: list[Passage] = []
        sources: list[dict[str, object]] = []
        # Search snippets provide partial answers even when a page blocks crawling.
        for source_number, result in enumerate(results, start=1):
            title, url, snippet = result.title[:300], result.url[:2048], result.snippet[:1200]
            sources.append({"n": source_number, "title": title, "url": url, "snippet": snippet[:500]})
            passages.extend(Passage(source_number, title, url, chunk) for chunk in split_passages(snippet))

        # Fetch only the best few URLs within the same deadline; a blocked page keeps its snippet.
        for source in sources[:3]:
            if cancel_event is not None and cancel_event.is_set():
                return SearchAnswer("", [], False)
            remaining = deadline - self.clock()
            if remaining <= 0.15:
                break
            number = int(source["n"])
            try:
                page = self.fetcher.fetch(str(source["url"]), timeout=min(1.5, remaining))
            except PageFetchError:
                continue
            source["title"] = page.title or source["title"]
            source["url"] = page.url
            source["snippet"] = page.text[:500]
            passages.extend(
                Passage(number, str(source["title"]), str(source["url"]), chunk)
                for chunk in split_passages(page.text)
            )

        if not passages:
            return SearchAnswer("I found search results, but they did not contain enough readable text to answer reliably.", sources, True, "empty_sources")
        ranked = rank_passages(query, passages, limit=16)
        selected = [item for _, item in ranked]
        # Send only relevant, bounded passages to the optional capable brain.
        selected = selected[:10]
        synthesis_time = max(0.0, deadline - self.clock())
        if cancel_event is not None and cancel_event.is_set():
            return SearchAnswer("", [], False)
        answer = synthesize_or_extract(
            query, selected, (int(source["n"]) for source in sources),
            self.synthesis_generate, self.synthesis_capable, synthesis_time,
        )
        return SearchAnswer(answer, sources, True)

    def __call__(self, query: str, mode: SearchMode = "auto") -> dict[str, Any]:
        result = self.search(query, mode)
        return {"answer": result.answer, "sources": result.sources, "searched": result.searched, "warning": result.warning}
