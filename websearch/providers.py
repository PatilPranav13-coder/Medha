"""Standard-library search provider adapters and fallback ordering."""
from __future__ import annotations

import json
import html
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

USER_AGENT = "MedhaAI/0.1 (+https://github.com/PatilPranav13-coder/Medha)"
MAX_PROVIDER_RESPONSE = 2 * 1024 * 1024


class SearchProviderError(RuntimeError):
    """A provider returned no usable response; the chain may try another provider."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, timeout: float) -> list[SearchResult]: ...


def _fetch_json(
    url: str,
    timeout: float,
    headers: dict[str, str] | None = None,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict:
    request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        response = opener(request, timeout=timeout)
        with response:
            raw = response.read(MAX_PROVIDER_RESPONSE + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SearchProviderError("Search provider could not be reached.") from exc
    if len(raw) > MAX_PROVIDER_RESPONSE:
        raise SearchProviderError("Search provider response exceeded the size limit.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SearchProviderError("Search provider returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise SearchProviderError("Search provider returned an unexpected response.")
    return data


def _clean_result(title: object, url: object, snippet: object, provider: str) -> SearchResult | None:
    if not isinstance(url, str):
        return None
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    from websearch.ranking import extract_html

    snippet_text, _ = extract_html(html.unescape(str(snippet or "")))
    return SearchResult(
        title=" ".join(html.unescape(str(title or "Untitled result")).split())[:300],
        url=url[:2048],
        snippet=" ".join(snippet_text.split())[:1200],
        provider=provider,
    )


class BraveSearchProvider:
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None, opener: Callable[..., object] = urllib.request.urlopen) -> None:
        self.api_key = api_key
        self.opener = opener

    def search(self, query: str, timeout: float) -> list[SearchResult]:
        if not self.api_key:
            raise SearchProviderError("Brave Search is not configured.")
        if not query.strip() or len(query) > 600:
            raise SearchProviderError("Search query is empty or too long.")
        url = self.endpoint + "?" + urllib.parse.urlencode({"q": query, "count": 8})
        data = _fetch_json(
            url, timeout,
            {"X-Subscription-Token": self.api_key},
            self.opener,
        )
        web = data.get("web", {})
        rows = web.get("results", []) if isinstance(web, dict) else []
        if not isinstance(rows, list):
            return []
        return [result for row in rows if isinstance(row, dict) if (result := _clean_result(
            row.get("title"), row.get("url"), row.get("description"), self.name
        )) is not None]


class SearXNGProvider:
    name = "searxng"

    def __init__(self, base_url: str | None, opener: Callable[..., object] = urllib.request.urlopen) -> None:
        self.base_url = base_url
        self.opener = opener

    def search(self, query: str, timeout: float) -> list[SearchResult]:
        if not self.base_url:
            raise SearchProviderError("SearXNG is not configured.")
        if not query.strip() or len(query) > 600:
            raise SearchProviderError("Search query is empty or too long.")
        base = urllib.parse.urlsplit(self.base_url)
        if base.scheme not in {"http", "https"} or not base.hostname:
            raise SearchProviderError("MEDHA_SEARXNG_URL must be an http or https URL.")
        endpoint = self.base_url.rstrip("/")
        if not urllib.parse.urlsplit(endpoint).path.rstrip("/").endswith("/search"):
            endpoint += "/search"
        separator = "&" if "?" in endpoint else "?"
        url = endpoint + separator + urllib.parse.urlencode({"q": query, "format": "json"})
        data = _fetch_json(url, timeout, opener=self.opener)
        rows = data.get("results", [])
        if not isinstance(rows, list):
            return []
        return [result for row in rows if isinstance(row, dict) if (result := _clean_result(
            row.get("title"), row.get("url"), row.get("content"), self.name
        )) is not None]


class WikipediaSearchProvider:
    """Public Wikipedia API is always retained as the final no-key fallback."""

    name = "wikipedia"
    endpoint = "https://en.wikipedia.org/w/api.php"

    def __init__(self, opener: Callable[..., object] = urllib.request.urlopen) -> None:
        self.opener = opener

    def search(self, query: str, timeout: float) -> list[SearchResult]:
        if not query.strip() or len(query) > 600:
            raise SearchProviderError("Search query is empty or too long.")
        params = {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "utf8": "1", "srlimit": "8",
        }
        url = self.endpoint + "?" + urllib.parse.urlencode(params)
        data = _fetch_json(url, timeout, opener=self.opener)
        query_result = data.get("query", {})
        rows = query_result.get("search", []) if isinstance(query_result, dict) else []
        if not isinstance(rows, list):
            return []
        results: list[SearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", ""))
            page = urllib.parse.quote(title.replace(" ", "_"), safe="()_,-")
            result = _clean_result(title, "https://en.wikipedia.org/wiki/" + page, row.get("snippet"), self.name)
            if result is not None:
                results.append(result)
        return results


class SearchProviderChain:
    def __init__(self, providers: Sequence[SearchProvider], max_results: int = 8) -> None:
        self.providers = list(providers)
        self.max_results = max_results

    def search(self, query: str, timeout: float = 4.0) -> list[SearchResult]:
        errors: list[str] = []
        for provider in self.providers:
            try:
                results = provider.search(query, timeout)
            except SearchProviderError:
                errors.append(provider.name)
                continue
            if results:
                # Keep the first useful provider's results; duplicate ranking is handled later.
                return results[:self.max_results]
        if errors:
            raise SearchProviderError("All configured search providers failed or returned no results.")
        return []


def build_provider_chain(
    brave_api_key: str | None = None,
    searxng_url: str | None = None,
    override: str | None = None,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> SearchProviderChain:
    """Order configured providers by override and always keep Wikipedia last."""
    providers: dict[str, SearchProvider] = {
        "brave": BraveSearchProvider(brave_api_key, opener),
        "searxng": SearXNGProvider(searxng_url, opener),
        "wikipedia": WikipediaSearchProvider(opener),
    }
    selected = (override or "auto").strip().lower()
    if selected not in {"auto", *providers.keys()}:
        raise ValueError("MEDHA_SEARCH_PROVIDER must be auto, brave, searxng, or wikipedia.")
    if selected == "auto":
        order = [name for name, configured in (("brave", brave_api_key), ("searxng", searxng_url)) if configured]
        order.append("wikipedia")
    else:
        order = [selected] + [name for name in ("brave", "searxng", "wikipedia") if name != selected]
    return SearchProviderChain([providers[name] for name in order])


def provider_chain_from_env() -> SearchProviderChain:
    return build_provider_chain(
        brave_api_key=os.getenv("BRAVE_API_KEY"),
        searxng_url=os.getenv("MEDHA_SEARXNG_URL"),
        override=os.getenv("MEDHA_SEARCH_PROVIDER"),
    )
