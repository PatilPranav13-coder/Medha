from __future__ import annotations

import asyncio
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx


class _TextOnly(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text(value: str, limit: int = 900) -> str:
    parser = _TextOnly()
    parser.feed(value or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()[:limit]


def _trusted(url: str, allowed: set[str]) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == item or host.endswith("." + item) for item in allowed)


async def _json(client: httpx.AsyncClient, url: str, params: dict[str, str]) -> dict:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except (httpx.HTTPError, ValueError):
        return {}


async def _mdn(client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
    payload = await _json(client, "https://developer.mozilla.org/api/v1/search", {"q": query, "locale": "en-US"})
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return []
    sources = []
    for item in documents[:3]:
        if not isinstance(item, dict):
            continue
        path = item.get("mdn_url", "")
        url = path if path.startswith("https://") else "https://developer.mozilla.org" + path
        if _trusted(url, {"developer.mozilla.org"}):
            sources.append({"title": _text(item.get("title", "MDN Web Docs"), 180), "url": url, "snippet": _text(item.get("summary", "")), "kind": "documentation"})
    return sources


async def _learn(client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
    payload = await _json(client, "https://learn.microsoft.com/api/search", {"search": query, "locale": "en-us"})
    items = payload.get("results", [])
    if not isinstance(items, list):
        return []
    sources = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        if _trusted(url, {"learn.microsoft.com"}):
            sources.append({"title": _text(item.get("title", "Microsoft Learn"), 180), "url": url, "snippet": _text(item.get("description", item.get("summary", ""))), "kind": "documentation"})
    return sources


async def _stackoverflow(client: httpx.AsyncClient, query: str) -> list[dict[str, str]]:
    payload = await _json(client, "https://api.stackexchange.com/2.3/search/excerpts", {"q": query, "site": "stackoverflow", "pagesize": "3", "order": "desc", "sort": "relevance"})
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    sources = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        url = item.get("link", "")
        if not _trusted(url, {"stackoverflow.com"}):
            continue
        owner = item.get("owner") or {}
        author = owner.get("display_name", "Stack Overflow contributor")
        sources.append({
            "title": _text(item.get("title", "Stack Overflow post"), 180),
            "url": url,
            "snippet": _text(item.get("excerpt", "")),
            "kind": "community",
            "attribution": f"By {author}; CC BY-SA (version depends on post)",
        })
    return sources


async def search_public_docs(query: str) -> list[dict[str, str]]:
    query = re.sub(r"\s+", " ", query).strip()[:320]
    if not query:
        return []
    headers = {"User-Agent": "Medha-Technical-Workspace/1.0"}
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=False, headers=headers) as client:
        groups = await asyncio.gather(_mdn(client, query), _learn(client, query), _stackoverflow(client, query), return_exceptions=True)
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        if isinstance(group, Exception):
            continue
        for source in group:
            if source["url"] not in seen:
                seen.add(source["url"])
                sources.append(source)
    return sources[:7]
