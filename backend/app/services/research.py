from __future__ import annotations

import httpx


async def search_academic(query: str, limit: int = 6) -> list[dict[str, str]]:
    if not query.strip():
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False, headers={"User-Agent": "Medha-Research/1.0 (academic reference lookup)"}) as client:
            response = await client.get("https://api.crossref.org/works", params={"query": query[:300], "rows": str(limit), "select": "DOI,title,author,published,URL,container-title"})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    items = ((payload.get("message") or {}).get("items") or [])
    references = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        title_values = item.get("title") or []
        title = str(title_values[0]) if title_values else "Untitled work"
        authors = []
        for author in (item.get("author") or [])[:4]:
            name = " ".join(filter(None, [author.get("given"), author.get("family")]))
            if name:
                authors.append(name)
        date_parts = (((item.get("published") or {}).get("date-parts") or [[None]])[0])
        year = str(date_parts[0]) if date_parts and date_parts[0] else ""
        url = item.get("URL", "")
        if not url.startswith("https://doi.org/") and item.get("DOI"):
            url = f"https://doi.org/{item['DOI']}"
        if not url.startswith("https://"):
            continue
        venue_values = item.get("container-title") or []
        venue = str(venue_values[0]) if venue_values else ""
        references.append({"title": title[:220], "url": url, "authors": ", ".join(authors), "year": year, "venue": venue, "kind": "academic"})
    return references
