"""Bounded public-page fetcher with DNS, redirect, MIME, robots, and cache guards."""
from __future__ import annotations

import ipaddress
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import OrderedDict
from dataclasses import dataclass
from email.message import Message as EmailMessage
from threading import Lock
from typing import Callable

from websearch.providers import USER_AGENT

MAX_PAGE_BYTES = 1_000_000
ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}


class PageFetchError(RuntimeError):
    """A page was invalid, private, disallowed by robots, or unsafe to parse."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    text: str
    content_type: str


def resolve_public_addresses(host: str, port: int) -> list[str]:
    try:
        rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise PageFetchError("Page host could not be resolved.") from exc
    return list(dict.fromkeys(row[4][0].split("%", 1)[0] for row in rows))


def validate_public_url(url: str, resolver: Callable[[str, int], list[str]] = resolve_public_addresses) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise PageFetchError("Page URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise PageFetchError("Only http and https page URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise PageFetchError("Page URLs with credentials are not allowed.")
    if port is not None and port not in {80, 443}:
        raise PageFetchError("Non-standard page ports are not allowed.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise PageFetchError("Local hostnames are not allowed.")
    chosen_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [str(literal)]
    except ValueError:
        addresses = resolver(hostname, chosen_port)
    if not addresses:
        raise PageFetchError("Page host has no public address.")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as exc:
            raise PageFetchError("Page host resolved to an invalid address.") from exc
        mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
        if not ip.is_global or (mapped is not None and not mapped.is_global):
            raise PageFetchError("Page host resolved to a non-public address.")
    return parsed


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 3

    def __init__(self, fetcher: "PublicPageFetcher") -> None:
        super().__init__()
        self.fetcher = fetcher

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.fetcher.validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicPageFetcher:
    def __init__(
        self,
        user_agent: str = USER_AGENT,
        timeout: float = 3.0,
        cache_ttl_seconds: int = 1800,
        cache_capacity: int = 256,
        max_bytes: int = MAX_PAGE_BYTES,
        resolver: Callable[[str, int], list[str]] = resolve_public_addresses,
        opener: Callable[..., object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_capacity = cache_capacity
        self.max_bytes = min(max_bytes, MAX_PAGE_BYTES)
        self.resolver = resolver
        self.clock = clock
        self._cache: OrderedDict[str, tuple[float, FetchedPage]] = OrderedDict()
        self._robots_cache: dict[str, tuple[float, str | None, bool]] = {}
        self._lock = Lock()
        if opener is None:
            handlers = [urllib.request.ProxyHandler({}), _SafeRedirectHandler(self)]
            self._opener = urllib.request.build_opener(*handlers).open
        else:
            self._opener = opener

    def validate_url(self, url: str) -> urllib.parse.SplitResult:
        return validate_public_url(url, self.resolver)

    def _open(self, url: str, timeout: float, byte_limit: int) -> tuple[bytes, str, str, str]:
        self.validate_url(url)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "text/html, text/plain, application/xhtml+xml"})
        try:
            response = self._opener(request, timeout=min(self.timeout, timeout))
            with response:
                headers = getattr(response, "headers", EmailMessage())
                content_type = headers.get_content_type() if hasattr(headers, "get_content_type") else ""
                charset = headers.get_content_charset() if hasattr(headers, "get_content_charset") else None
                length = headers.get("Content-Length") if hasattr(headers, "get") else None
                if length:
                    try:
                        if int(length) > byte_limit:
                            raise PageFetchError("Page exceeded the size limit.")
                    except ValueError:
                        raise PageFetchError("Page returned an invalid content length.")
                body = response.read(byte_limit + 1)
                final_url = getattr(response, "url", url)
        except urllib.error.HTTPError as exc:
            raise PageFetchError(f"Page returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if isinstance(exc, PageFetchError):
                raise
            raise PageFetchError("Page could not be fetched.") from exc
        if len(body) > byte_limit:
            raise PageFetchError("Page exceeded the size limit.")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise PageFetchError("Page content type is not supported.")
        # Recheck the final address even when an injected transport handled redirects.
        self.validate_url(final_url)
        return body, content_type, final_url, charset or "utf-8"

    def _robots_allowed(self, url: str, timeout: float) -> bool:
        parsed = self.validate_url(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname or ""
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        origin = f"{parsed.scheme}://{netloc}"
        with self._lock:
            cached = self._robots_cache.get(origin)
            if cached and cached[0] > self.clock():
                rules, available = cached[1], cached[2]
            else:
                rules, available = None, False
        if not available:
            robots_url = origin + "/robots.txt"
            try:
                body, content_type, _, charset = self._open(robots_url, timeout, 100_000)
                if content_type not in {"text/plain", "text/html"}:
                    return False
                rules = body.decode(charset, errors="replace")
                available = True
            except PageFetchError as exc:
                # RFC behavior treats a missing robots file as no restrictions; server errors fail closed.
                if "HTTP 404" in str(exc):
                    rules, available = "", True
                else:
                    return False
            with self._lock:
                self._robots_cache[origin] = (self.clock() + self.cache_ttl_seconds, rules, available)
                if len(self._robots_cache) > min(max(1, self.cache_capacity), 256):
                    self._robots_cache.pop(next(iter(self._robots_cache)))
        if rules is None:
            return False
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(origin + "/robots.txt")
        parser.parse(rules.splitlines())
        return parser.can_fetch(self.user_agent, url)

    def fetch(self, url: str, timeout: float | None = None) -> FetchedPage:
        self.validate_url(url)
        now = self.clock()
        with self._lock:
            cached = self._cache.get(url)
            if cached and cached[0] > now:
                self._cache.move_to_end(url)
                return cached[1]
            if cached:
                del self._cache[url]
        budget = min(timeout or self.timeout, self.timeout)
        started = self.clock()
        # Reserve most of the caller's short search budget for the page itself.
        allowed = self._robots_allowed(url, min(0.5, budget * 0.3))
        if not allowed:
            raise PageFetchError("The page disallows fetching or its robots policy could not be checked.")
        remaining = budget - (self.clock() - started)
        if remaining <= 0:
            raise PageFetchError("Page fetch deadline expired while checking robots policy.")
        body, content_type, final_url, charset = self._open(url, remaining, self.max_bytes)
        try:
            # Pages often declare a charset; honor it while replacing malformed bytes safely.
            if content_type in {"text/html", "application/xhtml+xml"}:
                from websearch.ranking import extract_html

                try:
                    decoded = body.decode(charset, errors="replace")
                except LookupError:
                    decoded = body.decode("utf-8", errors="replace")
                text, title = extract_html(decoded)
            else:
                try:
                    text = body.decode(charset, errors="replace")
                except LookupError:
                    text = body.decode("utf-8", errors="replace")
                title = urllib.parse.urlsplit(final_url).hostname or "Web page"
        except Exception as exc:
            if isinstance(exc, PageFetchError):
                raise
            raise PageFetchError("Page text could not be extracted.") from exc
        page = FetchedPage(final_url, title[:300], " ".join(text.split())[:200_000], content_type)
        with self._lock:
            self._cache[url] = (self.clock() + self.cache_ttl_seconds, page)
            self._cache.move_to_end(url)
            while len(self._cache) > self.cache_capacity:
                self._cache.popitem(last=False)
        return page
