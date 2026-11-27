"""Offline-testable web search, retrieval, and citation helpers for Medha."""

from websearch.answer import SearchAnswer, SearchMode, should_search
from websearch.service import SearchService

__all__ = ["SearchAnswer", "SearchMode", "SearchService", "should_search"]
