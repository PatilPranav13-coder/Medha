"""Manual live search command: python -m websearch "your question"."""
from __future__ import annotations

import argparse
import json
import sys

from websearch.service import SearchService


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the web and print a cited extractive answer.")
    parser.add_argument("query", nargs="+", help="Question to search for")
    args = parser.parse_args()
    result = SearchService().search(" ".join(args.query), "always")
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(json.dumps(source, ensure_ascii=False))
    if result.warning:
        print(f"\nSearch status: {result.warning}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
