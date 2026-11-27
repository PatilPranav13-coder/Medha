"""Fail CI when tracked files contain recognizable live credential formats."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"sk-[A-Za-z0-9]{32,}"),
)


def scan_tracked_files() -> list[str]:
    paths = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    findings = []
    for raw_path in paths.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        try:
            contents = path.read_bytes()
        except OSError:
            continue
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append(f"{path}:{line_number}")
    return findings


def main() -> int:
    findings = scan_tracked_files()
    if findings:
        print("Possible credentials found (matched values were withheld):", file=sys.stderr)
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print("No common credential formats found in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
