"""Load a local JSONL dataset only when its source and licence are declared."""
from __future__ import annotations

import json
from pathlib import Path


def load_public_jsonl(path: str | Path, licence: str) -> list[dict]:
    if not licence.strip():
        raise ValueError("A licence note is required before loading an external dataset.")
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL task at line {line_number}.") from exc
        if not isinstance(task, dict) or not all(key in task for key in ("id", "category", "prompt", "checker")):
            raise ValueError(f"Task at line {line_number} needs id, category, prompt, and checker fields.")
        task["dataset_licence"] = licence.strip()
        result.append(task)
    return result
