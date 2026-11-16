"""Streaming, SQLite-indexed JSONL deduplication and shard writer.

The two-pass design keeps record bodies out of RAM and ensures a duplicate found
in train and test is written only to the highest-priority split (test first).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .core import _record_fingerprint

SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}
SPLIT_BITS = {"train": 1, "validation": 2, "test": 4}


def _iter_rows(paths: list[Path]) -> Iterator[tuple[Path, int, dict[str, Any], str, str]]:
    for path in paths:
        source_path = path.expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {source_path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Each JSONL row must be an object: {source_path}:{line_number}")
                metadata = row.get("source_metadata") or {}
                split = row.get("dataset_split") or row.get("split") or metadata.get("dataset_split") or metadata.get("split")
                if split not in SPLIT_PRIORITY:
                    raise ValueError(f"Missing or invalid dataset split in {source_path}:{line_number}")
                yield source_path, line_number, row, str(split), _record_fingerprint(row)


def build_shards(
    input_paths: list[Path],
    output_dir: Path,
    *,
    examples_per_shard: int = 100_000,
) -> dict[str, Any]:
    """Write bounded JSONL shards with memory-bounded deduplication.

    Input rows are read twice. A temporary SQLite index stores only SHA-256
    fingerprints and split priority; full records are never accumulated in RAM.
    """
    if not input_paths:
        raise ValueError("At least one input JSONL path is required.")
    if type(examples_per_shard) is not int or not 1 <= examples_per_shard <= 1_000_000:
        raise ValueError("examples_per_shard must be between 1 and 1,000,000.")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    scanned = 0
    written = 0
    duplicate_rows = 0
    shard_manifest: list[dict[str, Any]] = []
    split_counts = {split: 0 for split in SPLIT_PRIORITY}
    active: dict[str, Any] = {}

    def close_shard(split: str) -> None:
        state = active.pop(split, None)
        if state is None:
            return
        state["handle"].close()
        shard_manifest.append({
            "split": split,
            "path": state["path"].name,
            "examples": state["examples"],
            "bytes": state["bytes"],
            "sha256": state["digest"].hexdigest(),
        })

    def write_row(split: str, row: dict[str, Any]) -> None:
        local_count = split_counts[split]
        shard_index = local_count // examples_per_shard
        state = active.get(split)
        if state is None or state["shard_index"] != shard_index:
            close_shard(split)
            path = output / f"{split}-{shard_index:05d}.jsonl"
            state = {
                "handle": path.open("wb"),
                "path": path,
                "shard_index": shard_index,
                "examples": 0,
                "bytes": 0,
                "digest": hashlib.sha256(),
            }
            active[split] = state
        payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        state["handle"].write(payload)
        state["digest"].update(payload)
        state["bytes"] += len(payload)
        state["examples"] += 1
        split_counts[split] += 1

    with tempfile.TemporaryDirectory(prefix=".medha-shard-index-", dir=str(output.parent)) as temp_dir:
        database = Path(temp_dir) / "fingerprints.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE fingerprints (fingerprint TEXT PRIMARY KEY, best_priority INTEGER NOT NULL, split_mask INTEGER NOT NULL)"
            )
            connection.execute("CREATE TABLE emitted (fingerprint TEXT PRIMARY KEY)")
            pending = 0

            # Pass one: record the best split for every unique content fingerprint.
            for _, _, _, split, fingerprint in _iter_rows(input_paths):
                scanned += 1
                priority = SPLIT_PRIORITY[split]
                bit = SPLIT_BITS[split]
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?)",
                    (fingerprint, priority, bit),
                )
                if cursor.rowcount == 0:
                    duplicate_rows += 1
                    connection.execute(
                        "UPDATE fingerprints SET best_priority = MIN(best_priority, ?), split_mask = split_mask | ? WHERE fingerprint = ?",
                        (priority, bit, fingerprint),
                    )
                pending += 1
                if pending >= 10_000:
                    connection.commit()
                    pending = 0
            connection.commit()

            # Pass two: emit only the preferred split copy, once.
            for _, _, row, split, fingerprint in _iter_rows(input_paths):
                priority = SPLIT_PRIORITY[split]
                preferred = connection.execute(
                    "SELECT best_priority FROM fingerprints WHERE fingerprint = ?",
                    (fingerprint,),
                ).fetchone()
                if preferred is None or preferred[0] != priority:
                    continue
                try:
                    connection.execute("INSERT INTO emitted VALUES (?)", (fingerprint,))
                except sqlite3.IntegrityError:
                    continue
                write_row(split, row)
                written += 1

            for split in list(active):
                close_shard(split)
            connection.commit()
            cross_split_duplicates = connection.execute(
                "SELECT COUNT(*) FROM fingerprints WHERE (split_mask & (split_mask - 1)) != 0"
            ).fetchone()[0]
            unique_fingerprints = connection.execute(
                "SELECT COUNT(*) FROM fingerprints"
            ).fetchone()[0]
        finally:
            connection.close()

    manifest = {
        "manifest_version": "1.0.0",
        "algorithm": "two-pass-streaming-sqlite-fingerprint-dedupe",
        "input_files": [str(path) for path in input_paths],
        "input_rows": scanned,
        "unique_fingerprints": unique_fingerprints,
        "examples_written": written,
        "duplicate_rows_removed": duplicate_rows,
        "cross_split_duplicate_fingerprints": cross_split_duplicates,
        "split_counts": split_counts,
        "examples_per_shard": examples_per_shard,
        "shard_count": len(shard_manifest),
        "shards": shard_manifest,
        "memory_strategy": "record bodies streamed; fingerprints indexed in temporary SQLite database",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
