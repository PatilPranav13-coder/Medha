from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import (
    deduplicate_jsonl, ingest_local_repository, inspect_repository,
    load_source_manifest, make_evaluation_record, make_preference_record,
    promote_reviewed_candidate,
)
from .github import discover_repositories, download_selected_repository
from .sandbox import verify_in_docker
from .quality import score_sft_candidate
from .scaling import build_progress_report, load_target_plan
from .shards import build_shards


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medha-data-factory",
        description="Bounded, local-first, licence-aware software engineering data ingestion.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Inspect a local repository without ingesting it.")
    inspect.add_argument("--repo", required=True, type=Path)

    ingest = commands.add_parser("ingest", help="Create pretraining records and review candidates.")
    ingest.add_argument("--repo", required=True, type=Path)
    ingest.add_argument("--manifest", required=True, type=Path)
    ingest.add_argument("--source-id", required=True)
    ingest.add_argument("--out", required=True, type=Path)
    ingest.add_argument("--max-files", type=int, default=1000)
    ingest.add_argument("--max-file-bytes", type=int, default=256_000)
    ingest.add_argument("--max-total-bytes", type=int, default=25_000_000)
    ingest.add_argument("--seed", type=int, default=42)

    discover = commands.add_parser("discover", help="Search public GitHub metadata only; does not download code.")
    discover.add_argument("--query", required=True)
    discover.add_argument("--limit", type=int, default=20)
    discover.add_argument("--allowed-license", nargs="*", default=None)
    discover.add_argument("--out", type=Path)

    harvest = commands.add_parser("harvest", help="Download a bounded subset from one approved GitHub repository.")
    harvest.add_argument("--repo", required=True)
    harvest.add_argument("--manifest", required=True, type=Path)
    harvest.add_argument("--source-id", required=True)
    harvest.add_argument("--out", required=True, type=Path)
    harvest.add_argument("--max-files", type=int, default=1000)
    harvest.add_argument("--max-file-bytes", type=int, default=256_000)
    harvest.add_argument("--max-total-bytes", type=int, default=25_000_000)
    harvest.add_argument("--seed", type=int, default=42)

    verify = commands.add_parser("verify", help="Explicit test execution in a restricted Docker container.")
    verify.add_argument("--repo", required=True, type=Path)
    verify.add_argument("--image", required=True)
    verify.add_argument("--test-command", required=True)
    verify.add_argument("--timeout", type=int, default=180)
    verify.add_argument("--out", required=True, type=Path)

    promote = commands.add_parser("promote-sft", help="Promote one reviewed candidate to SFT.")
    promote.add_argument("--candidate", required=True, type=Path)
    promote.add_argument("--review", required=True, type=Path)
    promote.add_argument("--repository-index", required=True, type=Path)
    promote.add_argument("--verification", type=Path)
    promote.add_argument("--out", required=True, type=Path)

    preference = commands.add_parser("make-preference", help="Convert one non-tied human comparison to a preference pair.")
    preference.add_argument("--input", required=True, type=Path, help="JSON object with prompt, response_1/2, strength and reason.")
    preference.add_argument("--manifest", required=True, type=Path)
    preference.add_argument("--source-id", required=True)
    preference.add_argument("--out", required=True, type=Path)

    evaluation = commands.add_parser("make-evaluation", help="Register one sealed test-only evaluation task.")
    evaluation.add_argument("--input", required=True, type=Path)
    evaluation.add_argument("--manifest", required=True, type=Path)
    evaluation.add_argument("--source-id", required=True)
    evaluation.add_argument("--out", required=True, type=Path)

    progress = commands.add_parser("progress", help="Report progress against the 100M engineering-example quotas.")
    progress.add_argument("--plan", required=True, type=Path)
    progress.add_argument("--counts", required=True, type=Path)
    progress.add_argument("--out", type=Path)

    score = commands.add_parser("score-example", help="Score one proposed SFT example against the quality gate.")
    score.add_argument("--candidate", required=True, type=Path)
    score.add_argument("--review", required=True, type=Path)
    score.add_argument("--repository-index", required=True, type=Path)
    score.add_argument("--out", type=Path)

    shard = commands.add_parser("shard-jsonl", help="Stream-deduplicate and create bounded train/validation/test JSONL shards.")
    shard.add_argument("--input", required=True, nargs="+", type=Path)
    shard.add_argument("--out", required=True, type=Path)
    shard.add_argument("--examples-per-shard", type=int, default=100000)

    dedupe = commands.add_parser("deduplicate", help="Exact-dedupe JSONL records and preserve duplicate provenance.")
    dedupe.add_argument("--input", required=True, nargs="+", type=Path)
    dedupe.add_argument("--out", required=True, type=Path)
    dedupe.add_argument("--ledger", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_repository(args.repo)
        elif args.command == "discover":
            result = discover_repositories(args.query, args.limit, set(args.allowed_license) if args.allowed_license else None)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif args.command == "progress":
            plan = load_target_plan(args.plan)
            counts = json.loads(args.counts.read_text(encoding="utf-8"))
            result = build_progress_report(plan, counts)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        elif args.command == "score-example":
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            review = json.loads(args.review.read_text(encoding="utf-8"))
            index = json.loads(args.repository_index.read_text(encoding="utf-8"))
            result = score_sft_candidate(candidate, review, index)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        elif args.command == "shard-jsonl":
            result = build_shards(args.input, args.out, examples_per_shard=args.examples_per_shard)
        elif args.command == "deduplicate":
            result = deduplicate_jsonl(args.input, args.out, args.ledger)
        elif args.command == "verify":
            result = verify_in_docker(args.repo, args.image, args.test_command, timeout_seconds=args.timeout)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        elif args.command == "promote-sft":
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            review = json.loads(args.review.read_text(encoding="utf-8"))
            if args.verification:
                review["verification_report"] = json.loads(args.verification.read_text(encoding="utf-8"))
            index = json.loads(args.repository_index.read_text(encoding="utf-8"))
            record = promote_reviewed_candidate(candidate, review, index)
            _write_jsonl(args.out, record)
            result = {"promoted": True, "record_id": record["record_id"], "output": str(args.out)}
        elif args.command == "make-preference":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            source = load_source_manifest(args.manifest, args.source_id)
            record = make_preference_record(
                record_id=payload.get("record_id", ""),
                prompt=payload["prompt"], response_1=payload["response_1"],
                response_2=payload["response_2"],
                preference_strength=payload["preference_strength"],
                preference_reason=payload["preference_reason"], source=source,
                domain_tags=payload.get("domain_tags"),
                task_type=payload.get("task_type", "code_generation"),
            )
            _write_jsonl(args.out, record)
            result = {"created": True, "record_id": record["record_id"], "source_split": record["source_split"]}
        elif args.command == "make-evaluation":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            source = load_source_manifest(args.manifest, args.source_id)
            source["dataset_split"] = "test"
            record = make_evaluation_record(source=source, **payload)
            _write_jsonl(args.out, record)
            result = {"created": True, "task_id": record["task_id"], "split": "test"}
        elif args.command == "harvest":
            source = load_source_manifest(args.manifest, args.source_id)
            stage = args.out / "repository_snapshot"
            download = download_selected_repository(
                args.repo, source, stage, max_files=args.max_files,
                max_file_bytes=args.max_file_bytes, max_total_bytes=args.max_total_bytes,
            )
            source = dict(source, revision=download["revision"])
            indexed = ingest_local_repository(
                stage, source, args.out / "processed", seed=args.seed,
                max_files=args.max_files, max_file_bytes=args.max_file_bytes,
                max_total_bytes=args.max_total_bytes,
            )
            result = {"download": download, "ingestion": indexed}
        else:
            source = load_source_manifest(args.manifest, args.source_id)
            result = ingest_local_repository(
                args.repo, source, args.out, seed=args.seed,
                max_files=args.max_files, max_file_bytes=args.max_file_bytes,
                max_total_bytes=args.max_total_bytes,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, PermissionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
