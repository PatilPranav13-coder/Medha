"""Core repository ingestion, provenance, deduplication and record validation."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "medha_dataset_starter_pack" / "schemas"
PIPELINE_VERSION = "medha-software-factory/0.3.0"
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "vendor",
    "dist", "build", "coverage", ".next", ".nuxt", "__pycache__",
    ".idea", ".vscode", "target", "Pods", "DerivedData",
}
SPECIAL_TEXT_FILES = {
    "readme", "license", "licence", "notice", "copying", "package.json",
    "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "build.gradle.kts", "gemfile",
    "composer.json", "pubspec.yaml", "makefile", "dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "openapi.yaml",
    "openapi.yml", "openapi.json", "schema.sql",
}
EXTENSIONS = {
    ".py":"python", ".pyi":"python", ".js":"javascript", ".jsx":"javascript",
    ".mjs":"javascript", ".cjs":"javascript", ".ts":"typescript", ".tsx":"typescript",
    ".java":"java", ".kt":"kotlin", ".kts":"kotlin", ".swift":"swift",
    ".c":"c", ".h":"c", ".cc":"cpp", ".cpp":"cpp", ".cxx":"cpp", ".hpp":"cpp",
    ".cs":"csharp", ".go":"go", ".rs":"rust", ".rb":"ruby", ".php":"php",
    ".dart":"dart", ".scala":"scala", ".sc":"scala", ".sql":"sql",
    ".html":"html", ".htm":"html", ".css":"css", ".scss":"scss",
    ".vue":"vue", ".svelte":"svelte", ".md":"markdown", ".mdx":"markdown",
    ".rst":"restructuredtext", ".yaml":"yaml", ".yml":"yaml", ".json":"json",
    ".toml":"toml", ".xml":"xml", ".sh":"shell", ".bash":"shell",
    ".ps1":"powershell", ".dockerfile":"dockerfile", ".proto":"protobuf",
    ".graphql":"graphql", ".gql":"graphql",
}
SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic_secret", re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]")),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_split(group_id: str, seed: int = 42) -> str:
    """Assign entire repositories/problem families to deterministic 90/5/5 splits."""
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1000
    return "train" if bucket < 900 else ("validation" if bucket < 950 else "test")


def detect_language(path: Path) -> str | None:
    name = path.name.lower()
    if name in SPECIAL_TEXT_FILES or name.startswith("readme.") or name.startswith("license.") or name.startswith("licence."):
        return "text"
    return EXTENSIONS.get(path.suffix.lower())


def excluded_reason(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if parts & EXCLUDED_DIRS:
        return "excluded_dependency_or_build_directory"
    name = path.name.lower()
    if name in {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "credentials", "secrets.json"}:
        return "sensitive_filename"
    if name.endswith((".min.js", ".min.css", ".map", ".lock")):
        return "generated_or_lock_file"
    if path.suffix.lower() in {".png",".jpg",".jpeg",".gif",".webp",".ico",".pdf",".zip",".gz",".tar",".7z",".exe",".dll",".so",".dylib",".bin",".pt",".pth",".onnx",".mp4",".mov",".woff",".woff2",".ttf",".otf"}:
        return "binary_or_model_artifact"
    return None


def secret_type(text: str) -> str | None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return label
    return None


def validate_source_policy(source: dict[str, Any]) -> None:
    required = ("source_id", "repository_url", "declared_license", "underlying_license",
                "usage_status", "license_evidence_url", "reviewed_at", "reviewer", "license_scope")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise PermissionError(f"Source manifest is incomplete: {', '.join(missing)}")
    if source["usage_status"] not in {"approved", "approved_with_attribution"}:
        raise PermissionError(f"Source {source['source_id']} is not approved for ingestion.")
    if source["underlying_license"].strip().lower() in {"unknown", "noassertion", "other", "none"}:
        raise PermissionError("Underlying file/code licence is unknown; ingestion is blocked.")
    if source["license_scope"] not in {"repository-reviewed", "file-reviewed"}:
        raise PermissionError("Licence scope must be repository-reviewed or file-reviewed.")
    if not source["license_evidence_url"].startswith(("https://", "http://")):
        raise PermissionError("A public licence evidence URL is required.")


def load_source_manifest(path: Path, source_id: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "1.0.0":
        raise ValueError("Unsupported source manifest version; expected 1.0.0.")
    for source in manifest.get("sources", []):
        if source.get("source_id") == source_id:
            validate_source_policy(source)
            return source
    raise KeyError(f"Source {source_id!r} is not in the manifest.")


def _git_revision(root: Path) -> str | None:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def validate_pretraining_record(record: dict[str, Any]) -> None:
    schema_path = SCHEMA_ROOT / "pretraining.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)


def ingest_local_repository(
    repository_path: Path,
    source: dict[str, Any],
    output_dir: Path,
    *,
    seed: int = 42,
    max_files: int = 1000,
    max_file_bytes: int = 256_000,
    max_total_bytes: int = 25_000_000,
) -> dict[str, Any]:
    """Index a repository snapshot; never executes files or accesses the network."""
    validate_source_policy(source)
    root = repository_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not 1 <= max_files <= 5000 or not 1 <= max_file_bytes <= 2_000_000 or not 1 <= max_total_bytes <= 200_000_000:
        raise ValueError("Requested ingestion limits exceed the hard safety ceilings.")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    repository_id = source["repository_url"].rstrip("/").lower()
    revision = source.get("revision") or _git_revision(root)
    split = stable_split(repository_id, seed)
    seen: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    total = 0
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            rejections.append({"path": str(path), "reason": "symlink_or_path_escape"})
            continue
        rel_path = Path(relative)
        reason = excluded_reason(rel_path)
        if reason:
            rejections.append({"path": relative, "reason": reason})
            continue
        language = detect_language(rel_path)
        if language is None:
            continue
        scanned += 1
        if scanned > max_files:
            rejections.append({"path": relative, "reason": "max_files_limit_reached"})
            break
        size = resolved.stat().st_size
        if size <= 0 or size > max_file_bytes:
            rejections.append({"path": relative, "reason": "file_size_limit_or_empty"})
            continue
        if total + size > max_total_bytes:
            rejections.append({"path": relative, "reason": "max_total_bytes_limit_reached"})
            break
        raw = resolved.read_bytes()
        if b"\x00" in raw:
            rejections.append({"path": relative, "reason": "binary_content"})
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            rejections.append({"path": relative, "reason": "non_utf8_content"})
            continue
        secret = secret_type(text)
        if secret:
            rejections.append({"path": relative, "reason": "secret_pattern_detected", "secret_type": secret})
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        content_hash = sha256_text(normalized)
        if content_hash in seen:
            rejections.append({"path": relative, "reason": "exact_duplicate", "canonical_path": seen[content_hash]["path"]})
            continue
        seen[content_hash] = {"path": relative}
        total += len(raw)
        is_code = language not in {"text", "markdown", "restructuredtext"}
        tags = ["software_engineering"]
        if is_code:
            tags += ["programming", language]
        else:
            tags += ["documentation"]
        record = {
            "schema_version": "1.0.0",
            "record_id": f"{repository_id}:{relative}:{content_hash[:12]}",
            "text": normalized,
            "content_type": "code" if is_code else "text",
            "dataset_split": split,
            "domain_tags": tags,
            "language": language,
            "source_metadata": {
                "repository": source["repository_url"],
                "relative_path": relative,
                "repository_revision": revision,
                "dataset_split": split,
                "file_bytes": len(raw),
            },
            "provenance": {
                "source_dataset": f"github:{source['source_id']}",
                "source_revision": revision,
                "source_config": None,
                "source_split": "repository_snapshot",
                "source_record_id": relative,
                "source_url": source["repository_url"],
                "source_file": relative,
                "source_date": None,
                "declared_license": source["declared_license"],
                "underlying_license": source["underlying_license"],
                "usage_status": source["usage_status"],
                "content_sha256": content_hash,
                "duplicate_cluster_id": None,
                "transformation_version": "1.0.0",
                "pipeline_version": PIPELINE_VERSION,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "selection_seed": seed,
                "source_metadata": {"license_evidence_url": source["license_evidence_url"]},
                "attribution_text": source.get("attribution_text"),
            },
        }
        validate_pretraining_record(record)
        records.append(record)

    for partition in ("train", "validation", "test"):
        _write_jsonl(output / f"pretraining_{partition}.jsonl",
                     (r for r in records if r["source_metadata"]["dataset_split"] == partition))
    _write_jsonl(output / "rejections.jsonl", rejections)

    index = {
        "repository_id": repository_id,
        "repository_url": source["repository_url"],
        "revision": revision,
        "declared_license": source["declared_license"],
        "underlying_license": source["underlying_license"],
        "usage_status": source["usage_status"],
        "license_scope": source["license_scope"],
        "license_evidence_url": source["license_evidence_url"],
        "reviewer": source["reviewer"],
        "reviewed_at": source["reviewed_at"],
        "dataset_split": split,
        "seed": seed,
        "files_indexed": len(records),
        "bytes_indexed": total,
        "pipeline_version": PIPELINE_VERSION,
    }
    (output / "repository_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    candidates = [{
        "candidate_id": f"{repository_id}:repository-understanding",
        "task_type": "instruction",
        "prompt": f"Explain the architecture and responsibilities of the repository at {source['repository_url']}. Identify its entry points, modules, data flow, dependencies, tests and likely extension points.",
        "context_paths": [r["source_metadata"]["relative_path"] for r in records[:100]],
        "training_eligible": False,
        "requires_human_review": True,
        "verification_required": True,
        "dataset_split": split,
        "source_repository": repository_id,
        "source_revision": revision,
    }]
    _write_jsonl(output / "task_candidates.jsonl", candidates)
    report = {
        "repository": repository_id, "revision": revision, "dataset_split": split,
        "files_written": len(records), "records_written": len(records), "bytes_written": total,
        "duplicates_removed": sum(r["reason"] == "exact_duplicate" for r in rejections),
        "rejections": len(rejections), "execution_performed": False, "network_access_performed": False,
        "network_access_performed": False, "schema_validation": "passed",
        "output_dir": str(output),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def promote_reviewed_candidate(
    candidate: dict[str, Any], review: dict[str, Any], repository_index: dict[str, Any]
) -> dict[str, Any]:
    """Create an SFT record only after rights, human review and verification gates pass."""
    if repository_index.get("usage_status") not in {"approved", "approved_with_attribution"}:
        raise PermissionError("Repository rights are not approved.")
    if repository_index.get("dataset_split") == "test":
        raise PermissionError("Evaluation/test repository groups cannot be promoted into training.")
    validate_source_policy({
        "source_id": repository_index["repository_id"],
        "repository_url": repository_index["repository_url"],
        "declared_license": repository_index["declared_license"],
        "underlying_license": repository_index["underlying_license"],
        "usage_status": repository_index["usage_status"],
        "license_evidence_url": repository_index["license_evidence_url"],
        "reviewed_at": repository_index["reviewed_at"],
        "reviewer": repository_index["reviewer"],
        "license_scope": repository_index["license_scope"],
    })
    for field in ("human_reviewed", "safety_reviewed", "factuality_pass"):
        if review.get(field) is not True:
            raise PermissionError(f"Review gate failed: {field} must be true.")
    if review.get("requires_execution", True):
        verification = review.get("verification_report") or {}
        if verification.get("status") != "passed" or verification.get("exit_code") != 0:
            raise PermissionError("Executable task requires a successful isolated verification report.")
        if review.get("build_passed") is not True or int(review.get("tests_passed") or 0) < 1:
            raise PermissionError("Executable task requires passing build and at least one test.")
        if verification.get("network_enabled") is not False or verification.get("source_mount_read_only") is not True:
            raise PermissionError("Verification environment does not meet isolation requirements.")
    answer = (review.get("assistant_answer") or "").strip()
    prompt = (candidate.get("prompt") or "").strip()
    if not answer or not prompt:
        raise ValueError("Candidate prompt and reviewed assistant answer are required.")
    record_hash = sha256_text(prompt + "\n" + answer)
    record = {
        "schema_version": "1.0.0",
        "record_id": f"sft:{candidate['candidate_id']}:{record_hash[:12]}",
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        "task_type": candidate.get("task_type", "instruction"),
        "domain_tags": ["software_engineering", "repository_understanding"],
        "quality_checks": {
            "human_reviewed": True, "factuality_pass": True,
            "tests_passed": review.get("tests_passed"),
            "safety_reviewed": True,
            "notes": review.get("verification_notes", []),
        },
        "source_metadata": {
            "candidate_id": candidate["candidate_id"],
            "repository": repository_index["repository_url"],
            "revision": repository_index.get("revision"),
            "dataset_split": repository_index["dataset_split"],
            "reviewer_id": review.get("reviewer_id"),
            "reviewed_at": review.get("reviewed_at"),
            "verification_report": review.get("verification_report"),
        },
        "synthetic": False,
        "provenance": {
            "source_dataset": "medha-reviewed-software-tasks",
            "source_revision": repository_index.get("revision"),
            "source_config": None,
            "source_split": repository_index["dataset_split"],
            "source_record_id": candidate["candidate_id"],
            "source_url": repository_index["repository_url"],
            "source_file": None,
            "source_date": repository_index.get("reviewed_at"),
            "declared_license": repository_index["declared_license"],
            "underlying_license": repository_index["underlying_license"],
            "usage_status": repository_index["usage_status"],
            "content_sha256": record_hash,
            "duplicate_cluster_id": None,
            "transformation_version": "1.0.0",
            "pipeline_version": PIPELINE_VERSION,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "selection_seed": repository_index.get("seed"),
            "source_metadata": {"license_evidence_url": repository_index["license_evidence_url"]},
            "attribution_text": repository_index.get("attribution_text"),
        },
    }
    schema = json.loads((SCHEMA_ROOT / "sft.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)
    return record


def inspect_repository(repository_path: Path) -> dict[str, Any]:
    """Inspect repository structure and file types without running project code."""
    root = repository_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    language_counts: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    skipped = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            skipped += 1
            continue
        rel_path = Path(relative)
        if excluded_reason(rel_path):
            skipped += 1
            continue
        language = detect_language(rel_path)
        if language is None:
            continue
        size = resolved.stat().st_size
        language_counts[language] = language_counts.get(language, 0) + 1
        total_bytes += size
        files.append({"path": relative, "language": language, "bytes": size})
    return {
        "repository_path": str(root),
        "revision": _git_revision(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "excluded_or_unsafe_count": skipped,
        "languages": dict(sorted(language_counts.items())),
        "files": files[:500],
        "files_truncated": len(files) > 500,
        "execution_performed": False,
        "network_access_performed": False,
    }


def _record_fingerprint(record: dict[str, Any]) -> str:
    provenance = record.get("provenance") or {}
    fingerprint = provenance.get("content_sha256")
    if fingerprint and re.fullmatch(r"[0-9a-f]{64}", str(fingerprint)):
        return str(fingerprint)
    body = record.get("text")
    if body is None:
        body = json.dumps(record.get("messages") or record, ensure_ascii=False, sort_keys=True)
    return sha256_text(str(body))


def deduplicate_jsonl(
    input_paths: list[Path], output_path: Path, ledger_path: Path
) -> dict[str, Any]:
    """Exact-dedupe JSONL files; when duplicates cross splits, keep evaluation isolated."""
    groups: dict[str, list[tuple[dict[str, Any], str, int]]] = {}
    read_count = 0
    for source_path in input_paths:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                fingerprint = _record_fingerprint(row)
                split = row.get("dataset_split") or (row.get("source_metadata") or {}).get("dataset_split")
                # An explicitly held-out evaluation record must not be copied into training.
                priority = {"test": 0, "validation": 1, "train": 2}.get(split, 3)
                groups.setdefault(fingerprint, []).append((row, str(source_path), line_number))
                read_count += 1
    canonical: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for fingerprint, entries in groups.items():
        # For identical content, prefer the most isolated split (test > validation > train).
        selected = entries[0]
        for candidate in entries[1:]:
            old_split = selected[0].get("dataset_split") or (selected[0].get("source_metadata") or {}).get("dataset_split")
            new_split = candidate[0].get("dataset_split") or (candidate[0].get("source_metadata") or {}).get("dataset_split")
            order = {"test": 0, "validation": 1, "train": 2}
            if order.get(new_split, 3) < order.get(old_split, 3):
                selected = candidate
        canonical.append(selected[0])
        for row, source, line_number in entries:
            if (row, source, line_number) == selected:
                continue
            ledger.append({
                "content_sha256": fingerprint,
                "canonical_record_id": selected[0].get("record_id") or selected[0].get("task_id"),
                "duplicate_record_id": row.get("record_id") or row.get("task_id"),
                "duplicate_source": source,
                "duplicate_line": line_number,
                "duplicate_split": row.get("dataset_split") or (row.get("source_metadata") or {}).get("dataset_split"),
                "reason": "exact_duplicate_removed_to_prevent_split_leakage",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, canonical)
    _write_jsonl(ledger_path, ledger)
    return {
        "input_records": read_count,
        "unique_records": len(canonical),
        "duplicates_removed": len(ledger),
        "evaluation_priority": "test > validation > train",
        "output": str(output_path),
        "ledger": str(ledger_path),
    }
