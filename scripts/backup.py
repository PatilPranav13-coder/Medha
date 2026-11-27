"""Create and restore verified SQLite backups without replacing existing files."""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri_path = quote(path.resolve().as_posix(), safe="/:")
    return sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=30)


def _copy_and_verify(source_path: Path, target_path: Path) -> None:
    source = _readonly_connection(source_path)
    target = sqlite3.connect(target_path, timeout=30)
    try:
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise sqlite3.DatabaseError("SQLite integrity check failed after copying.")
    finally:
        target.close()
        source.close()


def create_backup(database: Path, destination: Path, keep: int = 7) -> Path:
    """Take a consistent snapshot and rotate only files created by this script."""
    if keep < 1:
        raise ValueError("At least one backup must be retained.")
    if not database.is_file():
        raise FileNotFoundError("Database file does not exist.")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = destination / f"medha-{stamp}.sqlite3"
    try:
        _copy_and_verify(database, output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    backups = sorted(destination.glob("medha-*.sqlite3"), key=lambda item: item.name, reverse=True)
    for expired in backups[keep:]:
        expired.unlink()
    return output


def restore_backup(backup: Path, target: Path) -> None:
    """Restore to a new path so a typo can never overwrite a live database."""
    if not backup.is_file():
        raise FileNotFoundError("Backup file does not exist.")
    if target.exists():
        raise FileExistsError("Restore target already exists; choose a new path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _copy_and_verify(backup, target)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("create", help="create and rotate verified backups")
    make.add_argument("--database", type=Path, required=True)
    make.add_argument("--destination", type=Path, required=True)
    make.add_argument("--keep", type=int, default=7)
    restore = subparsers.add_parser("restore", help="restore to a new, unused path")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(create_backup(args.database, args.destination, args.keep))
    else:
        restore_backup(args.backup, args.target)
        print(f"Restored and verified {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
