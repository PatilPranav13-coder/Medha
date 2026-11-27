"""SQLite connection and migration layer used by the API repository."""
from __future__ import annotations

import hashlib
import hmac
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


MIGRATIONS = Path(__file__).resolve().parent / "migrations"
BOOLEAN_RUNTIME_FLAGS = frozenset({"maintenance", "read_only", "signups_enabled"})
RUNTIME_FLAGS = BOOLEAN_RUNTIME_FLAGS | {"max_users"}


class Database:
    """Open short-lived SQLite connections with WAL and foreign keys enabled."""

    def __init__(self, path: str | Path, migrations_dir: Path = MIGRATIONS) -> None:
        self.path = str(path)
        self.migrations_dir = migrations_dir

    def _connect(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        if self.path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Use an immediate transaction for race-safe invite and user-cap checks."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for migration in sorted(self.migrations_dir.glob("[0-9][0-9][0-9]_*.sql")):
                version_text = migration.name.split("_", 1)[0]
                if not version_text.isdigit():
                    raise ValueError(f"Invalid migration filename: {migration.name}")
                version = int(version_text)
                if version in applied:
                    continue
                # Execute complete statements individually so migration application is atomic.
                connection.execute("BEGIN IMMEDIATE")
                try:
                    statement = ""
                    for line in migration.read_text(encoding="utf-8").splitlines():
                        statement += line + "\n"
                        if sqlite3.complete_statement(statement):
                            if statement.strip():
                                connection.execute(statement)
                            statement = ""
                    if statement.strip():
                        raise ValueError(f"Incomplete SQL statement in {migration.name}")
                    connection.execute(
                        "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                        (version, migration.name),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def create_invite_code(
        self,
        code: str,
        max_uses: int = 1,
        expires_at: float | None = None,
        created_by: str | None = None,
    ) -> None:
        if max_uses < 1:
            raise ValueError("An invite code must allow at least one use.")
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO invite_codes(code_hash, max_uses, expires_at, created_by) VALUES (?, ?, ?, ?)",
                (code_hash, max_uses, expires_at, created_by),
            )

    def prune_events(self, retention_days: int, now: float | None = None) -> int:
        """Delete only metadata events older than the configured retention period."""
        if retention_days < 1:
            raise ValueError("Event retention must be at least one day.")
        current = time.time() if now is None else now
        cutoff = current - retention_days * 24 * 60 * 60
        with self.transaction() as connection:
            result = connection.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))
        return result.rowcount

    def get_runtime_flag(self, name: str, default: bool | int) -> bool | int:
        """Read a validated operator flag, falling back to the environment setting."""
        if name not in RUNTIME_FLAGS:
            raise ValueError("Unknown runtime flag.")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT flag_value FROM runtime_flags WHERE flag_name=?", (name,)
            ).fetchone()
        if row is None:
            return default
        value = int(row["flag_value"])
        if name in BOOLEAN_RUNTIME_FLAGS:
            if value not in {0, 1}:
                raise ValueError("Stored boolean runtime flag is invalid.")
            return bool(value)
        if value < 1:
            raise ValueError("Stored user cap must be positive.")
        return value

    def set_runtime_flag(self, name: str, value: bool | int) -> None:
        """Persist only the small allowlist that the admin CLI can change."""
        if name not in RUNTIME_FLAGS:
            raise ValueError("Unknown runtime flag.")
        if name in BOOLEAN_RUNTIME_FLAGS:
            if type(value) is not bool:
                raise ValueError("Boolean runtime flags require true or false.")
            stored_value = int(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("The user cap must be a positive integer.")
            stored_value = value
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO runtime_flags(flag_name, flag_value) VALUES (?, ?) "
                "ON CONFLICT(flag_name) DO UPDATE SET flag_value=excluded.flag_value",
                (name, stored_value),
            )

    def counters(self) -> dict[str, int]:
        """Expose aggregate counts only; the admin page never receives user content."""
        tables = ("users", "sessions", "conversations", "messages", "memories", "uploads", "feedback", "events")
        with self.connection() as connection:
            counts = {
                name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                for name in tables
            }
            counts["disabled_users"] = int(
                connection.execute("SELECT COUNT(*) FROM users WHERE disabled=1").fetchone()[0]
            )
        return counts


def token_digest(token: str) -> str:
    """Store a one-way digest, never a browser bearer token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def compare_digest(left: str | bytes, right: str | bytes) -> bool:
    """Keep secrets comparisons constant-time even when sizes differ."""
    left_bytes = left.encode("utf-8") if isinstance(left, str) else left
    right_bytes = right.encode("utf-8") if isinstance(right, str) else right
    return hmac.compare_digest(left_bytes, right_bytes)
