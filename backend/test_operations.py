from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.database import Database
from backend.main import sanitize_sentry_event
from backend.config import Settings
from scripts.backup import create_backup, restore_backup


class RuntimeFlagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp.name) / "medha.sqlite3")
        self.database.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def test_flags_use_environment_defaults_and_persist_only_valid_values(self):
        self.assertFalse(self.database.get_runtime_flag("maintenance", False))
        self.assertEqual(self.database.get_runtime_flag("max_users", 25), 25)
        self.database.set_runtime_flag("maintenance", True)
        self.database.set_runtime_flag("max_users", 12)
        self.assertTrue(self.database.get_runtime_flag("maintenance", False))
        self.assertEqual(self.database.get_runtime_flag("max_users", 25), 12)
        for name, value in (("unknown", True), ("read_only", 1), ("max_users", 0), ("max_users", True)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.database.set_runtime_flag(name, value)

    def test_counters_return_only_aggregate_integer_counts(self):
        counts = self.database.counters()
        self.assertEqual(counts["users"], 0)
        self.assertEqual(counts["messages"], 0)
        self.assertNotIn("email", counts)
        self.assertTrue(all(type(value) is int for value in counts.values()))


class BackupTests(unittest.TestCase):
    def test_snapshot_restores_data_and_never_overwrites_a_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "source.sqlite3")
            database.initialize()
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO events(id, event_type, created_at) VALUES ('restore-check', 'test', 1)",
                )
            backup = create_backup(Path(database.path), root / "backups", keep=2)
            restored = root / "restored.sqlite3"
            restore_backup(backup, restored)
            connection = sqlite3.connect(restored)
            try:
                value = connection.execute("SELECT event_type FROM events WHERE id='restore-check'").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(value, "test")
            with self.assertRaises(FileExistsError):
                restore_backup(backup, restored)

    def test_backup_rotation_rejects_zero_retention(self):
        with self.assertRaises(ValueError):
            create_backup(Path("missing.sqlite3"), Path("unused"), keep=0)


class ErrorEventPrivacyTests(unittest.TestCase):
    def test_sentry_event_scrubber_removes_request_content_and_credentials(self):
        event = {
            "user": {"email": "private@example.edu"},
            "request": {
                "url": "https://medha.example/api/chat?token=secret",
                "query_string": "token=secret",
                "data": "private message",
                "cookies": {"medha_session": "secret"},
                "headers": {
                    "Authorization": "Bearer secret",
                    "X-CSRF-Token": "secret",
                    "Content-Type": "application/json",
                },
            },
        }
        sanitized = sanitize_sentry_event(event, {})
        serialized = repr(sanitized)
        for secret in ("private@example.edu", "private message", "Bearer secret", "medha_session", "token=secret"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(sanitized["request"]["headers"], {"Content-Type": "application/json"})

    def test_sentry_dsn_must_be_https_when_enabled(self):
        Settings(sentry_dsn="https://public@example.invalid/1").validate()
        with self.assertRaisesRegex(ValueError, "https URL"):
            Settings(sentry_dsn="http://example.invalid/1").validate()


if __name__ == "__main__":
    unittest.main()
