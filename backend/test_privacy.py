from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from backend.config import Settings
from backend.database import Database
from backend.main import create_app
from backend.medha_brain import StubBrain


class EventRetentionTests(unittest.TestCase):
    def test_pruning_removes_only_events_beyond_the_configured_age(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            database.initialize()
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO events(id, event_type, created_at) VALUES ('old', 'old_event', ?)",
                    (1.0,),
                )
                connection.execute(
                    "INSERT INTO events(id, event_type, created_at) VALUES ('new', 'new_event', ?)",
                    (10 * 24 * 60 * 60,),
                )
            deleted = database.prune_events(retention_days=30, now=40 * 24 * 60 * 60)
            self.assertEqual(deleted, 1)
            with database.connection() as connection:
                remaining = [row[0] for row in connection.execute("SELECT id FROM events")]
            self.assertEqual(remaining, ["new"])

    def test_pruning_rejects_zero_or_negative_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            with self.assertRaises(ValueError):
                database.prune_events(0)

    def test_application_startup_applies_configured_event_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            database.initialize()
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO events(id, event_type, created_at) VALUES ('stale', 'old_event', ?)",
                    (time.time() - 31 * 24 * 60 * 60,),
                )
            settings = Settings(database_path=database.path, log_retention_days=30)
            app = create_app(settings, StubBrain("ready"), database)

            async def start_once() -> None:
                async with app.router.lifespan_context(app):
                    pass

            asyncio.run(start_once())
            with database.connection() as connection:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM events WHERE id='stale'"
                ).fetchone()[0]
            self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
