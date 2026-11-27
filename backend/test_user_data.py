from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.database import Database
from backend.user_data import MemoryRepository, UploadError, UploadStore, rank_memories


class UserDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database = Database(root / "medha.sqlite3")
        self.database.initialize()
        self.storage = root / "private-files"
        self.store = UploadStore(self.database, self.storage, max_file_bytes=1024, quota_bytes=1500)
        with self.database.transaction() as connection:
            for identifier, email in (("user-a", "a@example.edu"), ("user-b", "b@example.edu")):
                connection.execute(
                    "INSERT INTO users(id,email,password_salt,password_hash) VALUES (?,?,?,?)",
                    (identifier, email, b"salt", b"hash"),
                )
        self.memories = MemoryRepository(self.database)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_memories_are_bm25_retrieved_and_owner_scoped_for_crud(self) -> None:
        first = self.memories.add("user-a", "I prefer Python code examples")
        self.memories.add("user-a", "I am studying orbital mechanics")
        self.memories.add("user-b", "I prefer JavaScript examples")
        ranked = rank_memories("show Python examples", self.memories.list("user-a"))
        self.assertEqual(ranked[0]["id"], first["id"])
        self.assertFalse(self.memories.update("user-b", first["id"], "Changed by another user"))
        self.assertFalse(self.memories.delete("user-b", first["id"]))
        self.assertTrue(self.memories.update("user-a", first["id"], "Use short Python snippets"))
        self.assertTrue(self.memories.delete("user-a", first["id"]))

    def test_memory_length_is_bounded_and_duplicate_save_is_idempotent(self) -> None:
        first = self.memories.add("user-a", "  Remember   this  ")
        same = self.memories.add("user-a", "Remember this")
        self.assertEqual(first["id"], same["id"])
        with self.assertRaises(ValueError):
            self.memories.add("user-a", " " * 1001)
        with self.assertRaises(ValueError):
            self.memories.add("user-a", "x" * 1001)

    def test_uploads_use_random_private_names_and_retrieve_only_owned_chunks(self) -> None:
        content = b"The satellite uses a solar array to provide electrical power to its onboard systems."
        uploaded = self.store.add("user-a", "../satellite-notes.txt", content)
        path = next(self.storage.iterdir())
        self.assertRegex(path.name, r"^[0-9a-f]{32}\.txt$")
        self.assertNotEqual(path.name, "satellite-notes.txt")
        self.assertEqual(path.read_bytes(), content)
        results = self.store.retrieve("user-a", "solar array electrical power", [uploaded["id"]])
        self.assertTrue(results)
        self.assertIn("solar array", results[0]["content"])
        with self.assertRaises(KeyError):
            self.store.retrieve("user-b", "solar array", [uploaded["id"]])
        self.assertTrue(self.store.delete("user-a", uploaded["id"]))
        self.assertFalse(path.exists())

    def test_upload_type_signature_size_quota_and_storage_path_are_enforced(self) -> None:
        with self.assertRaises(UploadError):
            self.store.add("user-a", "program.exe", b"print('bad')")
        with self.assertRaises(UploadError):
            self.store.add("user-a", "notes.txt", b"\xff\xfe")
        with self.assertRaises(UploadError):
            self.store.add("user-a", "notes.pdf", b"not a pdf")
        with self.assertRaises(UploadError):
            self.store.add("user-a", "large.txt", b"x" * 1025)
        self.store.add("user-a", "one.txt", b"The launch vehicle uses a new orbital rocket stage. " * 12)
        with self.assertRaises(UploadError) as quota:
            self.store.add("user-a", "two.txt", b"z" * 1000)
        self.assertEqual(quota.exception.status_code, 413)
        with self.assertRaises(ValueError):
            self.store._path("..\\outside.txt")

    def test_delete_all_user_files_removes_disk_data_and_index(self) -> None:
        self.store.add("user-a", "keep-private.md", b"Important private engineering notes.")
        self.store.add("user-b", "other.md", b"A different account has a separate note.")
        self.assertEqual(self.store.delete_user_files("user-a"), 1)
        self.assertEqual(self.store.list_for_user("user-a"), [])
        self.assertEqual(len(list(self.storage.iterdir())), 1)
        with self.database.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM upload_chunks").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
