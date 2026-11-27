from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.auth import AuthError, AuthRepository, hash_password, verify_password
from backend.database import Database, token_digest


class AccountRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "medha.sqlite3"
        self.database = Database(self.path)
        self.repository = AuthRepository(self.database, max_users=2, session_ttl_seconds=3600)
        self.repository.initialize("invite-one")

    def tearDown(self):
        self.temp.cleanup()

    def test_migrations_are_numbered_idempotent_and_enable_wal(self):
        self.repository.initialize("invite-one")
        with self.database.connection() as connection:
            versions = [row[0] for row in connection.execute("SELECT version FROM schema_migrations")]
            journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(versions, [1, 2, 3, 4])
        self.assertEqual(journal.lower(), "wal")
        self.assertTrue({"users", "sessions", "conversations", "messages", "memories", "uploads", "feedback", "consents", "invite_codes", "events"} <= tables)

    def test_password_hash_uses_random_salt_and_verifies(self):
        salt_a, digest_a = hash_password("correct horse battery")
        salt_b, digest_b = hash_password("correct horse battery")
        self.assertNotEqual(salt_a, salt_b)
        self.assertNotEqual(digest_a, digest_b)
        self.assertTrue(verify_password("correct horse battery", salt_a, digest_a))
        self.assertFalse(verify_password("wrong password", salt_a, digest_a))

    def test_signup_requires_a_valid_single_use_invite_and_hashes_session(self):
        user, session_token, csrf_token = self.repository.signup(
            "Student@Example.edu", "this is a long password", "invite-one", "2026-11-27-v1", now=1000
        )
        self.assertEqual(user["email"], "student@example.edu")
        self.assertEqual(len(session_token.encode("ascii")), 43)
        self.assertNotEqual(session_token, token_digest(session_token))
        session = self.repository.find_session(session_token, now=1001)
        self.assertIsNotNone(session)
        self.assertEqual(session["csrf_hash"], token_digest(csrf_token))
        with self.database.connection() as connection:
            stored = connection.execute("SELECT session_hash FROM sessions").fetchone()[0]
            invite = connection.execute("SELECT used_count FROM invite_codes").fetchone()[0]
        self.assertEqual(stored, token_digest(session_token))
        self.assertEqual(invite, 1)
        with self.database.connection() as connection:
            consent = connection.execute(
                "SELECT notice_version FROM consents WHERE user_id=?", (user["id"],)
            ).fetchone()
        self.assertEqual(consent["notice_version"], "2026-11-27-v1")
        with self.assertRaises(AuthError):
            self.repository.signup("second@example.edu", "this is a long password", "invite-one", "2026-11-27-v1")

    def test_signup_enforces_password_minimum_and_total_user_cap(self):
        with self.assertRaisesRegex(ValueError, "10 characters"):
            self.repository.signup("a@example.edu", "short", "invite-one", "2026-11-27-v1")
        user, _, _ = self.repository.signup("a@example.edu", "this is a long password", "invite-one", "2026-11-27-v1")
        self.database.create_invite_code("invite-two")
        self.repository.signup("b@example.edu", "another long password", "invite-two", "2026-11-27-v1")
        self.database.create_invite_code("invite-three")
        with self.assertRaisesRegex(AuthError, "unavailable"):
            self.repository.signup("c@example.edu", "third long password", "invite-three", "2026-11-27-v1")
        with self.database.connection() as connection:
            used = connection.execute(
                "SELECT used_count FROM invite_codes WHERE code_hash=?", (token_digest("invite-three"),)
            ).fetchone()[0]
        self.assertEqual(used, 0)
        self.assertTrue(user["id"])

    def test_repeated_failures_lock_account_and_login_errors_are_generic(self):
        self.repository.signup("locked@example.edu", "long enough password", "invite-one", "2026-11-27-v1", now=1000)
        messages = []
        for index in range(5):
            with self.assertRaises(AuthError) as raised:
                self.repository.login("locked@example.edu", "incorrect password", now=2000 + index)
            messages.append(str(raised.exception))
        with self.assertRaises(AuthError) as raised:
            self.repository.login("missing@example.edu", "incorrect password", now=2005)
        messages.append(str(raised.exception))
        self.assertEqual(set(messages), {"Email or password is incorrect."})
        with self.database.connection() as connection:
            locked_until = connection.execute(
                "SELECT locked_until FROM users WHERE email='locked@example.edu'"
            ).fetchone()[0]
        self.assertEqual(locked_until, 2064)
        with self.assertRaises(AuthError):
            self.repository.login("locked@example.edu", "long enough password", now=2030)

    def test_logout_revokes_session(self):
        _, token, _ = self.repository.signup("logout@example.edu", "long enough password", "invite-one", "2026-11-27-v1", now=1000)
        self.assertIsNotNone(self.repository.find_session(token, now=1001))
        self.repository.revoke_session(token, now=1002)
        self.assertIsNone(self.repository.find_session(token, now=1003))


if __name__ == "__main__":
    unittest.main()
