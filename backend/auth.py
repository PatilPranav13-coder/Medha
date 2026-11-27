"""Password, invite, and opaque-session operations for the accounts repository."""
from __future__ import annotations

import base64
import hashlib
import secrets
import sqlite3
import time
import uuid
from typing import Any

from backend.database import Database, token_digest

SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
MIN_PASSWORD_LENGTH = 10
_DUMMY_SALT = b"medha-login-salt"
_DUMMY_HASH = hashlib.scrypt(
    b"not-a-real-password", salt=_DUMMY_SALT, n=SCRYPT_N, r=SCRYPT_R,
    p=SCRYPT_P, dklen=SCRYPT_DKLEN, maxmem=64 * 1024 * 1024,
)


class AuthError(RuntimeError):
    """A generic authentication failure safe to return to callers."""


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash a password using a random per-user salt and a deliberately costly KDF."""
    chosen_salt = salt or secrets.token_bytes(16)
    result = hashlib.scrypt(
        password.encode("utf-8"), salt=chosen_salt, n=SCRYPT_N, r=SCRYPT_R,
        p=SCRYPT_P, dklen=SCRYPT_DKLEN, maxmem=64 * 1024 * 1024,
    )
    return chosen_salt, result


def verify_password(password: str, salt: bytes, expected: bytes) -> bool:
    _, actual = hash_password(password, salt)
    return secrets.compare_digest(actual, expected)


def _new_token() -> str:
    # URL-safe encoding preserves exactly 32 random bytes without putting them in SQLite.
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _public_user(row: sqlite3.Row) -> dict[str, str]:
    return {"id": row["id"], "email": row["email"], "role": row["role"]}


class AuthRepository:
    def __init__(self, database: Database, max_users: int, session_ttl_seconds: int) -> None:
        self.database = database
        self.max_users = max_users
        self.session_ttl_seconds = session_ttl_seconds

    def initialize(self, bootstrap_invite_code: str | None = None) -> None:
        self.database.initialize()
        if bootstrap_invite_code:
            digest = token_digest(bootstrap_invite_code.strip())
            with self.database.transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM invite_codes WHERE code_hash=?", (digest,)
                ).fetchone()
                if not exists:
                    connection.execute(
                        "INSERT INTO invite_codes(code_hash, max_uses) VALUES (?, 1)", (digest,)
                    )

    def signup(self, email: str, password: str, invite_code: str, now: float | None = None) -> tuple[dict[str, str], str, str]:
        normalized_email = email.strip().lower()
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if not normalized_email or not invite_code.strip():
            raise AuthError("Signup details are invalid.")
        salt, password_hash = hash_password(password)
        current = time.time() if now is None else now
        user_id = str(uuid.uuid4())
        session_token, csrf_token = _new_token(), _new_token()
        with self.database.transaction() as connection:
            invite_hash = token_digest(invite_code.strip())
            invite = connection.execute(
                "SELECT max_uses, used_count, expires_at FROM invite_codes WHERE code_hash=?",
                (invite_hash,),
            ).fetchone()
            if (
                invite is None
                or invite["used_count"] >= invite["max_uses"]
                or (invite["expires_at"] is not None and invite["expires_at"] <= current)
            ):
                raise AuthError("Signup details are invalid.")
            user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if user_count >= self.max_users:
                raise AuthError("Signup is currently unavailable.")
            try:
                connection.execute(
                    "INSERT INTO users(id, email, password_salt, password_hash) VALUES (?, ?, ?, ?)",
                    (user_id, normalized_email, salt, password_hash),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError("Signup details are invalid.") from exc
            connection.execute(
                "UPDATE invite_codes SET used_count=used_count+1 WHERE code_hash=?", (invite_hash,)
            )
            connection.execute(
                "INSERT INTO sessions(session_hash, user_id, csrf_hash, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token_digest(session_token), user_id, token_digest(csrf_token), current,
                 current + self.session_ttl_seconds),
            )
            connection.execute(
                "INSERT INTO events(id, user_id, event_type, metadata_json, created_at) VALUES (?, ?, ?, '{}', ?)",
                (str(uuid.uuid4()), user_id, "signup", current),
            )
            user = connection.execute("SELECT id, email, role FROM users WHERE id=?", (user_id,)).fetchone()
        assert user is not None
        return _public_user(user), session_token, csrf_token

    def login(self, email: str, password: str, now: float | None = None) -> tuple[dict[str, str], str, str]:
        normalized_email = email.strip().lower()
        current = time.time() if now is None else now
        session_token, csrf_token = _new_token(), _new_token()
        failure = False
        user_public: dict[str, str] | None = None
        with self.database.transaction() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE email=?", (normalized_email,)
            ).fetchone()
            if user is None:
                # Spend comparable KDF work for unknown accounts to reduce email enumeration.
                verify_password(password, _DUMMY_SALT, _DUMMY_HASH)
                failure = True
            else:
                password_ok = verify_password(password, user["password_salt"], user["password_hash"])
                is_locked = user["locked_until"] is not None and user["locked_until"] > current
                if not password_ok or is_locked or user["disabled"]:
                    failed_count = user["failed_login_count"] + (0 if is_locked else 1)
                    locked_until = user["locked_until"]
                    if failed_count >= 5 and not is_locked:
                        backoff = min(3600, 60 * (2 ** min(failed_count - 5, 6)))
                        locked_until = current + backoff
                    connection.execute(
                        "UPDATE users SET failed_login_count=?, locked_until=? WHERE id=?",
                        (failed_count, locked_until, user["id"]),
                    )
                    failure = True
                else:
                    connection.execute(
                        "UPDATE users SET failed_login_count=0, locked_until=NULL WHERE id=?",
                        (user["id"],),
                    )
                    connection.execute(
                        "INSERT INTO sessions(session_hash, user_id, csrf_hash, created_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (token_digest(session_token), user["id"], token_digest(csrf_token), current,
                         current + self.session_ttl_seconds),
                    )
                    user_public = _public_user(user)
        if failure or user_public is None:
            raise AuthError("Email or password is incorrect.")
        return user_public, session_token, csrf_token

    def find_session(self, token: str, now: float | None = None) -> dict[str, Any] | None:
        current = time.time() if now is None else now
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT s.session_hash, s.csrf_hash, s.expires_at, u.id, u.email, u.role "
                "FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.session_hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND u.disabled=0",
                (token_digest(token), current),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_hash": row["session_hash"],
            "csrf_hash": row["csrf_hash"],
            "expires_at": row["expires_at"],
            "user": {"id": row["id"], "email": row["email"], "role": row["role"]},
        }

    def revoke_session(self, token: str, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET revoked_at=? WHERE session_hash=? AND revoked_at IS NULL",
                (current, token_digest(token)),
            )

    def set_csrf(self, token: str, csrf: str, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self.database.transaction() as connection:
            result = connection.execute(
                "UPDATE sessions SET csrf_hash=? WHERE session_hash=? AND revoked_at IS NULL AND expires_at>?",
                (token_digest(csrf), token_digest(token), current),
            )
            if result.rowcount != 1:
                raise AuthError("Session is no longer active.")

