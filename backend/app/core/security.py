from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

password_hash = PasswordHash.recommended()
TOKEN_TTL = timedelta(hours=12)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    return password_hash.verify(password, encoded)


def create_session_token(user_id: str) -> str:
    if not settings.auth_secret:
        raise RuntimeError("Set MEDHA_AUTH_SECRET before starting Medha.")
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + TOKEN_TTL}, settings.auth_secret, algorithm="HS256")


def decode_session_token(token: str) -> str | None:
    if not settings.auth_secret:
        return None
    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None
