"""Per-user conversation, message, and feedback storage."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from backend.database import Database


class ChatRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_conversation(self, user_id: str, title: str = "New chat") -> str:
        identifier, now = str(uuid.uuid4()), time.time()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO conversations(id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (identifier, user_id, title[:80], now, now),
            )
        return identifier

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count "
                "FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
                "WHERE c.user_id=? GROUP BY c.id ORDER BY c.updated_at DESC", (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        message = dict(row)
        try:
            sources = json.loads(message.pop("sources_json", "[]"))
        except (TypeError, json.JSONDecodeError):
            sources = []
        message["sources"] = sources if isinstance(sources, list) else []
        return message

    def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                "SELECT id, role, content, created_at, sources_json FROM messages "
                "WHERE conversation_id=? ORDER BY created_at, rowid", (conversation_id,),
            ).fetchall()
        result = dict(row)
        result["messages"] = [self._message(message) for message in messages]
        return result

    def count_messages(self, user_id: str, conversation_id: str) -> int:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE user_id=? AND conversation_id=?",
                (user_id, conversation_id),
            ).fetchone()
        return int(row["total"])

    def append_message(
        self, user_id: str, conversation_id: str, role: str, content: str,
        sources: list[dict[str, object]] | None = None,
    ) -> str:
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages can be stored.")
        identifier, now = str(uuid.uuid4()), time.time()
        with self.database.transaction() as connection:
            owner = connection.execute(
                "SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
            ).fetchone()
            if owner is None:
                raise KeyError("Conversation does not exist.")
            connection.execute(
                "INSERT INTO messages(id, conversation_id, user_id, role, content, created_at, sources_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (identifier, conversation_id, user_id, role, content, now,
                 json.dumps(sources or [], ensure_ascii=False, separators=(",", ":"))),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?",
                (now, conversation_id, user_id),
            )
        return identifier

    def prepare_regeneration(self, user_id: str, conversation_id: str) -> str | None:
        """Keep the last user request and remove only later generated turns."""
        with self.database.transaction() as connection:
            owner = connection.execute(
                "SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
            ).fetchone()
            if owner is None:
                return None
            last_user = connection.execute(
                "SELECT rowid, content FROM messages WHERE conversation_id=? AND user_id=? AND role='user' "
                "ORDER BY rowid DESC LIMIT 1", (conversation_id, user_id),
            ).fetchone()
            if last_user is None:
                return None
            connection.execute(
                "DELETE FROM messages WHERE conversation_id=? AND rowid>?",
                (conversation_id, last_user["rowid"]),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), conversation_id)
            )
            return last_user["content"]

    def edit_last_user_message(self, user_id: str, conversation_id: str, content: str) -> bool:
        now = time.time()
        with self.database.transaction() as connection:
            last_user = connection.execute(
                "SELECT m.rowid, m.id FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                "WHERE c.id=? AND c.user_id=? AND m.role='user' ORDER BY m.rowid DESC LIMIT 1",
                (conversation_id, user_id),
            ).fetchone()
            if last_user is None:
                return False
            connection.execute(
                "UPDATE messages SET content=?, created_at=? WHERE id=?",
                (content, now, last_user["id"]),
            )
            connection.execute(
                "DELETE FROM messages WHERE conversation_id=? AND rowid>?",
                (conversation_id, last_user["rowid"]),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?",
                (now, conversation_id, user_id),
            )
        return True

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=? AND user_id=?",
                (title.strip()[:80] or "New chat", time.time(), conversation_id, user_id),
            )
        return result.rowcount == 1

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute(
                "DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
            )
        return result.rowcount == 1

    def delete_all_conversations(self, user_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))

    def add_feedback(self, user_id: str, message_id: str, rating: int, comment: str | None) -> str:
        identifier = str(uuid.uuid4())
        with self.database.transaction() as connection:
            owned = connection.execute(
                "SELECT 1 FROM messages WHERE id=? AND user_id=?", (message_id, user_id)
            ).fetchone()
            if owned is None:
                raise KeyError("Message does not exist.")
            connection.execute(
                "INSERT INTO feedback(id, user_id, message_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, user_id, message_id, rating, comment, time.time()),
            )
        return identifier

    def export_user(self, user_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            user = connection.execute(
                "SELECT id, email, role, created_at FROM users WHERE id=?", (user_id,)
            ).fetchone()
            if user is None:
                return None
            conversations = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id=? ORDER BY created_at",
                (user_id,),
            ).fetchall()
            result = dict(user)
            result["conversations"] = []
            for conversation in conversations:
                row = dict(conversation)
                row["messages"] = [self._message(message) for message in connection.execute(
                    "SELECT id, role, content, created_at, sources_json FROM messages WHERE conversation_id=? ORDER BY created_at, rowid",
                    (row["id"],),
                ).fetchall()]
                result["conversations"].append(row)
            result["memories"] = [dict(row) for row in connection.execute(
                "SELECT id, content, created_at, updated_at FROM memories WHERE user_id=?", (user_id,)
            ).fetchall()]
            result["uploads"] = [dict(row) for row in connection.execute(
                "SELECT id, original_name, media_type, byte_size, created_at FROM uploads WHERE user_id=?",
                (user_id,),
            ).fetchall()]
            result["consents"] = [dict(row) for row in connection.execute(
                "SELECT notice_version, granted_at FROM consents WHERE user_id=?", (user_id,)
            ).fetchall()]
            result["feedback"] = [dict(row) for row in connection.execute(
                "SELECT id, message_id, rating, comment, created_at FROM feedback WHERE user_id=?",
                (user_id,),
            ).fetchall()]
        return result

    def delete_account(self, user_id: str) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute("DELETE FROM users WHERE id=?", (user_id,))
        return result.rowcount == 1

    def record_event(self, user_id: str, event_type: str, metadata: dict[str, object] | None = None) -> None:
        # Keep event payloads small and metadata-only; callers must not pass message contents.
        encoded = json.dumps(metadata or {}, separators=(",", ":"))
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO events(id, user_id, event_type, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), user_id, event_type, encoded, time.time()),
            )
