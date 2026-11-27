"""User-controlled memories and bounded file storage/retrieval."""
from __future__ import annotations

import io
import re
import time
import uuid
from pathlib import Path
from typing import Any

from backend.database import Database
from websearch.ranking import Passage, rank_passages, split_passages


class UploadError(ValueError):
    """A safe upload validation failure with an HTTP status for the API layer."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _clean_memory(content: str) -> str:
    clean = " ".join(content.split())
    if not clean or len(clean) > 1000:
        raise ValueError("Memory must contain between 1 and 1000 characters.")
    return clean


class MemoryRepository:
    """Store only facts users deliberately ask Medha to remember."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, user_id: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT id, content, created_at, updated_at FROM memories WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add(self, user_id: str, content: str) -> dict[str, Any]:
        clean = _clean_memory(content)
        now = time.time()
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM memories WHERE user_id=? AND content=? COLLATE NOCASE", (user_id, clean)
            ).fetchone()
            if existing:
                connection.execute(
                    "UPDATE memories SET updated_at=? WHERE id=? AND user_id=?",
                    (now, existing["id"], user_id),
                )
                identifier = existing["id"]
            else:
                identifier = str(uuid.uuid4())
                connection.execute(
                    "INSERT INTO memories(id, user_id, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (identifier, user_id, clean, now, now),
                )
        return {"id": identifier, "content": clean, "created_at": now, "updated_at": now}

    def update(self, user_id: str, memory_id: str, content: str) -> bool:
        clean = _clean_memory(content)
        with self.database.transaction() as connection:
            result = connection.execute(
                "UPDATE memories SET content=?, updated_at=? WHERE id=? AND user_id=?",
                (clean, time.time(), memory_id, user_id),
            )
        return result.rowcount == 1

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute("DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, user_id))
        return result.rowcount == 1


def rank_memories(query: str, memories: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Use the same dependency-free BM25 implementation as web search."""
    passages = [
        Passage(index, "User memory", str(memory["id"]), str(memory["content"]))
        for index, memory in enumerate(memories)
    ]
    ranked = rank_passages(query, passages, limit=limit)
    by_id = {str(item["id"]): item for item in memories}
    return [by_id[passage.url] for score, passage in ranked if score > 0 and passage.url in by_id]


class UploadStore:
    """Keep original files outside the static web root and index bounded text chunks."""

    ALLOWED_TYPES = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
    }
    STORAGE_NAME = re.compile(r"^[0-9a-f]{32}\.(?:txt|md|csv|pdf)$")
    MAX_PDF_PAGES = 200
    MAX_EXTRACTED_CHARS = 2_000_000

    def __init__(self, database: Database, storage_dir: str | Path, max_file_bytes: int, quota_bytes: int) -> None:
        self.database = database
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.max_file_bytes = max_file_bytes
        self.quota_bytes = quota_bytes

    def _path(self, storage_name: str) -> Path:
        # Database values are checked as random basenames before filesystem access.
        if not self.STORAGE_NAME.fullmatch(storage_name):
            raise ValueError("Stored upload name is invalid.")
        candidate = (self.storage_dir / storage_name).resolve()
        if candidate.parent != self.storage_dir:
            raise ValueError("Stored upload path escaped its storage directory.")
        return candidate

    @classmethod
    def _extract_text(cls, suffix: str, content: bytes) -> str:
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise UploadError("The PDF file signature is invalid.")
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(content), strict=True)
                if reader.is_encrypted:
                    raise UploadError("Encrypted PDFs are not supported.")
                if len(reader.pages) > cls.MAX_PDF_PAGES:
                    raise UploadError("PDFs may contain at most 200 pages.", 413)
                parts: list[str] = []
                total = 0
                for page in reader.pages:
                    text = page.extract_text() or ""
                    total += len(text)
                    if total > cls.MAX_EXTRACTED_CHARS:
                        raise UploadError("The extracted PDF text is too large.", 413)
                    parts.append(text)
                extracted = "\n".join(parts)
            except UploadError:
                raise
            except Exception as exc:
                raise UploadError("The PDF could not be safely read.") from exc
        else:
            try:
                extracted = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise UploadError("Text files must use UTF-8 encoding.") from exc
            if "\x00" in extracted:
                raise UploadError("The selected file is not valid text.")
        if len(extracted) > cls.MAX_EXTRACTED_CHARS:
            raise UploadError("The extracted file text is too large.", 413)
        if not extracted.strip():
            raise UploadError("No readable text was found in this file.")
        return extracted

    def add(self, user_id: str, original_name: str, content: bytes) -> dict[str, Any]:
        if len(content) > self.max_file_bytes:
            raise UploadError("This file exceeds the upload size limit.", 413)
        if not content:
            raise UploadError("The selected file is empty.")
        safe_name = Path(original_name.replace("\\", "/")).name
        safe_name = "".join(char for char in safe_name if char.isprintable())[:160]
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in self.ALLOWED_TYPES:
            raise UploadError("Upload a .txt, .md, .csv, or .pdf file.")
        extracted = self._extract_text(suffix, content)
        chunks = split_passages(extracted, max_chars=1200, overlap_chars=120)
        identifier = str(uuid.uuid4())
        storage_name = uuid.uuid4().hex + suffix
        target = self._path(storage_name)
        created_at = time.time()
        wrote_file = False
        try:
            with self.database.transaction() as connection:
                total = connection.execute(
                    "SELECT COALESCE(SUM(byte_size),0) FROM uploads WHERE user_id=?", (user_id,)
                ).fetchone()[0]
                if int(total) + len(content) > self.quota_bytes:
                    raise UploadError("Your uploaded files reached the account storage quota.", 413)
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stored:
                    wrote_file = True
                    stored.write(content)
                connection.execute(
                    "INSERT INTO uploads(id, user_id, storage_name, original_name, media_type, byte_size, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (identifier, user_id, storage_name, safe_name, self.ALLOWED_TYPES[suffix], len(content), created_at),
                )
                connection.executemany(
                    "INSERT INTO upload_chunks(upload_id, chunk_number, content) VALUES (?, ?, ?)",
                    [(identifier, number, text) for number, text in enumerate(chunks)],
                )
        except Exception:
            if wrote_file:
                target.unlink(missing_ok=True)
            raise
        return {
            "id": identifier,
            "original_name": safe_name,
            "media_type": self.ALLOWED_TYPES[suffix],
            "byte_size": len(content),
            "created_at": created_at,
            "chunk_count": len(chunks),
        }

    def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT id, original_name, media_type, byte_size, created_at FROM uploads "
                "WHERE user_id=? ORDER BY created_at DESC", (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_for_export(self, user_id: str, upload_id: str) -> bytes:
        """Read an owner's original file for export without exposing its filesystem path."""
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT storage_name FROM uploads WHERE id=? AND user_id=?", (upload_id, user_id)
            ).fetchone()
        if row is None:
            raise KeyError("Upload does not belong to this user.")
        return self._path(row["storage_name"]).read_bytes()

    def validate_selection(self, user_id: str, upload_ids: list[str]) -> list[str]:
        """Reject cross-account or excessive file selections before saving a chat turn."""
        unique_ids = list(dict.fromkeys(upload_ids))
        if len(unique_ids) > 5:
            raise KeyError("At most five files may be selected.")
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        with self.database.connection() as connection:
            found = connection.execute(
                f"SELECT id FROM uploads WHERE user_id=? AND id IN ({placeholders})",
                (user_id, *unique_ids),
            ).fetchall()
        if len(found) != len(unique_ids):
            raise KeyError("A selected file does not belong to this user.")
        return unique_ids

    def retrieve(self, user_id: str, query: str, upload_ids: list[str], limit: int = 5) -> list[dict[str, Any]]:
        unique_ids = self.validate_selection(user_id, upload_ids)
        if not unique_ids:
            raise KeyError("Selected file list is invalid.")
        placeholders = ",".join("?" for _ in unique_ids)
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT u.id AS upload_id, u.original_name, c.chunk_number, c.content "
                "FROM upload_chunks c JOIN uploads u ON u.id=c.upload_id "
                f"WHERE u.user_id=? AND u.id IN ({placeholders}) ORDER BY u.id, c.chunk_number",
                (user_id, *unique_ids),
            ).fetchall()
        records = [dict(row) for row in rows]
        passages = [
            Passage(index, record["original_name"], record["upload_id"], record["content"])
            for index, record in enumerate(records)
        ]
        ranked = rank_passages(query, passages, limit=limit)
        lookup = {(str(item["upload_id"]), item["chunk_number"]): item for item in records}
        results = []
        for score, passage in ranked:
            item = lookup[(passage.url, records[passage.source_number]["chunk_number"])]
            if score > 0:
                results.append(item)
        return results

    def delete(self, user_id: str, upload_id: str) -> bool:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT storage_name FROM uploads WHERE id=? AND user_id=?", (upload_id, user_id)
            ).fetchone()
            if row is None:
                return False
            self._path(row["storage_name"]).unlink(missing_ok=True)
            connection.execute("DELETE FROM uploads WHERE id=? AND user_id=?", (upload_id, user_id))
        return True

    def delete_user_files(self, user_id: str) -> int:
        with self.database.transaction() as connection:
            rows = connection.execute("SELECT storage_name FROM uploads WHERE user_id=?", (user_id,)).fetchall()
            for row in rows:
                self._path(row["storage_name"]).unlink(missing_ok=True)
            connection.execute("DELETE FROM uploads WHERE user_id=?", (user_id,))
        return len(rows)

