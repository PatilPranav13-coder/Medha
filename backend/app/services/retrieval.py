from __future__ import annotations

import re
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Document, DocumentChunk, Memory
from app.services.ai import embed_texts


def split_into_chunks(text: str, size: int = 1200, overlap: int = 160) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _terms(value: str) -> Counter[str]:
    return Counter(re.findall(r"[a-zA-Z0-9_+#.-]{2,}", value.casefold()))


def _lexical_score(query: Counter[str], content: str) -> float:
    words = _terms(content)
    if not query or not words:
        return 0.0
    common = set(query) & set(words)
    return sum(min(query[word], words[word]) for word in common) / max(1, sum(query.values()))


async def index_document(db: Session, document: Document) -> int:
    pieces = split_into_chunks(document.extracted_text)
    # Bound embedding work for very large code dumps; lexical retrieval still covers every stored chunk.
    vectors = await embed_texts(pieces[:200]) if pieces else []
    chunks = []
    for index, piece in enumerate(pieces):
        vector = vectors[index] if index < len(vectors) else None
        chunks.append(DocumentChunk(
            document_id=document.id,
            user_id=document.user_id,
            project_id=document.project_id,
            chunk_index=index,
            content=piece,
            embedding=vector,
        ))
    db.add_all(chunks)
    db.commit()
    return len(chunks)


async def retrieve_context(db: Session, user_id: str, project_id: str | None, query: str, limit: int = 5) -> list[dict[str, str]]:
    if not project_id:
        return []
    statement = select(DocumentChunk).where(DocumentChunk.user_id == user_id)
    statement = statement.where(DocumentChunk.project_id == project_id)
    chunks = list(db.scalars(statement.options(selectinload(DocumentChunk.document))).all())
    if not chunks:
        return []

    query_vectors = await embed_texts([query]) if query else []
    query_vector = query_vectors[0] if query_vectors else None
    if query_vector and db.get_bind().dialect.name == "postgresql":
        vector_query = (
            select(DocumentChunk)
            .where(DocumentChunk.user_id == user_id, DocumentChunk.embedding.is_not(None))
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(limit)
            .options(selectinload(DocumentChunk.document))
        )
        vector_query = vector_query.where(DocumentChunk.project_id == project_id)
        ranked = list(db.scalars(vector_query).all())
        if ranked:
            chunks = ranked
        else:
            query_terms = _terms(query)
            chunks.sort(key=lambda item: _lexical_score(query_terms, item.content), reverse=True)
    else:
        query_terms = _terms(query)
        chunks.sort(key=lambda item: _lexical_score(query_terms, item.content), reverse=True)

    return [
        {"title": item.document.filename, "url": f"document://{item.document_id}#chunk-{item.chunk_index + 1}", "snippet": item.content}
        for item in chunks[:limit]
    ]


def retrieve_memories(db: Session, user_id: str, project_id: str | None, query: str, limit: int = 4) -> list[Memory]:
    statement = select(Memory).where(Memory.user_id == user_id)
    if project_id:
        statement = statement.where((Memory.project_id == project_id) | (Memory.project_id.is_(None)))
    memories = list(db.scalars(statement).all())
    query_terms = _terms(query)
    memories.sort(key=lambda item: _lexical_score(query_terms, f"{item.label} {item.content}"), reverse=True)
    return memories[:limit]
