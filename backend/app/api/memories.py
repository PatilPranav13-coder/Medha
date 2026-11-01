from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.api.projects import get_owned_project
from app.db import get_db
from app.models import Memory, User
from app.schemas import MemoryCreate, MemoryView

router = APIRouter(prefix="/api/memories", tags=["memory"])


@router.get("", response_model=list[MemoryView])
def list_memories(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Memory]:
    return list(db.scalars(select(Memory).where(Memory.user_id == user.id).order_by(Memory.created_at.desc())).all())


@router.post("", response_model=MemoryView, status_code=status.HTTP_201_CREATED)
def create_memory(body: MemoryCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Memory:
    if body.project_id:
        get_owned_project(db, body.project_id, user.id)
    memory = Memory(user_id=user.id, **body.model_dump())
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(memory_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    memory = db.scalar(select(Memory).where(Memory.id == memory_id, Memory.user_id == user.id))
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found.")
    db.delete(memory)
    db.commit()
