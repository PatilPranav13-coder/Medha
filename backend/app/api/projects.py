from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.dependencies import get_current_user
from app.db import get_db
from app.models import Document, Project, User
from app.schemas import DocumentView, ProjectCreate, ProjectUpdate, ProjectView
from app.services.retrieval import index_document

router = APIRouter(prefix="/api/projects", tags=["projects"])
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".csv"}


def get_owned_project(db: Session, project_id: str, user_id: str) -> Project:
    project = db.scalar(select(Project).where(Project.id == project_id, Project.user_id == user_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@router.get("", response_model=list[ProjectView])
def list_projects(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Project]:
    return list(db.scalars(select(Project).where(Project.user_id == user.id).order_by(Project.updated_at.desc())).all())


@router.post("", response_model=ProjectView, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Project:
    project = Project(user_id=user.id, **body.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.patch("/{project_id}", response_model=ProjectView)
def update_project(project_id: str, body: ProjectUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Project:
    project = get_owned_project(db, project_id, user.id)
    for key, value in body.model_dump(exclude_unset=True).items():
        if value is None:
            raise HTTPException(status_code=422, detail=f"Project field '{key}' cannot be null.")
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    project = get_owned_project(db, project_id, user.id)
    db.delete(project)
    db.commit()


@router.get("/{project_id}/documents", response_model=list[DocumentView])
def list_documents(project_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[DocumentView]:
    get_owned_project(db, project_id, user.id)
    documents = db.scalars(
        select(Document).where(Document.project_id == project_id, Document.user_id == user.id).options(selectinload(Document.chunks)).order_by(Document.created_at.desc())
    ).all()
    return [DocumentView(id=item.id, filename=item.filename, extension=item.extension, size_bytes=item.size_bytes, created_at=item.created_at, chunk_count=len(item.chunks)) for item in documents]


@router.post("/{project_id}/documents", response_model=DocumentView, status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentView:
    project = get_owned_project(db, project_id, user.id)
    filename = PurePosixPath((file.filename or "upload").replace("\\", "/")).name
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported files: PDF, TXT, MD, Python, JavaScript, TypeScript, HTML, CSS, JSON, and CSV.")
    from app.core.config import settings

    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        limit_mb = max(1, settings.max_upload_bytes // (1024 * 1024))
        raise HTTPException(status_code=413, detail=f"Files are limited to {limit_mb} MB.")
    try:
        if extension == ".pdf":
            from io import BytesIO
            from pypdf import PdfReader

            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        else:
            text = content.decode("utf-8-sig")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Medha could not read this file as UTF-8 text or PDF.") from exc
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No readable text was found in this file.")
    document = Document(user_id=user.id, project_id=project.id, filename=filename[:255], extension=extension, size_bytes=len(content), extracted_text=text)
    db.add(document)
    db.commit()
    db.refresh(document)
    chunk_count = await index_document(db, document)
    return DocumentView(id=document.id, filename=document.filename, extension=document.extension, size_bytes=document.size_bytes, created_at=document.created_at, chunk_count=chunk_count)


@router.delete("/{project_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(project_id: str, document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    get_owned_project(db, project_id, user.id)
    document = db.scalar(select(Document).where(Document.id == document_id, Document.project_id == project_id, Document.user_id == user.id))
    if not document:
        raise HTTPException(status_code=404, detail="File not found.")
    db.delete(document)
    db.commit()
