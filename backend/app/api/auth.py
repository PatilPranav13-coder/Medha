from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.security import TOKEN_TTL, create_session_token, hash_password, verify_password
from app.db import get_db
from app.models import User
from app.schemas import UserCreate, UserLogin, UserView

router = APIRouter(prefix="/api/auth", tags=["authentication"])


def _set_session(response: Response, user: User) -> None:
    response.set_cookie(
        "medha_session",
        create_session_token(user.id),
        max_age=int(TOKEN_TTL.total_seconds()),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=UserView, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate, response: Response, db: Session = Depends(get_db)) -> User:
    normalized_email = str(body.email).lower()
    if db.scalar(select(User.id).where(User.email == normalized_email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = User(email=normalized_email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    db.refresh(user)
    _set_session(response, user)
    return user


@router.post("/login", response_model=UserView)
def login(body: UserLogin, response: Response, db: Session = Depends(get_db)) -> User:
    email = str(body.email).lower()
    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _set_session(response, user)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> Response:
    response.delete_cookie("medha_session", path="/", httponly=True, secure=settings.cookie_secure, samesite="lax")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserView)
def me(user: User = Depends(get_current_user)) -> User:
    return user
