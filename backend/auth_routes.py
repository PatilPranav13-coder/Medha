"""Invite-only signup and cookie-session API routes."""
from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.auth import AuthError, AuthRepository
from backend.config import Settings
from backend.database import compare_digest, token_digest
from backend.privacy import CONSENT_NOTICE

SESSION_COOKIE = "medha_session"
CSRF_COOKIE = "medha_csrf"
EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"


class SignupBody(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=10, max_length=128)
    invite_code: str = Field(min_length=1, max_length=256)
    consent: bool = False
    consent_version: str = Field(min_length=1, max_length=64)


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=1, max_length=128)


def _session_from_request(request: Request, repository: AuthRepository):
    token = request.cookies.get(SESSION_COOKIE)
    return (token, repository.find_session(token)) if token else (None, None)


def _check_csrf(request: Request, session: dict | None) -> None:
    supplied = request.headers.get("X-CSRF-Token", "")
    if session is not None:
        valid = bool(supplied) and compare_digest(token_digest(supplied), session["csrf_hash"])
    else:
        cookie = request.cookies.get(CSRF_COOKIE, "")
        valid = bool(supplied and cookie) and compare_digest(supplied, cookie)
    if not valid:
        raise HTTPException(status_code=403, detail="CSRF validation failed.")


def _set_session_cookies(response: Response, token: str, csrf: str, settings: Settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf,
        max_age=settings.session_ttl_seconds,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def create_auth_router(settings: Settings, repository: AuthRepository) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.get("/consent-notice")
    def consent_notice() -> dict[str, object]:
        contact = settings.grievance_email or "the beta operator"
        notice = [*CONSENT_NOTICE, f"For privacy questions, contact {contact}."]
        return {"version": settings.consent_notice_version, "notice": notice}

    @router.get("/csrf")
    def get_csrf(request: Request, response: Response) -> dict[str, str]:
        _, session = _session_from_request(request, repository)
        existing = request.cookies.get(CSRF_COOKIE, "")
        if session is not None:
            # A safe GET may return the existing token, but must not rotate server state.
            if not existing or not compare_digest(token_digest(existing), session["csrf_hash"]):
                raise HTTPException(status_code=403, detail="Refresh sign-in to restore CSRF protection.")
            return {"csrf_token": existing}
        csrf = secrets.token_urlsafe(32)
        response.set_cookie(
            CSRF_COOKIE,
            csrf,
            max_age=settings.session_ttl_seconds,
            httponly=False,
            secure=settings.cookie_secure,
            samesite="lax",
            path="/",
        )
        return {"csrf_token": csrf}

    @router.post("/signup")
    def signup(body: SignupBody, request: Request, response: Response) -> dict[str, object]:
        _, session = _session_from_request(request, repository)
        _check_csrf(request, session)
        runtime_signups = repository.database.get_runtime_flag("signups_enabled", True)
        runtime_maintenance = repository.database.get_runtime_flag("maintenance", False)
        runtime_read_only = repository.database.get_runtime_flag("read_only", False)
        if (not settings.signups_enabled or not runtime_signups
                or settings.maintenance or runtime_maintenance
                or settings.read_only or runtime_read_only):
            raise HTTPException(status_code=503, detail="Signup is currently unavailable.")
        if not body.consent or body.consent_version != settings.consent_notice_version:
            raise HTTPException(status_code=400, detail="Review and accept the current consent notice to create an account.")
        try:
            user, token, csrf = repository.signup(
                body.email, body.password, body.invite_code, body.consent_version,
            )
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _set_session_cookies(response, token, csrf, settings)
        return {"user": user, "csrf_token": csrf}

    @router.post("/login")
    def login(body: LoginBody, request: Request, response: Response) -> dict[str, object]:
        _, session = _session_from_request(request, repository)
        _check_csrf(request, session)
        if settings.maintenance or settings.read_only:
            raise HTTPException(status_code=503, detail="Sign-in is temporarily unavailable.")
        try:
            user, token, csrf = repository.login(body.email, body.password)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _set_session_cookies(response, token, csrf, settings)
        return {"user": user, "csrf_token": csrf}

    @router.post("/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        raw_session, session = _session_from_request(request, repository)
        _check_csrf(request, session)
        if raw_session and session:
            repository.revoke_session(raw_session)
        response.delete_cookie(SESSION_COOKIE, path="/", secure=settings.cookie_secure, httponly=True, samesite="lax")
        response.delete_cookie(CSRF_COOKIE, path="/", secure=settings.cookie_secure, httponly=False, samesite="lax")
        return {"ok": True}

    @router.get("/me")
    def me(request: Request) -> dict[str, object]:
        _, session = _session_from_request(request, repository)
        if session is None:
            raise HTTPException(status_code=401, detail="Sign-in required.")
        return {"user": session["user"], "csrf_token": request.cookies.get(CSRF_COOKIE, "")}

    return router
