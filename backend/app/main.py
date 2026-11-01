from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api import auth, conversations, memories, projects
from app.api.dependencies import get_current_user
from app.core.config import settings
from app.db import get_db, init_db
from app.models import User
from app.services.ai import model_catalog


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if len(settings.auth_secret.encode("utf-8")) < 32 or settings.auth_secret.startswith("replace-with-"):
        raise RuntimeError("Set MEDHA_AUTH_SECRET to a unique secret of at least 32 bytes before starting Medha.")
    if settings.environment == "production" and not settings.cookie_secure:
        raise RuntimeError("Set MEDHA_COOKIE_SECURE=true when MEDHA_ENV=production.")
    init_db()
    yield


app = FastAPI(title="Medha Technical Workspace", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(projects.router)
app.include_router(memories.router)

_rate_windows: dict[str, deque[float]] = {}
_rate_lock = Lock()


@app.middleware("http")
async def request_guards(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in settings.cors_origins and request.method not in {"GET", "HEAD", "OPTIONS"}:
        return JSONResponse(status_code=403, content={"detail": "Request origin is not allowed."})

    path = request.url.path
    if request.method == "POST" and (path in {"/api/auth/login", "/api/auth/register"} or ("/messages" in path and path.startswith("/api/conversations/"))):
        client_ip = request.client.host if request.client else "unknown"
        bucket = f"{client_ip}:{'auth' if path.startswith('/api/auth/') else 'chat'}"
        now = time.monotonic()
        limit = 10 if bucket.endswith(":auth") else 30
        with _rate_lock:
            timestamps = _rate_windows.setdefault(bucket, deque())
            while timestamps and now - timestamps[0] >= 60:
                timestamps.popleft()
            if len(timestamps) >= limit:
                return JSONResponse(status_code=429, content={"detail": "Too many requests. Wait a minute and try again."})
            timestamps.append(now)
    return await call_next(request)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not ready.") from exc
    return {"status": "ready"}


@app.get("/api/models")
def models(_user: User = Depends(get_current_user)) -> dict:
    return model_catalog()
