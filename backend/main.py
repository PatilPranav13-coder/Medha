"""Production FastAPI application: same-origin Medha OS, accounts, and streamed chat."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.auth import AuthRepository
from backend.auth_routes import CSRF_COOKIE, SESSION_COOKIE, _check_csrf, create_auth_router
from backend.chat_repository import ChatRepository
from backend.config import Settings
from backend.database import Database
from backend.medha_brain import (
    Brain,
    GenerationCapacityError,
    GenerationGate,
    GenerationParams,
    build_brain,
    prepare_messages,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "index.html"


def make_project_plan(brief: str) -> dict:
    """Keep the existing deterministic project-plan helper for its current callers."""
    summary = " ".join((brief or "").split())
    if not summary:
        raise ValueError("A project brief is required.")
    words = re.findall(r"[a-zA-Z0-9]+", summary.lower())
    slug = "-".join(words[:5]) or "new-project"
    mobile = any(word in summary.lower() for word in ("ios", "android", "mobile", "swift", "kotlin", "flutter"))
    first_step = (
        "Define the mobile target, supported OS versions, navigation, device capabilities and acceptance criteria."
        if mobile else "Define the product scope, target users, supported platforms and acceptance criteria."
    )
    return {
        "slug": slug,
        "summary": summary,
        "steps": [
            first_step,
            "Map the frontend, backend, API contracts, database model and external integrations.",
            "Build a thin end-to-end product slice with clear module boundaries and documented assumptions.",
            "Add unit, integration and UI tests; review accessibility, security, performance and failure states.",
            "Package the release, document setup and operations, and validate it against the acceptance criteria.",
        ],
        "status": "draft_plan",
    }


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=80)


class ConversationPatch(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ChatBody(BaseModel):
    message: str | None = Field(default=None, max_length=10000)
    conversation_id: str | None = Field(default=None, max_length=64)
    action: Literal["send", "regenerate", "edit"] = "send"
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class FeedbackBody(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


def _current_session(request: Request, auth: AuthRepository) -> tuple[str, dict]:
    raw_token = request.cookies.get(SESSION_COOKIE)
    session = auth.find_session(raw_token) if raw_token else None
    if session is None:
        raise HTTPException(status_code=401, detail="Sign-in required.")
    return raw_token or "", session


def _require_csrf(request: Request, session: dict | None) -> None:
    _check_csrf(request, session)


def _require_writable(settings: Settings) -> None:
    if settings.maintenance:
        raise HTTPException(status_code=503, detail="Medha is temporarily in maintenance mode.")
    if settings.read_only:
        raise HTTPException(status_code=503, detail="Medha is temporarily read-only.")


def _title_from_message(message: str) -> str:
    return " ".join(message.split())[:60] or "New chat"


def _sse(event: str, payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


_END = object()


def _next_chunk(iterator: Iterator[str]) -> str | object:
    try:
        return next(iterator)
    except StopIteration:
        return _END


def create_app(
    settings: Settings | None = None,
    brain: Brain | None = None,
    database: Database | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    selected_brain = brain or build_brain(configured)
    db = database or Database(configured.database_path)
    auth = AuthRepository(db, configured.max_users, configured.session_ttl_seconds)
    chats = ChatRepository(db)
    generation_gate = GenerationGate(
        max_active=configured.max_concurrent_generations,
        max_queued=configured.max_queued_generations,
        queue_timeout=configured.generation_queue_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Initialize the DB and warm the operator-selected brain before accepting traffic.
        configured.validate()
        auth.initialize(configured.bootstrap_invite_code)
        selected_brain.warmup()
        yield

    app = FastAPI(title="Medha API", lifespan=lifespan)
    app.state.settings = configured
    app.state.brain = selected_brain
    app.state.database = db
    app.state.auth_repository = auth
    app.state.chat_repository = chats
    app.state.generation_gate = generation_gate
    app.include_router(create_auth_router(configured, auth))

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(FRONTEND)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        ready = selected_brain.health()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "brain": configured.brain_provider},
            status_code=200 if ready else 503,
        )

    @app.get("/api/me")
    def me(request: Request) -> dict[str, object]:
        _, session = _current_session(request, auth)
        return {"user": session["user"], "csrf_token": request.cookies.get(CSRF_COOKIE, "")}

    @app.get("/api/conversations")
    def list_conversations(request: Request) -> list[dict]:
        _, session = _current_session(request, auth)
        return chats.list_conversations(session["user"]["id"])

    @app.post("/api/conversations")
    def create_conversation(body: ConversationCreate, request: Request) -> dict[str, object]:
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        identifier = chats.create_conversation(session["user"]["id"], body.title)
        return {"id": identifier, "title": body.title}

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: Request) -> dict[str, object]:
        _, session = _current_session(request, auth)
        conversation = chats.get_conversation(session["user"]["id"], conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conversation

    @app.patch("/api/conversations/{conversation_id}")
    def patch_conversation(conversation_id: str, body: ConversationPatch, request: Request) -> dict[str, bool]:
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        if not chats.rename_conversation(session["user"]["id"], conversation_id, body.title):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"ok": True}

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, request: Request) -> Response:
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        if not chats.delete_conversation(session["user"]["id"], conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return Response(status_code=204)

    @app.delete("/api/conversations")
    def delete_all_conversations(request: Request) -> dict[str, bool]:
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        chats.delete_all_conversations(session["user"]["id"])
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(body: ChatBody, request: Request):
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        user_id = session["user"]["id"]
        if body.action in {"send", "edit"} and not (body.message or "").strip():
            raise HTTPException(status_code=422, detail="A message is required.")

        conversation_id = body.conversation_id
        if body.action == "send":
            message = (body.message or "").strip()
            if conversation_id is None:
                conversation_id = chats.create_conversation(user_id, _title_from_message(message))
            else:
                conversation = chats.get_conversation(user_id, conversation_id)
                if conversation is None:
                    raise HTTPException(status_code=404, detail="Conversation not found.")
                if not conversation["messages"] and conversation["title"] == "New chat":
                    chats.rename_conversation(user_id, conversation_id, _title_from_message(message))
            try:
                chats.append_message(user_id, conversation_id, "user", message)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc
        elif conversation_id is None:
            raise HTTPException(status_code=422, detail="This action needs a conversation.")
        elif body.action == "edit":
            if not chats.edit_last_user_message(user_id, conversation_id, (body.message or "").strip()):
                raise HTTPException(status_code=404, detail="A previous user message was not found.")
        else:
            if chats.prepare_regeneration(user_id, conversation_id) is None:
                raise HTTPException(status_code=404, detail="A previous user message was not found.")

        conversation = chats.get_conversation(user_id, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        model_messages = prepare_messages(conversation["messages"], configured.history_token_budget)
        params = GenerationParams(
            max_tokens=min(body.max_tokens or configured.max_output_tokens, configured.max_output_tokens),
            temperature=configured.temperature,
            top_p=configured.top_p,
        )
        try:
            await asyncio.to_thread(generation_gate.acquire)
        except GenerationCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail="Medha is busy. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from exc

        request_id = str(uuid.uuid4())

        async def stream():
            generated: list[str] = []
            iterator = selected_brain.generate_stream(model_messages, params)
            persisted_id: str | None = None
            disconnected = False
            try:
                while True:
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    chunk = await asyncio.to_thread(_next_chunk, iterator)
                    if chunk is _END:
                        break
                    if not isinstance(chunk, str) or not chunk:
                        continue
                    generated.append(chunk)
                    yield _sse("token", {"text": chunk, "conversation_id": conversation_id})
                if generated:
                    persisted_id = chats.append_message(user_id, conversation_id or "", "assistant", "".join(generated))
                if not disconnected:
                    # WP5 supplies search citations; an empty list here is explicitly honest.
                    yield _sse("sources", [])
                    yield _sse("done", {"conversation_id": conversation_id, "message_id": persisted_id})
            except Exception:
                if not disconnected:
                    yield _sse("error", {"message": "Medha could not complete this response. Please try again."})
            finally:
                # Keep a partial answer in history when the client stops generation.
                if generated and persisted_id is None:
                    try:
                        chats.append_message(user_id, conversation_id or "", "assistant", "".join(generated))
                    except Exception:
                        pass
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                generation_gate.release()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    @app.post("/api/feedback")
    def feedback(body: FeedbackBody, request: Request) -> dict[str, bool]:
        _, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        try:
            chats.add_feedback(session["user"]["id"], body.message_id, body.rating, body.comment)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Message not found.") from exc
        return {"ok": True}

    @app.get("/api/export")
    def export_data(request: Request) -> JSONResponse:
        _, session = _current_session(request, auth)
        export = chats.export_user(session["user"]["id"])
        if export is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        return JSONResponse(export, headers={"Content-Disposition": "attachment; filename=medha-export.json"})

    @app.delete("/api/account")
    def delete_account(request: Request, response: Response) -> Response:
        raw_token, session = _current_session(request, auth)
        _require_csrf(request, session)
        _require_writable(configured)
        auth.revoke_session(raw_token)
        chats.delete_account(session["user"]["id"])
        response.delete_cookie(SESSION_COOKIE, path="/", secure=configured.cookie_secure, httponly=True, samesite="lax")
        response.delete_cookie(CSRF_COOKIE, path="/", secure=configured.cookie_secure, httponly=False, samesite="lax")
        response.status_code = 204
        return response

    return app


app = create_app()
