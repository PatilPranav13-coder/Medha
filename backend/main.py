import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Medha API")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    response: str


def make_project_plan(brief: str) -> dict:
    """Return a deterministic, lightweight implementation plan for a project brief."""
    summary = " ".join((brief or "").split())
    if not summary:
        raise ValueError("A project brief is required.")

    words = re.findall(r"[a-zA-Z0-9]+", summary.lower())
    slug = "-".join(words[:5]) or "new-project"
    mobile = any(
        word in summary.lower()
        for word in ("ios", "android", "mobile", "swift", "kotlin", "flutter")
    )
    first_step = (
        "Define the mobile target, supported OS versions, navigation, device capabilities and acceptance criteria."
        if mobile
        else "Define the product scope, target users, supported platforms and acceptance criteria."
    )
    steps = [
        first_step,
        "Map the frontend, backend, API contracts, database model and external integrations.",
        "Build a thin end-to-end product slice with clear module boundaries and documented assumptions.",
        "Add unit, integration and UI tests; review accessibility, security, performance and failure states.",
        "Package the release, document setup and operations, and validate it against the acceptance criteria.",
    ]
    return {"slug": slug, "summary": summary, "steps": steps, "status": "draft_plan"}


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    # Placeholder until Medha's local LLM inference is connected here.
    return ChatResponse(
        response=(
            "Medha received your message: “"
            + message
            + "”. The frontend and FastAPI backend are connected."
        )
    )
