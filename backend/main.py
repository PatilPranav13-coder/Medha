from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STUDIO_DIR = PROJECT_ROOT / "studio"

app = FastAPI(title="Medha API", version="0.2.0")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    response: str


class ProjectPlanRequest(BaseModel):
    brief: str = Field(min_length=8, max_length=4000)


def make_project_plan(brief: str) -> dict:
    """Create a safe, deterministic starter plan (not an LLM-generated plan)."""
    cleaned = " ".join(brief.split())
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    name = " ".join(words[:5]).title() or "Untitled Project"
    if len(name) > 54:
        name = name[:51].rstrip() + "..."
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "new-project"

    lower = cleaned.lower()
    if any(term in lower for term in ("ios", "android", "mobile", "iphone", "ipad")):
        platform_step = "Choose the mobile target and decide between native and cross-platform implementation."
    elif any(term in lower for term in ("api", "backend", "server")):
        platform_step = "Define API boundaries, request/response models and backend responsibilities."
    else:
        platform_step = "Confirm the target platform, core users and success criteria."

    steps = [
        platform_step,
        "Break the brief into screens, user flows and acceptance criteria.",
        "Create the project structure and implement the first usable interface.",
        "Connect application logic and data persistence where required.",
        "Run tests, review accessibility and document how to launch the project.",
    ]
    return {
        "name": name,
        "slug": slug,
        "summary": cleaned,
        "steps": steps,
        "status": "draft_plan",
        "note": "This is a deterministic starter plan. AI-generated planning and file creation are not connected yet.",
    }


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/studio", include_in_schema=False)
def serve_studio() -> FileResponse:
    return FileResponse(STUDIO_DIR / "index.html")


app.mount("/studio", StaticFiles(directory=STUDIO_DIR), name="studio")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "medha-api", "version": app.version}


@app.post("/api/projects/plan")
def create_project_plan(request: ProjectPlanRequest) -> dict:
    brief = request.brief.strip()
    if len(brief) < 8:
        raise HTTPException(status_code=422, detail="Please provide a little more detail.")
    return {"project": make_project_plan(brief)}


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
