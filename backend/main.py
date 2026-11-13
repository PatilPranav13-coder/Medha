"""Self-hosted Medha chat API backed by Medha's own scratch-trained model."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_DIR = PROJECT_ROOT / "medha_local_llm"
sys.path.insert(0, str(MODEL_DIR))
from model import load_checkpoint  # noqa: E402

CHECKPOINT = Path(os.getenv("MEDHA_CHECKPOINT", MODEL_DIR / "checkpoints" / "medha.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Medha — local AI")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
model = tokenizer = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    response: str


def get_model():
    global model, tokenizer
    if model is None:
        if not CHECKPOINT.exists():
            raise HTTPException(
                status_code=503,
                detail="Medha has no trained checkpoint yet. Follow medha_local_llm/README.md to train it locally.",
            )
        model, tokenizer = load_checkpoint(CHECKPOINT, DEVICE)
    return model, tokenizer


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready" if CHECKPOINT.exists() else "needs_training", "inference": "local"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    net, vocab = get_model()
    # Use the same role markers as the from-scratch training corpus.
    turns = request.history[-6:] + [Message(role="user", content=request.message.strip())]
    prompt = "".join(
        ("User: " if turn.role == "user" else "Medha: ") + turn.content.strip() + "\n"
        for turn in turns
    ) + "Medha:"
    ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        generated = net.generate(ids, max_new_tokens=240, temperature=0.8)
    decoded = vocab.decode(generated[0].tolist())
    answer = decoded[len(prompt):].split("\nUser:", 1)[0].split("\nMedha:", 1)[0].strip()
    return ChatResponse(response=answer or "I need more training examples to answer that well.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
