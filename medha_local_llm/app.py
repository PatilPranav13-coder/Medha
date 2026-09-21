"""Local-only chat server for trained Medha weights."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import load_checkpoint

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "medha.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Medha Local LLM")
model = tokenizer = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)


def get_model():
    global model, tokenizer
    if model is None:
        if not CHECKPOINT.exists():
            raise HTTPException(503, "No trained model yet. Run: python train.py --steps 2500")
        model, tokenizer = load_checkpoint(CHECKPOINT, DEVICE)
    return model, tokenizer


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    net, vocab = get_model()
    # The model learns this prompt format when it is present in your corpus.
    prompt = f"User: {request.message}\nMedha:"
    input_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        generated = net.generate(input_ids, max_new_tokens=300, temperature=request.temperature)
    text = vocab.decode(generated[0].tolist())
    answer = text[len(prompt):].split("\nUser:", 1)[0].strip()
    return {"reply": answer or "I need more training data to answer that well."}


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 lets a container platform route traffic to the local app.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
