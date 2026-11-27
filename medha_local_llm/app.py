"""Optional development server for Medha's experimental from-scratch model."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from medha_local_llm.model import MedhaTransformer, load_checkpoint
from medha_local_llm.tokenizer import ByteBPETokenizer

ROOT = Path(__file__).parent
CHECKPOINT = Path(os.getenv("MEDHA_EXPERIMENTAL_CHECKPOINT", ROOT / "checkpoints" / "medha.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Medha Experimental Model")
model: MedhaTransformer | None = None
tokenizer: ByteBPETokenizer | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


def get_model() -> tuple[MedhaTransformer, ByteBPETokenizer]:
    global model, tokenizer
    if model is None or tokenizer is None:
        if not CHECKPOINT.exists():
            raise HTTPException(503, "No trained experimental checkpoint is installed.")
        try:
            model, tokenizer = load_checkpoint(CHECKPOINT, DEVICE)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(503, "The experimental checkpoint could not be loaded.") from exc
    return model, tokenizer


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    net, vocab = get_model()
    reserve = min(32, max(1, net.config.block_size // 4))
    content_limit = max(0, net.config.block_size - reserve - 3)
    # Keep the chat markers and reserve context for a reply, even for long inputs.
    prompt = [vocab.bos_token_id, vocab.special_token_ids["<user>"]]
    prompt.extend(vocab.encode(request.message)[-content_limit:] if content_limit else [])
    prompt.append(vocab.special_token_ids["<assistant>"])
    input_ids = torch.tensor([prompt], dtype=torch.long, device=DEVICE)
    generated = net.generate(
        input_ids,
        max_new_tokens=min(128, net.config.block_size - input_ids.shape[1]),
        temperature=request.temperature,
        top_p=request.top_p,
        eos_token_id=vocab.eos_token_id,
    )
    answer_tokens = generated[0, input_ids.shape[1]:].tolist()
    answer = vocab.decode(answer_tokens, skip_special_tokens=True).strip()
    return {"reply": answer or "I do not know enough to answer that reliably yet."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
