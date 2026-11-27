"""Train Medha's experimental model from a local text corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medha_local_llm.model import MedhaTransformer, ModelConfig, save_checkpoint
from medha_local_llm.tokenizer import ByteBPETokenizer

ROOT = Path(__file__).parent


def parse_conversations(text: str) -> list[list[dict[str, str]]]:
    """Read the simple User:/Medha: starter format without assuming one pair per file."""
    conversations: list[list[dict[str, str]]] = []
    pending_user: str | None = None
    for line in text.splitlines():
        if line.startswith("User: "):
            pending_user = line[6:].strip()
        elif line.startswith("Medha: ") and pending_user is not None:
            conversations.append([
                {"role": "user", "content": pending_user},
                {"role": "assistant", "content": line[7:].strip()},
            ])
            pending_user = None
    if pending_user is not None:
        conversations.append([{"role": "user", "content": pending_user}])
    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the experimental Medha research model.")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embed", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "medha.pt")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "training.txt")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conversations = parse_conversations(args.corpus.read_text(encoding="utf-8"))
    if not conversations:
        raise ValueError("The corpus has no User:/Medha: training pairs.")
    tokenizer = ByteBPETokenizer.train(
        (message["content"] for conversation in conversations for message in conversation),
        vocab_size=args.vocab_size,
    )
    token_stream = [
        token
        for conversation in conversations
        for token in tokenizer.encode_chat(conversation)
    ]
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
    )
    if len(token_stream) <= config.block_size:
        raise ValueError("Training data must exceed the model context window.")
    data = torch.tensor(token_stream, dtype=torch.long)
    model = MedhaTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Training {sum(parameter.numel() for parameter in model.parameters()):,} parameters on {device}.")

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(data) - config.block_size, (args.batch_size,))
        x = torch.stack([data[index:index + config.block_size] for index in starts]).to(device)
        y = torch.stack([data[index + 1:index + config.block_size + 1] for index in starts]).to(device)
        return x, y

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch()
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f}")
    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"Saved experimental Medha weights to {args.checkpoint}")


if __name__ == "__main__":
    main()
