"""Train Medha from random weights on your local training corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import CharacterTokenizer, MedhaTransformer, ModelConfig, save_checkpoint

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "medha.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = (ROOT / "data" / "training.txt").read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    config = ModelConfig(vocab_size=len(tokenizer.chars))
    if len(data) <= config.block_size:
        raise ValueError("Training data must be longer than the model context window.")
    model = MedhaTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {device}.")

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(data) - config.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in starts]).to(device)
        return x, y

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f}")
    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"Saved Medha's learned weights to {args.checkpoint}")


if __name__ == "__main__":
    main()
