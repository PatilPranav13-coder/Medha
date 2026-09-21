"""A minimal decoder-only Transformer trained entirely from scratch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embed: int = 192
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1


class CharacterTokenizer:
    """Deliberately simple tokenizer whose vocabulary comes only from training text."""
    def __init__(self, characters: Iterable[str]):
        self.chars = sorted(set(characters))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        fallback = self.stoi.get(" ", 0)
        return [self.stoi.get(char, fallback) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(token), "") for token in token_ids)

    def state(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state(cls, state: dict) -> "CharacterTokenizer":
        return cls(state["chars"])


class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.scale = head_size ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        _, length, _ = x.shape
        scores = (self.query(x) @ self.key(x).transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        head_size = config.n_embed // config.n_heads
        self.heads = nn.ModuleList([AttentionHead(config, head_size) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.projection(torch.cat([head(x) for head in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed), nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed), nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm_1 = nn.LayerNorm(config.n_embed)
        self.norm_2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_1(x))
        return x + self.feed_forward(self.norm_2(x))


class MedhaTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError(f"Context is {length}; maximum is {self.config.block_size}.")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.lm_head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def save_checkpoint(path: Path, model: MedhaTransformer, tokenizer: CharacterTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "tokenizer": tokenizer.state(), "model": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[MedhaTransformer, CharacterTokenizer]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**state["config"])
    model = MedhaTransformer(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, CharacterTokenizer.from_state(state["tokenizer"])
