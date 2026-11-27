"""Small decoder-only Transformer used for Medha's from-scratch research track.

The model stays deliberately separate from the launch brain. Its architecture is
useful for learning and experiments, while the production adapter can point at a
locally served open-weights instruct model.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # Support both package use and the documented local unittest discovery command.
    from .tokenizer import ByteBPETokenizer
except ImportError:  # pragma: no cover - exercised only when run from this folder.
    from tokenizer import ByteBPETokenizer

CHECKPOINT_VERSION = 2
KVCache: TypeAlias = tuple[tuple[Tensor, Tensor], ...]


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embed: int = 192
    n_heads: int = 6
    n_kv_heads: int = 2
    n_layers: int = 6
    dropout: float = 0.1
    rotary_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.block_size <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, block_size, and n_layers must be positive.")
        if self.n_embed <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("Embedding and attention dimensions must be positive.")
        if self.n_embed % self.n_heads:
            raise ValueError("n_embed must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if self.rotary_base <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rotary_base and rms_norm_eps must be positive.")

    @property
    def head_dim(self) -> int:
        return self.n_embed // self.n_heads


class RMSNorm(nn.Module):
    """Normalize by root mean square while keeping a learned scale only."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, positions: Tensor, base: float) -> tuple[Tensor, Tensor]:
    """Apply rotary positions to query and key tensors, including cached steps."""
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    angles = torch.outer(positions.float(), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos = angles.cos().to(dtype=q.dtype)
    sin = angles.sin().to(dtype=q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    """Use fewer key/value heads than query heads to reduce attention state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_base = config.rotary_base
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embed, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embed, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embed, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)

    def forward(
        self,
        x: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, query_len, _ = x.shape
        q = self.q_proj(x).view(batch, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        positions = torch.arange(past_len, past_len + query_len, device=x.device)
        q, k = _apply_rope(q, k, positions, self.rotary_base)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None

        group_size = self.n_heads // self.n_kv_heads
        if group_size != 1:
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)

        key_positions = torch.arange(k.shape[-2], device=x.device)
        query_positions = past_len + torch.arange(query_len, device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.out_proj(attended), present


class SwiGLU(nn.Module):
    """Position-wise gated feed-forward block used by current decoder models."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int((8 * config.n_embed / 3 + 63) // 64 * 64)
        self.gate_proj = nn.Linear(config.n_embed, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embed, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embed, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.feed_forward_norm = RMSNorm(config.n_embed, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attended, present = self.attention(self.attention_norm(x), past_key_value, use_cache)
        x = x + attended
        return x + self.feed_forward(self.feed_forward_norm(x)), present


class MedhaTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.n_embed, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, KVCache]:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, non-empty sequence].")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one key/value pair per layer.")
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_len + tokens.shape[1] > self.config.block_size:
            raise ValueError(f"Context exceeds the configured {self.config.block_size}-token window.")

        hidden = self.token_embedding(tokens)
        presents: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = block(hidden, past, use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @staticmethod
    def _filter_logits(logits: Tensor, top_k: int, top_p: float) -> Tensor:
        if top_k > 0 and top_k < logits.shape[-1]:
            cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            mask = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
            logits = logits.masked_fill(mask, float("-inf"))
        return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, non-empty sequence].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1].")
        if max_new_tokens == 0:
            return tokens

        original_mode = self.training
        self.eval()
        output = tokens
        try:
            context = tokens[:, -self.config.block_size:]
            # The KV cache cannot grow past the training context window.
            max_new_tokens = min(max_new_tokens, self.config.block_size - context.shape[1])
            if max_new_tokens == 0:
                return tokens
            result = self(context, use_cache=use_cache)
            logits, _, cache = result if use_cache else (*result, None)
            finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

            for step in range(max_new_tokens):
                next_logits = logits[:, -1, :] / temperature
                next_logits = self._filter_logits(next_logits, top_k, top_p)
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
                if eos_token_id is not None:
                    next_token = torch.where(finished[:, None], eos_token_id, next_token)
                    finished |= next_token[:, 0].eq(eos_token_id)
                output = torch.cat((output, next_token), dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
                if step + 1 < max_new_tokens:
                    if use_cache:
                        logits, _, cache = self(next_token, past_key_values=cache, use_cache=True)
                    else:
                        context = output[:, -self.config.block_size:]
                        logits, _ = self(context)
        finally:
            self.train(original_mode)
        return output


def save_checkpoint(path: Path, model: MedhaTransformer, tokenizer: ByteBPETokenizer) -> None:
    """Write a versioned, tensor-and-primitive-only checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "tokenizer": tokenizer.state(),
        "model": model.state_dict(),
    }
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, device: torch.device | str) -> tuple[MedhaTransformer, ByteBPETokenizer]:
    """Load only safe checkpoint primitives and tensors; reject unknown versions."""
    state = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(state, dict) or state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Expected checkpoint version {CHECKPOINT_VERSION}.")
    if not isinstance(state.get("config"), dict) or not isinstance(state.get("model"), dict):
        raise ValueError("Checkpoint is missing model configuration or weights.")
    config = ModelConfig(**state["config"])
    model = MedhaTransformer(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    tokenizer = ByteBPETokenizer.from_state(state.get("tokenizer", {}))
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the model configuration.")
    return model, tokenizer
