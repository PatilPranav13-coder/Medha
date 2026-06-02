"""
Transformer Neural Network — "Attention Is All You Need" (Vaswani et al., 2017)
================================================================================
A complete, from-scratch implementation of the original Transformer architecture.

Components implemented:
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Position-wise Feed-Forward Network
  - Positional Encoding
  - Encoder Layer & Encoder Stack
  - Decoder Layer & Decoder Stack
  - Full Transformer (Encoder-Decoder)
  - Label-smoothed Cross-Entropy Loss
  - Noam learning-rate scheduler (as in the paper)
  - Greedy & beam-search decoding helpers

Dependencies: Python 3.8+, PyTorch ≥ 2.0

Quick-start (language-model style demo at the bottom):
    python transformer.py
"""

import math
import copy
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# 1. SCALED DOT-PRODUCT ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    query: Tensor,          # (B, heads, T_q, d_k)
    key:   Tensor,          # (B, heads, T_k, d_k)
    value: Tensor,          # (B, heads, T_k, d_v)
    mask:  Optional[Tensor] = None,   # broadcastable boolean mask (True = ignore)
    dropout: Optional[nn.Dropout] = None,
) -> tuple[Tensor, Tensor]:
    """
    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Returns:
        output      — (B, heads, T_q, d_v)
        attn_weights — (B, heads, T_q, T_k)  useful for visualisation
    """
    d_k = query.size(-1)
    # Raw scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# 2. MULTI-HEAD ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1, …, head_h) · W_O
    where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    All h heads are computed in parallel via a single batched matrix multiply.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads   # dimension per head

        # Projection matrices (no bias in the original paper, but common to add)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[Tensor] = None   # saved for inspection

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) → (B, heads, T, d_k)"""
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, heads, T, d_k) → (B, T, d_model)"""
        B, _, T, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: Tensor,                  # (B, T_q, d_model)
        key:   Tensor,                  # (B, T_k, d_model)
        value: Tensor,                  # (B, T_k, d_model)
        mask:  Optional[Tensor] = None, # (B, 1, T_q, T_k) or similar
    ) -> Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        merged = self._merge_heads(attn_out)
        return self.W_o(merged)


# ──────────────────────────────────────────────────────────────────────────────
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────────────────────

class PositionWiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2

    Uses ReLU as in the paper (d_ff = 4 × d_model by default).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. POSITIONAL ENCODING
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (fixed, not learned):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding table once
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]       # type: ignore[index]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# 5. LAYER NORMALISATION + RESIDUAL WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class SublayerConnection(nn.Module):
    """
    Applies LayerNorm → sublayer → dropout, then adds the residual.
    The paper uses Post-LN; many modern implementations use Pre-LN (done here
    as Pre-LN is more stable to train without a warm-up scheduler).
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ──────────────────────────────────────────────────────────────────────────────
# 6. ENCODER LAYER
# ──────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    One Transformer encoder block:
        x → Self-Attention → Add&Norm → FFN → Add&Norm
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer  = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.sublayer[0](x, lambda z: self.self_attn(z, z, z, src_mask))
        x = self.sublayer[1](x, self.ffn)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 7. ENCODER STACK
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Stack of N identical EncoderLayers."""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# 8. DECODER LAYER
# ──────────────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    One Transformer decoder block:
        x → Masked Self-Attention → Add&Norm
          → Cross-Attention (over encoder output) → Add&Norm
          → FFN → Add&Norm
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer   = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(
        self,
        x:         Tensor,
        memory:    Tensor,
        src_mask:  Optional[Tensor] = None,
        tgt_mask:  Optional[Tensor] = None,
    ) -> Tensor:
        # 1. Masked self-attention (decoder sees only past tokens)
        x = self.sublayer[0](x, lambda z: self.self_attn(z, z, z, tgt_mask))
        # 2. Cross-attention over encoder output
        x = self.sublayer[1](x, lambda z: self.cross_attn(z, memory, memory, src_mask))
        # 3. Feed-forward
        x = self.sublayer[2](x, self.ffn)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 9. DECODER STACK
# ──────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Stack of N identical DecoderLayers."""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        Tensor,
        memory:   Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# 10. EMBEDDING + OUTPUT PROJECTION
# ──────────────────────────────────────────────────────────────────────────────

class Embeddings(nn.Module):
    """Standard token embedding, scaled by √d_model as in the paper."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.scale  = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embed(x) * self.scale


class Generator(nn.Module):
    """Linear projection + log-softmax to produce token-level log-probs."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 11. FULL TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    The complete Transformer model (encoder-decoder) from
    "Attention Is All You Need" (Vaswani et al., 2017).

    Architecture:
        src_tokens → src_embed + pos_enc → Encoder ──┐
                                                      ↓ cross-attn
        tgt_tokens → tgt_embed + pos_enc → Decoder → Generator → log-probs
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        num_heads: int   = 8,
        num_layers: int  = 6,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        max_len:   int   = 5000,
    ):
        super().__init__()

        # Encoder side
        enc_layer  = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, num_layers)
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )

        # Decoder side
        dec_layer  = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layer, num_layers)
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )

        # Output head
        self.generator = Generator(d_model, tgt_vocab_size)

        # Weight tying (share src/tgt embeddings + generator when vocabs match)
        if src_vocab_size == tgt_vocab_size:
            self.tgt_embed[0].embed.weight = self.src_embed[0].embed.weight
            self.generator.proj.weight     = self.src_embed[0].embed.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all linear & embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── mask helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def make_pad_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
        """
        Padding mask: True where the token is <pad>.
        Shape: (B, 1, 1, T)  — broadcasts over heads and query positions.
        """
        return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size: int, device: torch.device) -> Tensor:
        """
        Causal (look-ahead) mask: upper triangle is True (blocked).
        Shape: (1, 1, size, size)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    # ── forward ───────────────────────────────────────────────────────────────

    def encode(
        self,
        src:     Tensor,             # (B, S)
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory:   Tensor,            # (B, S, d_model) — encoder output
        tgt:      Tensor,            # (B, T)
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(
        self,
        src:      Tensor,            # (B, S)
        tgt:      Tensor,            # (B, T)
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns log-probabilities: (B, T, tgt_vocab_size)."""
        memory  = self.encode(src, src_mask)
        decoded = self.decode(memory, tgt, src_mask, tgt_mask)
        return self.generator(decoded)


# ──────────────────────────────────────────────────────────────────────────────
# 12. LABEL-SMOOTHED LOSS
# ──────────────────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing (ε = smoothing).
    Replaces the one-hot target distribution with:
        y_smooth = (1 - ε) * y_one_hot + ε / (V - 1)
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        log_probs : (N, V) — already log-softmaxed
        target    : (N,)   — ground-truth token ids
        """
        V = log_probs.size(-1)
        # Build smooth distribution
        smooth_dist = torch.full_like(log_probs, self.smoothing / (V - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0.0

        # Zero out loss for padding positions
        mask = (target == self.pad_idx)
        smooth_dist[mask] = 0.0

        return self.criterion(log_probs, smooth_dist)


# ──────────────────────────────────────────────────────────────────────────────
# 13. NOAM LEARNING-RATE SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

class NoamScheduler:
    """
    lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})

    Increases linearly for warmup_steps, then decays ∝ step^{-0.5}.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0
        self._lr          = 0.0

    def step(self):
        self._step += 1
        lr = self._compute_lr()
        self._lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _compute_lr(self) -> float:
        s = self._step
        w = self.warmup_steps
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * w ** (-1.5))

    @property
    def current_lr(self) -> float:
        return self._lr


# ──────────────────────────────────────────────────────────────────────────────
# 14. GREEDY DECODING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model:   Transformer,
    src:     Tensor,          # (1, S)
    src_mask: Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
    device:  torch.device,
) -> Tensor:
    """Returns the greedily decoded token sequence (1, T)."""
    memory  = model.encode(src, src_mask)
    ys      = torch.tensor([[bos_idx]], device=device)

    for _ in range(max_len - 1):
        T       = ys.size(1)
        tgt_mask = model.make_causal_mask(T, device)
        out      = model.decode(memory, ys, src_mask, tgt_mask)
        log_prob = model.generator(out[:, -1])          # last position
        next_tok = log_prob.argmax(dim=-1, keepdim=True)
        ys       = torch.cat([ys, next_tok], dim=1)
        if next_tok.item() == eos_idx:
            break

    return ys


# ──────────────────────────────────────────────────────────────────────────────
# 15. BEAM-SEARCH DECODING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search_decode(
    model:    Transformer,
    src:      Tensor,         # (1, S)
    src_mask: Tensor,
    max_len:  int,
    bos_idx:  int,
    eos_idx:  int,
    device:   torch.device,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> Tensor:
    """
    Beam search with length-normalisation penalty α.
    score = log P(Y|X) / ((5 + |Y|) / 6)^α   (Wu et al., 2016)
    Returns the best token sequence (1, T).
    """
    memory = model.encode(src, src_mask)   # (1, S, d)

    # Each beam: (cumulative_log_prob, token_ids_list)
    beams: list[tuple[float, list[int]]] = [(0.0, [bos_idx])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len - 1):
        all_candidates: list[tuple[float, list[int]]] = []

        for score, tokens in beams:
            if tokens[-1] == eos_idx:
                completed.append((score, tokens))
                continue

            ys = torch.tensor([tokens], device=device)
            T  = ys.size(1)
            tgt_mask = model.make_causal_mask(T, device)
            out      = model.decode(memory, ys, src_mask, tgt_mask)
            log_probs = model.generator(out[:, -1])[0]  # (V,)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidate = (score + lp, tokens + [tid])
                all_candidates.append(candidate)

        # Keep top beam_size beams (normalised for comparison)
        def _norm(item: tuple[float, list[int]]) -> float:
            s, toks = item
            lp = ((5 + len(toks)) / 6) ** length_penalty
            return s / lp

        all_candidates.sort(key=_norm, reverse=True)
        beams = all_candidates[:beam_size]

        if all(t[-1] == eos_idx for _, t in beams):
            completed.extend(beams)
            break

    if not completed:
        completed = beams

    best = max(completed, key=lambda item: item[0] / ((5 + len(item[1])) / 6) ** length_penalty)
    return torch.tensor([best[1]], device=device)


# ──────────────────────────────────────────────────────────────────────────────
# 16. MODEL FACTORY (paper defaults)
# ──────────────────────────────────────────────────────────────────────────────

def build_transformer(
    src_vocab: int,
    tgt_vocab: int,
    d_model:    int   = 512,
    num_heads:  int   = 8,
    num_layers: int   = 6,
    d_ff:       int   = 2048,
    dropout:    float = 0.1,
    max_len:    int   = 5000,
) -> Transformer:
    """Convenience constructor matching the base model in the paper."""
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
    )
    print(
        f"Transformer built — "
        f"params: {sum(p.numel() for p in model.parameters()):,} | "
        f"d_model={d_model}, heads={num_heads}, layers={num_layers}, d_ff={d_ff}"
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 17. TRAINING LOOP (minimal, self-contained demo)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleTranslationDataset(torch.utils.data.Dataset):
    """
    Tiny synthetic copy-task dataset:
    learn to copy source tokens to the target (a classic sanity-check task).
    """

    def __init__(self, vocab_size: int = 11, seq_len: int = 10, n_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.data = [
            torch.randint(1, vocab_size, (seq_len,))   # avoid 0 (pad)
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        seq = self.data[idx]
        return seq, seq   # src == tgt for copy task


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    srcs, tgts = zip(*batch)
    return torch.stack(srcs), torch.stack(tgts)


def run_demo():
    """Train a tiny Transformer on the copy task for a few steps."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB      = 11     # tokens 1–10, 0 = pad
    PAD_IDX    = 0
    BOS_IDX    = 1
    EOS_IDX    = 2
    SEQ_LEN    = 10
    BATCH_SIZE = 32
    N_STEPS    = 500
    D_MODEL    = 128
    NUM_HEADS  = 4
    NUM_LAYERS = 2
    D_FF       = 256

    print(f"\nDevice: {device}")
    print("=" * 60)
    print("  Transformer — 'Attention Is All You Need' (Vaswani 2017)")
    print("  Task: copy-task (learn to copy source → target)")
    print("=" * 60)

    # Model
    model = build_transformer(VOCAB, VOCAB, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, D_MODEL, warmup_steps=200)
    criterion = LabelSmoothingLoss(VOCAB, PAD_IDX, smoothing=0.1)

    # Data
    dataset    = SimpleTranslationDataset(VOCAB, SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # Training loop
    model.train()
    step        = 0
    total_loss  = 0.0
    start       = time.time()

    print(f"\n{'Step':>6}  {'Loss':>9}  {'LR':>12}  {'Elapsed':>8}")
    print("-" * 45)

    data_iter = iter(dataloader)

    while step < N_STEPS:
        try:
            src, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            src, tgt  = next(data_iter)

        src = src.to(device)
        tgt = tgt.to(device)

        # Decoder input: prepend BOS, drop last token
        tgt_in  = torch.cat([torch.full((tgt.size(0), 1), BOS_IDX, device=device), tgt[:, :-1]], dim=1)
        tgt_out = tgt

        # Masks
        src_mask = Transformer.make_pad_mask(src, PAD_IDX).to(device)
        T        = tgt_in.size(1)
        tgt_mask = (
            Transformer.make_pad_mask(tgt_in, PAD_IDX).to(device)
            | Transformer.make_causal_mask(T, device)
        )

        # Forward
        log_probs = model(src, tgt_in, src_mask, tgt_mask)   # (B, T, V)

        # Loss
        loss = criterion(
            log_probs.reshape(-1, VOCAB),
            tgt_out.reshape(-1),
        )
        tokens = (tgt_out != PAD_IDX).sum().item()
        loss_per_tok = loss / tokens

        # Backward
        optimizer.zero_grad()
        loss_per_tok.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()
        optimizer.step()

        total_loss += loss_per_tok.item()
        step       += 1

        if step % 50 == 0:
            avg   = total_loss / 50
            elapsed = time.time() - start
            print(f"{step:>6}  {avg:>9.4f}  {scheduler.current_lr:>12.2e}  {elapsed:>7.1f}s")
            total_loss = 0.0

    print("\nTraining complete.\n")

    # Quick inference demo
    print("Inference demo (greedy decode — should copy source tokens):")
    model.eval()
    demo_src = torch.tensor([[3, 5, 7, 2, 9, 4, 6, 8, 1, 10]], device=device)
    src_mask  = Transformer.make_pad_mask(demo_src, PAD_IDX).to(device)
    output    = greedy_decode(model, demo_src, src_mask, SEQ_LEN + 2, BOS_IDX, EOS_IDX, device)

    print(f"  Source : {demo_src[0].tolist()}")
    print(f"  Output : {output[0].tolist()}")

    # Save
    ckpt_path = "transformer_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": {
        "src_vocab": VOCAB, "tgt_vocab": VOCAB,
        "d_model": D_MODEL, "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS, "d_ff": D_FF,
    }}, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo()
