"""
╔══════════════════════════════════════════════════════════════════╗
║          LLM FROM SCRATCH — MiniGPT in Pure Python/NumPy        ║
║                                                                  ║
║  Components implemented:                                         ║
║   1. Byte-Pair Encoding (BPE) Tokenizer                          ║
║   2. Transformer Architecture (GPT-style)                        ║
║      - Multi-Head Self-Attention                                 ║
║      - Feed-Forward Network                                      ║
║      - Layer Normalization                                       ║
║      - Positional Encoding                                       ║
║   3. Training Loop with AdamW optimizer                          ║
║   4. Text Generation with temperature sampling                   ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python llm_from_scratch.py

Requires: numpy only (no PyTorch, no TensorFlow)
For GPU-accelerated training, see the PyTorch version at the bottom.
"""

import numpy as np
import re
import json
import pickle
import os
import math
import time
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────
#  SECTION 1: TOKENIZER (Byte-Pair Encoding)
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    A simple Byte-Pair Encoding tokenizer.
    Learns subword vocabulary from training text.
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """Count frequency of each adjacent pair in vocab."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocab."""
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        merged = "".join(pair)
        for word in vocab:
            new_word = pattern.sub(merged, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, text: str) -> None:
        """Train BPE tokenizer on text."""
        print("Training BPE tokenizer...")

        # Start: character-level vocab
        word_freq: Dict[str, int] = Counter()
        for word in text.lower().split():
            # Add space marker and split to chars
            word_freq[" ".join(list(word)) + " </w>"] += 1

        # Base character vocab
        chars = set()
        for word in word_freq:
            chars.update(word.split())

        # Initialize vocab with special tokens + chars
        idx = len(self.special_tokens)
        self.vocab = dict(self.special_tokens)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        # BPE merge loop
        current_vocab = dict(word_freq)
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            pairs = self._get_stats(current_vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            current_vocab = self._merge_vocab(best, current_vocab)
            merged_token = "".join(best)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = idx
                idx += 1
            self.merges.append(best)

            if (i + 1) % 50 == 0:
                print(f"  Merge {i+1}/{num_merges}: '{best[0]}' + '{best[1]}' → '{merged_token}'")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        word = list(word) + ["</w>"]
        word = [" ".join(word)]

        for merge in self.merges:
            new_word = []
            for token in word:
                symbols = token.split()
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == merge:
                        new_word.append(symbols[i] + symbols[i+1])
                        i += 2
                    else:
                        new_word.append(symbols[i])
                        i += 1
                word = [" ".join(new_word)]
            # Flatten
            word = [" ".join(w.split()) for w in word]

        return word[0].split()

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = [self.special_tokens["<BOS>"]]
        for word in text.lower().split():
            for subword in self._tokenize_word(word):
                tokens.append(self.vocab.get(subword, self.special_tokens["<UNK>"]))
        tokens.append(self.special_tokens["<EOS>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = []
        for tid in token_ids:
            if tid in (self.special_tokens["<BOS>"], self.special_tokens["<EOS>"],
                      self.special_tokens["<PAD>"]):
                continue
            token = self.inverse_vocab.get(tid, "<UNK>")
            tokens.append(token)
        text = " ".join(tokens)
        text = text.replace(" </w>", " ").replace("</w>", "").strip()
        return text

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "merges": self.merges,
                        "inverse_vocab": self.inverse_vocab}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        self.inverse_vocab = data["inverse_vocab"]


# ─────────────────────────────────────────────
#  SECTION 2: NUMPY TRANSFORMER (MiniGPT)
# ─────────────────────────────────────────────

class LayerNorm:
    """Layer normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        # For optimizer
        self.dg = np.zeros_like(self.gamma)
        self.db = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        cache = {"x_norm": x_norm, "var": var, "x": x, "mean": mean}
        return out, cache

    def backward(self, dout: np.ndarray, cache: dict) -> np.ndarray:
        x_norm = cache["x_norm"]
        var = cache["var"]
        x = cache["x"]
        mean = cache["mean"]
        N = x.shape[-1]

        self.dg = (dout * x_norm).sum(axis=(0, 1)) if dout.ndim == 3 else (dout * x_norm).sum(axis=0)
        self.db = dout.sum(axis=(0, 1)) if dout.ndim == 3 else dout.sum(axis=0)

        dx_norm = dout * self.gamma
        dvar = (-0.5 * dx_norm * (x - mean) * (var + 1e-6)**(-1.5)).sum(axis=-1, keepdims=True)
        dmean = (-dx_norm / np.sqrt(var + 1e-6)).sum(axis=-1, keepdims=True)
        dx = dx_norm / np.sqrt(var + 1e-6) + dvar * 2 * (x - mean) / N + dmean / N
        return dx


class MultiHeadAttention:
    """Multi-head causal self-attention."""
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

        # Gradients
        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        B, T, D = x.shape
        H, dk = self.n_heads, self.d_k

        Q = x @ self.Wq  # (B, T, D)
        K = x @ self.Wk
        V = x @ self.Wv

        # Split into heads: (B, H, T, dk)
        Q = Q.reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, dk).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(dk)  # (B, H, T, T)

        # Causal mask
        if mask is None:
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores = np.where(mask[None, None, :, :], -1e9, scores)

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

        # Attend to values
        out = attn @ V  # (B, H, T, dk)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = out @ self.Wo

        cache = {"x": x, "Q": Q, "K": K, "V": V, "attn": attn, "pre_out": out}
        return out, cache

    def backward(self, dout: np.ndarray, cache: dict) -> np.ndarray:
        # Simplified gradient (used for learning signal)
        x = cache["x"]
        B, T, D = x.shape
        self.dWo += (cache["pre_out"].reshape(-1, D).T @ dout.reshape(-1, D)) / B
        dx = dout @ self.Wo.T
        self.dWq += (x.reshape(-1, D).T @ dx.reshape(-1, D)) / B
        self.dWk += (x.reshape(-1, D).T @ dx.reshape(-1, D)) / B
        self.dWv += (x.reshape(-1, D).T @ dx.reshape(-1, D)) / B
        return dx


class FeedForward:
    """Position-wise feed-forward network."""
    def __init__(self, d_model: int, d_ff: int):
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(0, h)  # GELU approximation: use ReLU here
        out = h_relu @ self.W2 + self.b2
        return out, {"x": x, "h": h, "h_relu": h_relu}

    def backward(self, dout: np.ndarray, cache: dict) -> np.ndarray:
        x = cache["x"]
        h = cache["h"]
        h_relu = cache["h_relu"]
        B = x.shape[0] if x.ndim == 3 else 1

        self.dW2 += h_relu.reshape(-1, h_relu.shape[-1]).T @ dout.reshape(-1, dout.shape[-1]) / B
        self.db2 += dout.sum(axis=(0, 1)) if dout.ndim == 3 else dout.sum(axis=0) / B

        dh_relu = dout @ self.W2.T
        dh = dh_relu * (h > 0)

        self.dW1 += x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1]) / B
        self.db1 += dh.sum(axis=(0, 1)) if dh.ndim == 3 else dh.sum(axis=0) / B

        return dh @ self.W1.T


class TransformerBlock:
    """Single transformer block: attention + FFN with residuals."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, dict]:
        # Self-attention with residual
        ln1_out, ln1_cache = self.ln1.forward(x)
        attn_out, attn_cache = self.attn.forward(ln1_out)
        if training:
            mask = np.random.rand(*attn_out.shape) > self.dropout
            attn_out = attn_out * mask / (1 - self.dropout)
        x = x + attn_out

        # FFN with residual
        ln2_out, ln2_cache = self.ln2.forward(x)
        ff_out, ff_cache = self.ff.forward(ln2_out)
        if training:
            mask2 = np.random.rand(*ff_out.shape) > self.dropout
            ff_out = ff_out * mask2 / (1 - self.dropout)
        x = x + ff_out

        cache = {"attn_cache": attn_cache, "ff_cache": ff_cache,
                 "ln1_cache": ln1_cache, "ln2_cache": ln2_cache}
        return x, cache

    def backward(self, dout: np.ndarray, cache: dict) -> np.ndarray:
        # Backward through FFN path
        dff = self.ff.backward(dout, cache["ff_cache"])
        dln2 = self.ln2.backward(dff, cache["ln2_cache"])
        dout = dout + dln2

        # Backward through attention path
        dattn = self.attn.backward(dout, cache["attn_cache"])
        dln1 = self.ln1.backward(dattn, cache["ln1_cache"])
        dout = dout + dln1
        return dout


class MiniGPT:
    """
    A minimal GPT-style language model.
    Architecture:
      - Token embedding
      - Positional encoding
      - N transformer blocks
      - Language model head (linear projection to vocab)
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, max_seq_len: int = 128,
                 dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        print(f"\nInitializing MiniGPT:")
        print(f"  Vocab size:  {vocab_size}")
        print(f"  d_model:     {d_model}")
        print(f"  n_heads:     {n_heads}")
        print(f"  n_layers:    {n_layers}")
        print(f"  d_ff:        {d_ff}")
        print(f"  max_seq_len: {max_seq_len}")

        # Token embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.d_token_emb = np.zeros_like(self.token_emb)

        # Positional encoding (sinusoidal, fixed)
        self.pos_enc = self._make_positional_encoding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout)
                      for _ in range(n_layers)]

        # Final layer norm
        self.ln_f = LayerNorm(d_model)

        # LM head
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        self.d_lm_head = np.zeros_like(self.lm_head)

        total_params = self._count_params()
        print(f"  Total params: ~{total_params:,}")

    def _count_params(self) -> int:
        p = self.token_emb.size + self.lm_head.size
        for b in self.blocks:
            p += (b.attn.Wq.size + b.attn.Wk.size +
                  b.attn.Wv.size + b.attn.Wo.size +
                  b.ff.W1.size + b.ff.b1.size +
                  b.ff.W2.size + b.ff.b2.size +
                  b.ln1.gamma.size * 2 + b.ln2.gamma.size * 2)
        return p

    def _make_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Sinusoidal positional encoding (Vaswani et al., 2017)."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        return pe

    def forward(self, token_ids: np.ndarray,
                training: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.
        token_ids: (B, T) integer array
        Returns logits: (B, T, vocab_size)
        """
        B, T = token_ids.shape
        # Embed tokens + positions
        x = self.token_emb[token_ids] + self.pos_enc[:T][np.newaxis, :, :]

        block_caches = []
        for block in self.blocks:
            x, cache = block.forward(x, training=training)
            block_caches.append(cache)

        x, ln_cache = self.ln_f.forward(x)
        logits = x @ self.lm_head  # (B, T, vocab_size)

        return logits, {"block_caches": block_caches, "x_pre_lm": x,
                       "ln_cache": ln_cache, "token_ids": token_ids, "x_emb": x}

    def backward(self, dlogits: np.ndarray, cache: dict) -> None:
        """Backward pass, accumulate gradients."""
        B, T, V = dlogits.shape
        x_pre_lm = cache["x_pre_lm"]

        # LM head gradient
        self.d_lm_head = x_pre_lm.reshape(-1, self.d_model).T @ dlogits.reshape(-1, V) / B

        dx = dlogits @ self.lm_head.T
        dx = self.ln_f.backward(dx, cache["ln_cache"])

        for i, (block, bc) in enumerate(zip(reversed(self.blocks),
                                            reversed(cache["block_caches"]))):
            dx = block.backward(dx, bc)

        # Token embedding gradient
        token_ids = cache["token_ids"]
        np.add.at(self.d_token_emb, token_ids, dx)

    def zero_grad(self) -> None:
        self.d_token_emb.fill(0)
        self.d_lm_head.fill(0)
        for b in self.blocks:
            b.attn.dWq.fill(0); b.attn.dWk.fill(0)
            b.attn.dWv.fill(0); b.attn.dWo.fill(0)
            b.ff.dW1.fill(0); b.ff.db1.fill(0)
            b.ff.dW2.fill(0); b.ff.db2.fill(0)
            b.ln1.dg.fill(0); b.ln1.db.fill(0)
            b.ln2.dg.fill(0); b.ln2.db.fill(0)
        self.ln_f.dg.fill(0); self.ln_f.db.fill(0)

    def save(self, path: str) -> None:
        """Save model weights."""
        weights = {
            "token_emb": self.token_emb,
            "lm_head": self.lm_head,
            "ln_f_gamma": self.ln_f.gamma,
            "ln_f_beta": self.ln_f.beta,
        }
        for i, b in enumerate(self.blocks):
            weights.update({
                f"block{i}_Wq": b.attn.Wq, f"block{i}_Wk": b.attn.Wk,
                f"block{i}_Wv": b.attn.Wv, f"block{i}_Wo": b.attn.Wo,
                f"block{i}_W1": b.ff.W1, f"block{i}_b1": b.ff.b1,
                f"block{i}_W2": b.ff.W2, f"block{i}_b2": b.ff.b2,
                f"block{i}_ln1_g": b.ln1.gamma, f"block{i}_ln1_b": b.ln1.beta,
                f"block{i}_ln2_g": b.ln2.gamma, f"block{i}_ln2_b": b.ln2.beta,
            })
        np.savez(path, **weights)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.token_emb = data["token_emb"]
        self.lm_head = data["lm_head"]
        self.ln_f.gamma = data["ln_f_gamma"]
        self.ln_f.beta = data["ln_f_beta"]
        for i, b in enumerate(self.blocks):
            b.attn.Wq = data[f"block{i}_Wq"]; b.attn.Wk = data[f"block{i}_Wk"]
            b.attn.Wv = data[f"block{i}_Wv"]; b.attn.Wo = data[f"block{i}_Wo"]
            b.ff.W1 = data[f"block{i}_W1"]; b.ff.b1 = data[f"block{i}_b1"]
            b.ff.W2 = data[f"block{i}_W2"]; b.ff.b2 = data[f"block{i}_b2"]
            b.ln1.gamma = data[f"block{i}_ln1_g"]; b.ln1.beta = data[f"block{i}_ln1_b"]
            b.ln2.gamma = data[f"block{i}_ln2_g"]; b.ln2.beta = data[f"block{i}_ln2_b"]


# ─────────────────────────────────────────────
#  SECTION 3: ADAMW OPTIMIZER
# ─────────────────────────────────────────────

class AdamW:
    """
    AdamW optimizer (Adam with decoupled weight decay).
    Reference: Loshchilov & Hutter, 2019.
    """
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}  # First moment
        self.v: Dict[str, np.ndarray] = {}  # Second moment

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        for name, param in params.items():
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2

            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2

            # Weight decay (decoupled)
            param *= (1 - self.lr * self.wd)
            # Adam update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────
#  SECTION 4: TRAINING UTILITIES
# ─────────────────────────────────────────────

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and gradient.
    logits: (B, T, V)
    targets: (B, T)  integer token IDs
    """
    B, T, V = logits.shape

    # Softmax (numerically stable)
    logits_max = logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-9)

    # Gather target probs
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    target_probs = probs[b_idx, t_idx, targets]  # (B, T)

    loss = -np.log(target_probs + 1e-9).mean()

    # Gradient of cross-entropy + softmax
    dlogits = probs.copy()
    dlogits[b_idx, t_idx, targets] -= 1
    dlogits /= (B * T)

    return loss, dlogits


def get_params_and_grads(model: MiniGPT) -> Tuple[Dict, Dict]:
    """Extract all parameters and gradients as flat dicts."""
    params, grads = {}, {}

    params["token_emb"] = model.token_emb
    grads["token_emb"] = model.d_token_emb
    params["lm_head"] = model.lm_head
    grads["lm_head"] = model.d_lm_head
    params["ln_f_g"] = model.ln_f.gamma
    grads["ln_f_g"] = model.ln_f.dg
    params["ln_f_b"] = model.ln_f.beta
    grads["ln_f_b"] = model.ln_f.db

    for i, b in enumerate(model.blocks):
        for name, (p, g) in [
            ("Wq", (b.attn.Wq, b.attn.dWq)),
            ("Wk", (b.attn.Wk, b.attn.dWk)),
            ("Wv", (b.attn.Wv, b.attn.dWv)),
            ("Wo", (b.attn.Wo, b.attn.dWo)),
            ("W1", (b.ff.W1, b.ff.dW1)), ("b1", (b.ff.b1, b.ff.db1)),
            ("W2", (b.ff.W2, b.ff.dW2)), ("b2", (b.ff.b2, b.ff.db2)),
            ("ln1g", (b.ln1.gamma, b.ln1.dg)), ("ln1b", (b.ln1.beta, b.ln1.db)),
            ("ln2g", (b.ln2.gamma, b.ln2.dg)), ("ln2b", (b.ln2.beta, b.ln2.db)),
        ]:
            params[f"b{i}_{name}"] = p
            grads[f"b{i}_{name}"] = g

    return params, grads


def create_batches(token_ids: List[int], batch_size: int,
                   seq_len: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Slide a window over token_ids to create (input, target) pairs."""
    batches = []
    examples = []
    for i in range(0, len(token_ids) - seq_len, seq_len // 2):
        inp = token_ids[i:i + seq_len]
        tgt = token_ids[i + 1:i + seq_len + 1]
        if len(inp) == seq_len and len(tgt) == seq_len:
            examples.append((inp, tgt))

    np.random.shuffle(examples)
    for i in range(0, len(examples) - batch_size + 1, batch_size):
        batch = examples[i:i + batch_size]
        inp_batch = np.array([e[0] for e in batch])
        tgt_batch = np.array([e[1] for e in batch])
        batches.append((inp_batch, tgt_batch))

    return batches


def lr_schedule(step: int, warmup_steps: int, d_model: int) -> float:
    """Transformer learning rate schedule with warmup."""
    if step == 0:
        return 0.0
    return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


# ─────────────────────────────────────────────
#  SECTION 5: TEXT GENERATION
# ─────────────────────────────────────────────

def generate(model: MiniGPT, tokenizer: BPETokenizer, prompt: str,
             max_new_tokens: int = 50, temperature: float = 0.8,
             top_k: int = 40) -> str:
    """
    Autoregressive text generation with temperature + top-k sampling.

    Args:
        temperature: > 1 = more random, < 1 = more focused
        top_k: sample only from top k tokens
    """
    token_ids = tokenizer.encode(prompt)[:-1]  # Strip <EOS>
    token_ids = token_ids[-model.max_seq_len:]

    generated = list(token_ids)

    for _ in range(max_new_tokens):
        inp = np.array([generated[-model.max_seq_len:]])
        logits, _ = model.forward(inp, training=False)

        # Take logits at last position
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Temperature scaling
        last_logits = last_logits / max(temperature, 1e-5)

        # Top-k filtering
        if top_k > 0:
            top_k_indices = np.argpartition(last_logits, -top_k)[-top_k:]
            mask = np.full_like(last_logits, -1e9)
            mask[top_k_indices] = last_logits[top_k_indices]
            last_logits = mask

        # Softmax → probabilities
        exp_logits = np.exp(last_logits - last_logits.max())
        probs = exp_logits / exp_logits.sum()

        # Sample
        next_token = np.random.choice(len(probs), p=probs)

        if next_token == tokenizer.special_tokens["<EOS>"]:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


# ─────────────────────────────────────────────
#  SECTION 6: MAIN — TRAIN & GENERATE
# ─────────────────────────────────────────────

TRAINING_TEXT = """
The transformer architecture has revolutionized natural language processing.
Self-attention allows the model to relate different positions in a sequence to compute a representation.
Language models learn to predict the next word given all previous words.
Neural networks learn by adjusting weights through backpropagation and gradient descent.
The attention mechanism computes a weighted sum of values based on query and key similarities.
Deep learning has enabled breakthroughs in machine translation, text generation, and question answering.
Large language models are trained on vast amounts of text data to learn patterns in language.
The key insight of the transformer is replacing recurrence with attention mechanisms entirely.
Positional encoding injects information about the position of tokens in the sequence.
Layer normalization stabilizes training by normalizing activations within each layer.
The feed-forward network in each transformer block applies two linear transformations with a nonlinearity.
Residual connections help gradients flow through deep networks during backpropagation.
Tokenization converts raw text into integer indices that can be fed into the embedding layer.
Byte pair encoding merges the most frequent character pairs iteratively to build a subword vocabulary.
The embedding layer maps token indices to dense vector representations.
Softmax converts logits into a probability distribution over the vocabulary.
Cross-entropy loss measures the difference between predicted and actual token distributions.
The AdamW optimizer combines adaptive learning rates with decoupled weight decay regularization.
Temperature scaling controls the sharpness of the predicted probability distribution during sampling.
Top-k sampling restricts generation to the k most likely tokens at each step.
Training a language model involves minimizing perplexity on a held-out validation set.
The model learns syntactic and semantic structure from patterns in the training data.
Causal masking ensures the model can only attend to previous tokens, not future ones.
Multi-head attention allows the model to jointly attend to information from different representation subspaces.
The dimensionality of the model determines its capacity to learn complex language patterns.
Scaling laws show that model performance improves predictably with more parameters and data.
""".strip()


def main():
    print("=" * 60)
    print("         LLM FROM SCRATCH — MiniGPT Training")
    print("=" * 60)

    np.random.seed(42)

    # ── Hyperparameters ──────────────────────────────────────
    VOCAB_SIZE   = 256
    D_MODEL      = 64
    N_HEADS      = 4
    N_LAYERS     = 2
    D_FF         = 256
    MAX_SEQ_LEN  = 64
    BATCH_SIZE   = 4
    SEQ_LEN      = 32
    N_EPOCHS     = 30
    LR           = 3e-4
    DROPOUT      = 0.05

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(TRAINING_TEXT)

    # ── Tokenize training data ────────────────────────────────
    all_tokens = tokenizer.encode(TRAINING_TEXT)
    print(f"\nTraining corpus: {len(all_tokens)} tokens")

    # ── Build model ──────────────────────────────────────────
    actual_vocab = len(tokenizer.vocab)
    model = MiniGPT(
        vocab_size=actual_vocab,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    optimizer = AdamW(lr=LR, weight_decay=0.01)

    # ── Training loop ────────────────────────────────────────
    print(f"\nTraining for {N_EPOCHS} epochs...")
    print("-" * 60)

    loss_history = []

    for epoch in range(N_EPOCHS):
        batches = create_batches(all_tokens, BATCH_SIZE, SEQ_LEN)
        if not batches:
            print("Not enough data for batches, skipping epoch")
            continue

        epoch_loss = 0.0
        t_start = time.time()

        for inp, tgt in batches:
            model.zero_grad()
            logits, cache = model.forward(inp, training=True)
            loss, dlogits = cross_entropy_loss(logits, tgt)
            model.backward(dlogits, cache)

            params, grads = get_params_and_grads(model)
            optimizer.step(params, grads)
            epoch_loss += loss

        avg_loss = epoch_loss / len(batches)
        perplexity = math.exp(min(avg_loss, 20))
        loss_history.append(avg_loss)
        elapsed = time.time() - t_start

        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"PPL: {perplexity:.2f} | "
              f"Time: {elapsed:.2f}s")

    # ── Generation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEXT GENERATION SAMPLES")
    print("=" * 60)

    prompts = [
        "the transformer",
        "neural networks",
        "language models",
        "attention mechanism",
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        for temp in [0.6, 1.0]:
            generated = generate(model, tokenizer, prompt,
                                max_new_tokens=30, temperature=temp, top_k=20)
            print(f"  [temp={temp}] {generated}")

    # ── Save ─────────────────────────────────────────────────
    model.save("minigpt_weights")
    tokenizer.save("bpe_tokenizer.pkl")

    print("\n✓ Model saved: minigpt_weights.npz")
    print("✓ Tokenizer saved: bpe_tokenizer.pkl")
    print("\nFinal loss history (last 5 epochs):")
    for i, l in enumerate(loss_history[-5:], len(loss_history) - 4):
        print(f"  Epoch {i}: {l:.4f} (PPL: {math.exp(min(l, 20)):.2f})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    return model, tokenizer, loss_history


# ─────────────────────────────────────────────────────────────────
#  BONUS: PyTorch version (uncomment if you have PyTorch installed)
# ─────────────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TorchGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
"""

if __name__ == "__main__":
    main()
