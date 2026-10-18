"""
GPT Architecture
================
A GPT (Generatively Pre-trained Transformer) is a stack of transformer blocks
that predicts the next token given all previous tokens.

Data flow:
  token IDs  →  embeddings  →  N transformer blocks  →  layer norm  →  logits over vocab

Each transformer block:
  x  →  LayerNorm  →  Multi-Head Self-Attention  →  (+x)  →  LayerNorm  →  FeedForward  →  (+x)

The (+x) parts are "residual connections" — they let the gradient flow straight
back to early layers during training, which is what makes deep networks trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention head
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """
    One self-attention head.

    Self-attention lets each token decide which other tokens to "look at" and
    how much weight to give them. This is what allows the model to learn that
    "it" refers to the noun from 10 words earlier, for example.

    Three projections are learned:
      Q (query)  — "what am I looking for?"
      K (key)    — "what do I contain?"
      V (value)  — "what do I send to the output?"

    Attention score = softmax( Q @ K^T / sqrt(head_size) ) @ V
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: a lower-triangular matrix of 1s.
        # Token at position t may only attend to positions 0 … t (no peeking at the future).
        # Registered as a buffer so it moves to GPU with the model but isn't a trainable parameter.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape          # batch, time (sequence length), channels (n_embd)

        k = self.key(x)            # (B, T, head_size)
        q = self.query(x)          # (B, T, head_size)

        # Compute raw attention scores and scale.
        # Without the 1/sqrt(head_size) factor, dot products grow large for big head_size,
        # pushing softmax into near-zero gradient regions ("saturation").
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Apply causal mask: future positions become -inf, so softmax assigns them ~0 weight.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)             # (B, T, T) — now sums to 1 per row
        wei = self.dropout(wei)

        v   = self.value(x)                      # (B, T, head_size)
        out = wei @ v                            # (B, T, head_size)
        return out


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Run H attention heads in parallel and concatenate their outputs.

    Why multiple heads? Each head can specialize in a different type of
    relationship (syntax, coreference, positional proximity, …). More heads
    = richer representations, at the cost of compute.

    After concatenation the combined output is projected back to n_embd.
    """

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)   # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


# ---------------------------------------------------------------------------
# Feed-forward network (FFN)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    A small two-layer MLP applied independently to every token position.

    If attention is the mechanism for tokens to communicate with each other,
    the FFN is where each token "thinks" by itself. The 4× expansion factor
    (n_embd → 4*n_embd → n_embd) comes from the original Transformer paper
    and is retained in virtually all modern LLMs.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),                          # smoother alternative to ReLU; used in GPT-2+
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    One transformer block = Multi-Head Attention + FeedForward, each wrapped
    in a Pre-LayerNorm residual connection.

    Pre-norm (LayerNorm *before* the sublayer) trains more stably than the
    original post-norm design from "Attention Is All You Need".
    """

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads   # each head gets an equal slice of the embedding
        self.sa   = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # attention sub-layer
        x = x + self.ffwd(self.ln2(x))  # feed-forward sub-layer
        return x


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    A small GPT language model.

    Given a sequence of token IDs, it predicts the probability distribution
    over the next token at every position (teacher-forced training).
    At inference, we sample autoregressively: feed output back as new input.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd:     int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        block_size: int   = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        # Learned position embeddings: the model learns a different vector for
        # each position 0…block_size-1. Simpler than sinusoidal and equally effective.
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)         # final layer norm before the head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection.
        # Intuition: the token embedding and the output layer are both "token ↔ vector"
        # mappings; tying them halves the parameters here and consistently improves loss.
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb   # broadcast: pos_emb added to every item in the batch

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # Cross-entropy loss: flatten batch and time into one dimension
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample `max_new_tokens` new tokens.

        temperature < 1  → more focused / repetitive output
        temperature > 1  → more diverse / chaotic output
        top_k            → only sample from the k most likely next tokens
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]     # never exceed context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx      = torch.cat((idx, idx_next), dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
