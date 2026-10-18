"""
Training loop
=============
How a language model learns:

1. Sample a random chunk of text (the "input") and the chunk shifted by one
   character (the "target").  Example with block_size=4:
     input:   [H, e, l, l]
     target:  [e, l, l, o]

2. Feed the input through the model → logits (un-normalized scores for every
   character at every position).

3. Compute cross-entropy loss between logits and targets.  Loss measures how
   surprised the model is by the correct next character.

4. Backpropagate: compute gradients of the loss w.r.t. every parameter.

5. AdamW optimizer nudges each parameter slightly in the direction that reduces
   the loss.

Repeat millions of times.  Loss starts around ln(vocab_size) ≈ 4.2 (random
guessing) and should drop below 1.5 on Shakespeare after ~5000 steps.
"""

import os
import pickle
import urllib.request

import torch

from config import (
    BATCH_SIZE, BLOCK_SIZE, N_EMBD, N_HEADS, N_LAYERS, DROPOUT,
    LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS,
    DEVICE, DATA_URL, DATA_FILE,
)
from tokenizer import CharTokenizer
from model import GPT


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("Downloading Shakespeare dataset (~1 MB)...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print(f"Saved to {DATA_FILE}")


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    # Random starting positions for each example in the batch
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([data[i    : i + BLOCK_SIZE    ] for i in ix])
    y  = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()   # disable gradient tracking — we only need forward passes here
def estimate_loss(
    model:      GPT,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
) -> dict[str, float]:
    """Average loss over EVAL_ITERS batches for a stable estimate."""
    model.eval()
    losses: dict[str, float] = {}
    for split in ('train', 'val'):
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download_data()

    text = open(DATA_FILE, encoding='utf-8').read()
    print(f"Dataset: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    print(f"  {''.join(sorted(tokenizer.stoi))}\n")

    # Persist tokenizer so generate.py can reuse the same vocabulary mapping
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Encode the full text into a flat tensor of token IDs, then split 90/10
    data  = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n     = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Train: {len(train_data):,} tokens  |  Val: {len(val_data):,} tokens\n")

    model = GPT(
        vocab_size  = tokenizer.vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        block_size  = BLOCK_SIZE,
        dropout     = DROPOUT,
    ).to(DEVICE)

    print(f"Parameters : {model.num_parameters():,}")
    print(f"Device     : {DEVICE}\n")
    print(f"{'Step':>6}  {'Train loss':>10}  {'Val loss':>8}")
    print("-" * 32)

    # AdamW = Adam with decoupled weight decay.  Standard choice for transformers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_ITERS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"{step:>6}  {losses['train']:>10.4f}  {losses['val']:>8.4f}")

        x, y = get_batch('train', train_data, val_data)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: cap gradient norm at 1.0 to prevent the occasional
        # very large gradient from destabilizing training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.save(model.state_dict(), "model.pt")
    print("\nModel saved → model.pt")
    print("Run  python generate.py  to sample from it.")


if __name__ == "__main__":
    main()
