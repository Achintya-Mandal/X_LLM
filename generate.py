"""
Text generation
===============
After training, the model has learned a probability distribution over the next
character given a context window.  We generate by:

  1. Feed a prompt (or an empty start) into the model.
  2. Sample the next character from the output distribution.
  3. Append it to the sequence and repeat.

Two knobs that affect output quality:
  --temperature  lower → more focused/repetitive, higher → more creative/random
  --top_k        only sample from the top-k most likely next tokens, ignoring
                 the long tail of very unlikely characters
"""

import argparse
import pickle

import torch

from config import BLOCK_SIZE, N_EMBD, N_HEADS, N_LAYERS, DEVICE
from model import GPT


def load_model_and_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = GPT(
        vocab_size  = tokenizer.vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        block_size  = BLOCK_SIZE,
        dropout     = 0.0,   # dropout is only for training; disable at inference
    ).to(DEVICE)

    state = torch.load("model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt:         str   = "",
    max_new_tokens: int   = 300,
    temperature:    float = 0.8,
    top_k:          int   = 40,
) -> str:
    if prompt:
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)
    else:
        # Start from token 0 (unconditional generation — model picks its own starting point)
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    output = model.generate(idx, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k)
    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from the trained GPT model")
    parser.add_argument("--prompt",      default="",    help="Seed text (empty = unconditional)")
    parser.add_argument("--tokens",      type=int,   default=300,  help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8,  help="Sampling temperature (default 0.8)")
    parser.add_argument("--top_k",       type=int,   default=40,   help="Top-k sampling (default 40)")
    args = parser.parse_args()

    print("Loading model…")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Generating {args.tokens} characters  (temperature={args.temperature}, top_k={args.top_k})\n")
    print("=" * 60)
    text = generate(model, tokenizer,
                    prompt         = args.prompt,
                    max_new_tokens = args.tokens,
                    temperature    = args.temperature,
                    top_k          = args.top_k)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
