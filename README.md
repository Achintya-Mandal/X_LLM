# X_LLM — Build an LLM from Scratch

A hands-on learning project that builds a GPT language model from the ground up using PyTorch — no libraries that hide the internals, no pre-built transformer modules.

Every file is written to be **read and understood**, not just run. The goal is to see exactly how each piece of a real LLM works: how text becomes numbers, how attention lets tokens talk to each other, how the network learns from its own mistakes, and how you sample coherent text from a probability distribution.

~207k parameters. Trains on Shakespeare in ~5 minutes on CPU.

---

## How it works

```
Raw text
   │
   ▼
CharTokenizer          each unique character → integer ID
   │
   ▼
Token + Position       learned embedding vectors
Embeddings
   │
   ▼
Transformer Block ×4   Multi-Head Self-Attention + Feed-Forward
   │                   (with residual connections + LayerNorm)
   ▼
LM Head                linear projection → probability over vocabulary
   │
   ▼
Next character
```

The model is trained to predict the next character at every position. After 5000 steps on ~1M characters of Shakespeare, it generates plausible (if nonsensical) Elizabethan-sounding text.

---

## Project structure

```
X_LLM/
├── config.py          hyperparameters (block size, layers, embedding dim…)
├── tokenizer.py       character-level tokenizer  (char ↔ int)
├── model.py           GPT architecture (Head, MultiHeadAttention, Block, GPT)
├── train.py           training loop — downloads data, trains, saves model.pt
├── generate.py        CLI text generation from a saved model
├── server.py          FastAPI server exposing /generate endpoint
├── static/
│   └── index.html     browser UI with prompt input and sampling controls
└── requirements.txt
```

---

## Quick start

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/Achintya-Mandal/X_LLM.git
cd X_LLM
python -m venv .venv
```

**2. Activate the environment**

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
# CPU-only PyTorch (faster install, fine for this model size)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn
```

**4. Train**

```bash
python train.py
```

Downloads ~1 MB of Shakespeare, trains for 5000 steps, saves `model.pt` and `tokenizer.pkl`.

Expected output:

```
Dataset: 1,115,394 characters
Vocabulary: 65 unique characters

  Step  Train loss  Val loss
--------------------------------
     0      4.1695    4.1696
   500      2.4602    2.4611
  1000      2.2821    2.2943
  ...
  5000      1.4812    1.5103

Model saved → model.pt
```

**5. Generate text (CLI)**

```bash
python generate.py --prompt "To be or not to be"
python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.7
python generate.py --temperature 1.2 --top_k 0    # unconditional, wild output
```

**6. Start the server + UI**

```bash
python server.py
```

Open **http://localhost:8000** in your browser.

---

## Browser UI

The web interface lets you adjust sampling parameters interactively without touching the command line.

- **Prompt** — seed text for the model (leave empty for unconditional generation)
- **Max tokens** — number of characters to generate
- **Temperature** — `0.3` = focused/repetitive, `1.2` = creative/chaotic
- **Top-K** — sample only from the K most likely next characters

The API is also available at **http://localhost:8000/docs** (auto-generated Swagger UI).

---

## Key concepts

### Loss
Cross-entropy loss measures how surprised the model is by the correct next character. It starts at `ln(65) ≈ 4.17` (random guessing) and should reach `~1.45` by step 5000.

### Self-attention
Each token computes a query (what am I looking for?), key (what do I contain?), and value (what do I send?). Attention scores determine how much each token should attend to every other token in its context window. A causal mask prevents tokens from attending to future positions.

### Residual connections
Every transformer block adds its output back to its input: `x = x + sublayer(x)`. This lets gradients flow directly to early layers, making deep networks trainable.

### Temperature & Top-K sampling
At generation time, logits are divided by temperature before softmax. Lower temperature sharpens the distribution (more deterministic); higher temperature flattens it (more random). Top-K restricts sampling to only the K most likely tokens, avoiding very low-probability noise.

---

## Hyperparameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| `BLOCK_SIZE` | 64 | Context window — tokens the model sees at once |
| `N_EMBD` | 64 | Embedding dimension |
| `N_HEADS` | 4 | Parallel attention heads per block |
| `N_LAYERS` | 4 | Number of stacked transformer blocks |
| `DROPOUT` | 0.1 | Dropout rate during training |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 3e-4 | AdamW learning rate |
| `MAX_ITERS` | 5000 | Training steps |

Increasing any of these improves the model but requires more compute. Doubling `N_LAYERS` roughly doubles training time.

---

## API reference

**`POST /generate`**

```json
{
  "prompt": "To be or not to be",
  "max_tokens": 200,
  "temperature": 0.8,
  "top_k": 40
}
```

Response:

```json
{
  "text": "To be or not to be the duke...",
  "prompt": "To be or not to be",
  "tokens_generated": 200
}
```

**`GET /health`** — returns model info (parameters, vocab size, device, block size).

---

## Acknowledgements

Architecture based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Training data: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
