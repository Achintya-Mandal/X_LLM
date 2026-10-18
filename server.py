"""
Local API server for the GPT model.

Endpoints:
  GET  /             → HTML UI
  GET  /health       → JSON model info
  POST /generate     → generate text from a prompt

Interactive docs (auto-generated):
  http://localhost:8000/docs
"""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BLOCK_SIZE, N_EMBD, N_HEADS, N_LAYERS, DEVICE
from model import GPT


# ---------------------------------------------------------------------------
# State — model is loaded once when the server starts
# ---------------------------------------------------------------------------

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load tokenizer + model into memory
    try:
        with open("tokenizer.pkl", "rb") as f:
            state["tokenizer"] = pickle.load(f)

        model = GPT(
            vocab_size  = state["tokenizer"].vocab_size,
            n_embd      = N_EMBD,
            n_heads     = N_HEADS,
            n_layers    = N_LAYERS,
            block_size  = BLOCK_SIZE,
            dropout     = 0.0,
        ).to(DEVICE)

        model_state = torch.load("model.pt", map_location=DEVICE, weights_only=True)
        model.load_state_dict(model_state)
        model.eval()

        state["model"] = model
        print(f"Model loaded — {model.num_parameters():,} parameters on {DEVICE}")
    except FileNotFoundError as e:
        raise RuntimeError(
            "model.pt or tokenizer.pkl not found. Run `python train.py` first."
        ) from e

    yield  # server runs here

    # Shutdown: nothing to clean up
    state.clear()


app = FastAPI(
    title="Nano GPT",
    description="A tiny character-level GPT trained on Shakespeare.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:      str   = Field(default="",   description="Seed text. Leave empty for unconditional generation.")
    max_tokens:  int   = Field(default=200,  ge=1,   le=2000, description="Number of characters to generate.")
    temperature: float = Field(default=0.8,  gt=0.0, le=2.0,  description="Lower = focused, higher = creative.")
    top_k:       int   = Field(default=40,   ge=1,   le=200,  description="Sample only from the top-k most likely next tokens.")


class GenerateResponse(BaseModel):
    text:        str
    prompt:      str
    tokens_generated: int


class HealthResponse(BaseModel):
    status:     str
    parameters: int
    vocab_size: int
    device:     str
    block_size: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health():
    model     = state["model"]
    tokenizer = state["tokenizer"]
    return HealthResponse(
        status     = "ok",
        parameters = model.num_parameters(),
        vocab_size = tokenizer.vocab_size,
        device     = DEVICE,
        block_size = BLOCK_SIZE,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model     = state["model"]
    tokenizer = state["tokenizer"]

    # Validate that every character in the prompt is in the vocabulary
    unknown = [c for c in req.prompt if c not in tokenizer.stoi]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt contains characters not in vocabulary: {list(set(unknown))}",
        )

    if req.prompt:
        idx = torch.tensor(tokenizer.encode(req.prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens = req.max_tokens,
            temperature    = req.temperature,
            top_k          = req.top_k,
        )

    full_text = tokenizer.decode(output[0].tolist())

    return GenerateResponse(
        text             = full_text,
        prompt           = req.prompt,
        tokens_generated = req.max_tokens,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
