class CharTokenizer:
    """
    Maps every unique character in a text to an integer ID and back.

    This is the simplest possible tokenizer. Real LLMs (GPT-4, Claude) use
    Byte-Pair Encoding (BPE) which merges common character sequences into
    single tokens, giving a vocabulary of ~50k instead of ~65. The math
    inside the model is identical either way.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))           # every unique character, sorted for reproducibility
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string  → int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int     → string

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        return ''.join(self.itos[i] for i in tokens)
