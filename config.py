import torch

# --- Model architecture ---
BLOCK_SIZE = 64      # how many characters the model sees at once (context window)
N_EMBD     = 64      # size of each token's vector representation
N_HEADS    = 4       # number of parallel attention heads (N_EMBD must be divisible by N_HEADS)
N_LAYERS   = 4       # number of stacked transformer blocks
DROPOUT    = 0.1     # fraction of neurons randomly zeroed during training (regularization)

# --- Training ---
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
MAX_ITERS     = 5000
EVAL_INTERVAL = 500   # how often to print loss
EVAL_ITERS    = 50    # how many batches to average for the loss estimate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data ---
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "data/shakespeare.txt"
