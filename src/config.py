import os
from pathlib import Path
import torch

# ==========================================
# BASE DIRECTORIES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"

# Automatically ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
DATASET_PATH = DATASET_DIR / "IMDB Dataset.csv"
VOCAB_PATH = DATASET_DIR / "vocab.pkl"
MODEL_PATH = MODELS_DIR / "rnn_model.pth"
BEST_MODEL_PATH = MODELS_DIR / "best_rnn_model.pth"

TRAIN_DATA_PATH = DATASET_DIR / "train.csv"
VAL_DATA_PATH = DATASET_DIR / "validation.csv"
TEST_DATA_PATH = DATASET_DIR / "test.csv"

# ==========================================
# DEVICE SETUP (CUDA / CPU)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_info():
    """Returns a string describing the current active device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        return f"CUDA GPU: {gpu_name} (Total GPUs: {gpu_count}, CUDA: {cuda_version})"
    return "CPU (No CUDA GPU detected)"

# ==========================================
# DATA SPLIT PARAMETERS
# ==========================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# ==========================================
# VOCABULARY & TOKENIZATION
# ==========================================
MAX_VOCAB_SIZE = 25000
MAX_SEQUENCE_LENGTH = 200

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# ==========================================
# MODEL ARCHITECTURE (HIGH ACCURACY CONFIG)
# ==========================================
# 'lstm' provides gating to eliminate vanishing gradients and boost accuracy to 85%+
RNN_TYPE = "lstm"           # Options: "lstm", "gru", "rnn"
BIDIRECTIONAL = True        # Captures forward and backward context
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4               # Regularization to prevent overfitting

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# On Windows, num_workers=0 avoids DataLoader multiprocessing errors
NUM_WORKERS = 0