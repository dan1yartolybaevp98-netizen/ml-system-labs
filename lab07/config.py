from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

BATCH_SIZE = 128
BASELINE_EPOCHS = 5
PRUNING_FINE_TUNE_EPOCHS = 3
STUDENT_EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
ALPHA = 0.5
RANDOM_SEED = 42
NUM_WORKERS = 0  # reliable on macOS; increase only if needed

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
