"""Default configurations for ST-GCN traffic density forecasting."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "delhitraffic"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Dataset Specs
NUM_CAMERAS = 6
FEATURES = ["QueueDensity", "StopDensity"]
NUM_FEATURES = len(FEATURES)

# Temporal & Forecasting Specs
RESAMPLE_FREQ = "1min"
INPUT_WINDOW = 60          # 60 timesteps (60 minutes past history)
HORIZONS = [5, 15, 30]     # Prediction targets at t+5, t+15, t+30 minutes
MAX_HORIZON = max(HORIZONS)

# Dataset Split (Chronological by recorded days)
TRAIN_RATIO = 0.70         # ~28 days
VAL_RATIO = 0.15           # ~6 days
TEST_RATIO = 0.15          # ~6 days

# Graph Construction Specs
CORR_THRESHOLD = 0.6       # Pearson correlation threshold for graph edge connection
CHEB_K = 3                 # Chebyshev polynomial order (per paper: K=3)

# Model Architecture Specs (ST-GCN)
TEMPORAL_KERNEL_KT = 3     # 1D Conv kernel size along time dimension
ST_BLOCK_CHANNELS = [
    [2, 32, 64],           # Block 1: Cin=2, Cmid=32, Cout=64
    [64, 32, 128],         # Block 2: Cin=64, Cmid=32, Cout=128
]
DROPOUT = 0.2

# Training Specs
BATCH_SIZE = 32            # Optimal for RTX 4050 / CPU stability
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
SEED = 42
