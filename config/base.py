"""
Configuration settings.
"""
import os
import torch
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MQLoss, DistributionLoss

# === Data Configuration ===
DATA_PATH = 'data/final/dataset.parquet'
DATE_COLUMN = 'Date'
DATE_RENAMED = 'ds'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
UNIQUE_ID_VALUE = 'Bitcoin'

# === Forecasting Configuration ===
HORIZON = 7
LEVELS = [80, 90]
TEST_LENGTH_MULTIPLIER = 5
SEED = 42

# === Rolling Forecast Configuration ===
ENABLE_ROLLING_FORECAST = True  # Enable rolling forecast for neural models when horizon < test_length
ROLLING_REFIT_FREQUENCY = 0  # Refit every N windows (1=every window, 3=every 3 windows, 0=no refit)

# === Model Configuration ===
FREQUENCY = 'D'
SCALER_TYPE = ['standard']  # List for tune.choice()
LOCAL_SCALER_TYPE = 'standard'  # String for direct use

# === Cross-validation Configuration ===
CV_N_WINDOWS = 10
CV_STEP_SIZE = HORIZON
PI_N_WINDOWS_FOR_CONFORMAL = 20 

# === Hyperparameter Tuning Configuration ===
NUM_SAMPLES_PER_MODEL = 1


# === Search Algorithm Configuration ===
DEFAULT_SEARCH_ALGORITHM = 'optuna'  # Default search algorithm
SEARCH_ALGORITHM_MAX_CONCURRENT = 4  # Max concurrent trials
SEARCH_ALGORITHM_REPEAT_TRIALS = None  # Number of repeated evaluations (None = no repeat)
FAST_SEARCH_ALGORITHM = 'hyperopt'  # Algorithm to use in fast mode

# === Ray Configuration ===
RAY_ADDRESS = 'local'
RAY_NUM_CPUS = os.cpu_count()
RAY_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# === Loss function mapping ===
LOSS_MAP = {
    'MAE': MAE(),
    'MAE()': MAE(),
    'MQLoss': MQLoss(),
    'MQLoss()': MQLoss(),
    'RMSE': RMSE(),
    'RMSE()': RMSE(),
    'MSE': MSE(),
    'MSE()': MSE()
}

# === Columns to exclude when processing best hyperparameters ===
EXCLUDE_HYPERPARAMETER_KEYS = [
    'model_name', 
    'loss', 
    'valid_loss', 
    'best_valid_loss', 
    'training_iteration'
]

# === JSON parseable hyperparameter keys ===
JSON_PARSEABLE_KEYS = [
    'n_pool_kernel_size', 
    'n_freq_downsample', 
    'n_blocks', 
    'mlp_units'
]

# === Output Directories ===
RESULTS_DIR: str = 'results'
HPO_DIR: str = f"{RESULTS_DIR}/hpo"
PLOT_DIR: str = f"{RESULTS_DIR}/plot"
FINAL_DIR: str = f"{RESULTS_DIR}/final"
MODELS_DIR: str = f"{RESULTS_DIR}/models"
BEST_HYPERPARAMETERS_CSV = f"{RESULTS_DIR}/best_hyperparameters.json"

def __post_init__(self):
        """Set default values that depend on other attributes."""
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.hpo_dir, exist_ok=True)
        os.makedirs(self.PLOT_DIR, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

