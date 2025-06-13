"""
Configuration settings.
"""
import os
from pathlib import Path
import torch
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.optuna import OptunaSearch
from typing import Any

# === Data Configuration ===
# DATA_PATH = 'data/final/dataset.parquet'
DATA_PATH = 'data/final/final_feature_selected_data.parquet'
DATE_COLUMN = 'Date'
DATE_RENAMED = 'ds'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
UNIQUE_ID_VALUE = 'Bitcoin'

# === Forecasting Configuration ===
HORIZON = 7
LEVELS = [95]
TEST_LENGTH_MULTIPLIER = 1
SEED = 42

# === Rolling Forecast Configuration ===
ENABLE_ROLLING_FORECAST = True  # Enable rolling forecast for neural models when horizon < test_length
ROLLING_REFIT_FREQUENCY = 0  # Refit every N windows (1=every window, 3=every 3 windows, 0=no refit)


# === Cross-validation Configuration ===
CV_N_WINDOWS = 49
# CV_STEP_SIZE = 30
CV_STEP_SIZE = HORIZON
INPUT_SIZE = 1095

# === Conformal Prediction Configuration ===
PI_N_WINDOWS_FOR_CONFORMAL = 20 

# === Hyperparameter Tuning Configuration ===
NUM_SAMPLES_PER_MODEL = 10


# === Ray Configuration ===
RAY_ADDRESS = 'local'
RAY_NUM_CPUS = os.cpu_count()
RAY_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


# === Output Directories ===
RESULTS_DIR = Path('results')
CV_DIR = RESULTS_DIR / 'cv'
FINAL_DIR = RESULTS_DIR / 'final'
PLOT_DIR = FINAL_DIR / 'plot'

MODELS_DIR = RESULTS_DIR / 'models'
BEST_HYPERPARAMETERS_CSV = RESULTS_DIR / 'best_hyperparameters.json'

def __post_init__(self):
        """Set default values that depend on other attributes."""
        # Create output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.hpo_dir.mkdir(parents=True, exist_ok=True)
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# For initial baseline establishment (as recommended in algo.txt)
BASELINE = ['random', 'hyperopt']

# For efficient optimization (BOHB and Optuna with TPE)
EFFICIENT = ['bohb', 'optuna']

# For noisy time series data
NOISY_DATA = ['hebo', 'random']

# BOHB (Bayesian Optimization HyperBand): 
# - bayesian optimization (efficiency) + hyperband (resource adaptiveness)
# -> require careful evaluate config

# HEBO (Heteroscedastic Evolutionary Bayesian Optimization)
# - suit with noisy data
# -> hard to config 

# OptunaSearch 
# - TPE (Tree-structured Parzen Estimator): a robust Bayesian optimization technique
# - CMA-ES (Covariance Matrix Adaptation Evolution Strategy): excellent for more complex, non-convex, or ill-conditioned search spaces.
# -> start with TPE

# Suggestion:
# 1. Start with Random Search or HyperOptSearch: Run a small number of trials (e.g., 20-30) to get a feel for the search space and establish a baseline.
# 2. Move to BOHB or Optuna (with TPE/HyperBand Pruner): These are often my first choices for serious tuning due to their balance of performance and efficiency. HEBO is a strong alternative if you anticipate a noisy or complex landscape.

def get_search_algorithm_class(algorithm: str) -> Any:
    if algorithm == 'random':
        return BasicVariantGenerator()
        
    elif algorithm == 'hyperopt':
        return HyperOptSearch()
        
    elif algorithm == 'bohb':
        return TuneBOHB()
        
    elif algorithm == 'hebo':
        return HEBOSearch()
        
    elif algorithm == 'optuna':
        return OptunaSearch()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
