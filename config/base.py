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

# === Forecasting Configuration ===
HORIZON = 60
LEVELS = [95]
TEST_LENGTH_MULTIPLIER = 1
SEED = 42

# === Data Configuration ===
RAW_DATA_PATH = 'data/final/raw_dataset.parquet'
DATA_PATH = f'data/final/feature_selection_{HORIZON}_mc.parquet'
DATE_COLUMN = 'Date'
DATE_RENAMED = 'ds'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
UNIQUE_ID_VALUE = 'Bitcoin'

# === Rolling Forecast Configuration ===
ENABLE_ROLLING_FORECAST = True  # Enable rolling forecast for neural models when horizon < test_length
ROLLING_REFIT_FREQUENCY = 0  # Refit every N windows (1=every window, 3=every 3 windows, 0=no refit)


# === Cross-validation Configuration ===
CV_N_WINDOWS = 30
CV_STEP_SIZE = HORIZON
INPUT_SIZE = 548

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
RESULTS_7D_DIR = RESULTS_DIR / 'results_7d'
RESULTS_14D_DIR = RESULTS_DIR / 'results_14d' 
RESULTS_30D_DIR = RESULTS_DIR / 'results_30d'
RESULTS_60D_DIR = RESULTS_DIR / 'results_60d'
RESULTS_90D_DIR = RESULTS_DIR / 'results_90d'

CV_7D_DIR = RESULTS_7D_DIR / 'cv'
CV_14D_DIR = RESULTS_14D_DIR / 'cv'
CV_30D_DIR = RESULTS_30D_DIR / 'cv'
CV_60D_DIR = RESULTS_60D_DIR / 'cv'
CV_90D_DIR = RESULTS_90D_DIR / 'cv'

FINAL_7D_DIR = RESULTS_7D_DIR / 'final'
FINAL_14D_DIR = RESULTS_14D_DIR / 'final'
FINAL_30D_DIR = RESULTS_30D_DIR / 'final'
FINAL_60D_DIR = RESULTS_60D_DIR / 'final'
FINAL_90D_DIR = RESULTS_90D_DIR / 'final'

PLOT_7D_DIR = FINAL_7D_DIR / 'plots'
PLOT_14D_DIR = FINAL_14D_DIR / 'plots'
PLOT_30D_DIR = FINAL_30D_DIR / 'plots'
PLOT_60D_DIR = FINAL_60D_DIR / 'plots'
PLOT_90D_DIR = FINAL_90D_DIR / 'plots'

for path in [CV_7D_DIR, CV_14D_DIR, CV_30D_DIR, CV_60D_DIR, CV_90D_DIR, 
             FINAL_7D_DIR, FINAL_14D_DIR, FINAL_30D_DIR, FINAL_60D_DIR, FINAL_90D_DIR,
             PLOT_7D_DIR, PLOT_14D_DIR, PLOT_30D_DIR, PLOT_60D_DIR, PLOT_90D_DIR]:
    path.mkdir(parents=True, exist_ok=True)

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
