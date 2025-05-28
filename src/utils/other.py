"""
Utility functions for neural forecasting pipeline.
"""
import logging
import random
import torch
import numpy as np
import pandas as pd
import ray
from datetime import datetime
import os
import warnings


def seed_everything(seed=42):
    """Set seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment(seed=42, ray_config=None):
    """Setup the environment for neural forecasting."""
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Setup logging
    warnings.filterwarnings('ignore')
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)
    torch.set_float32_matmul_precision('high')

    # Initialize Ray
    if ray_config is None:
        ray_config = {
            'address': 'local',
            'log_to_driver': False,
            'logging_level': logging.ERROR,
            'num_cpus': os.cpu_count(),
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    ray.init(**ray_config)
    
    print(f"Pipeline execution started at: {pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').strftime('%Y-%m-%d %H:%M:%S')} (Ho Chi Minh City Time)")


def print_data_info(df, train_df, test_df):
    """Print information about data splits."""
    print(f"\nTotal data shape: {df.shape}")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"  Train set covers: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"  Test set covers: {test_df['ds'].min()} to {test_df['ds'].max()}")


def get_historical_exogenous_features(df, exclude_cols=None):
    """Get list of historical exogenous features from dataframe."""
    if exclude_cols is None:
        exclude_cols = ['ds', 'unique_id', 'y']
    
    all_cols = df.columns.tolist()
    hist_exog_list = [col for col in all_cols if col not in exclude_cols]
    return hist_exog_list


def calculate_evaluation_metrics(df_pl, model_cols, exclude_cols=None):
    """Calculate evaluation metrics for cross-validation results."""
    from utilsforecast.losses import mse, mae
    
    if exclude_cols is None:
        exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
    
    results = {}
    
    for model in model_cols:
        if model not in exclude_cols:
            # Calculate MSE and MAE using utilsforecast
            mse_val = mse(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
            mae_val = mae(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
            rmse_val = np.sqrt(mse_val)
            
            results[model] = {
                'MSE': mse_val,
                'MAE': mae_val,
                'RMSE': rmse_val
            }
            
            print(f"\nEvaluation for model: {model}")
            print(f"  MSE : {mse_val:.4f}")
            print(f"  MAE : {mae_val:.4f}")
            print(f"  RMSE: {rmse_val:.4f}")
    
    return results 