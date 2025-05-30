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
from typing import Dict
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse


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

def calculate_metrics_old(df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate metrics from cross-validation results."""
    try:
        if model_name not in df.columns:
            return {'error': f'Model {model_name} not found in CV results'}
        y_true, y_pred = df['y'].values, df[model_name].values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean, y_pred_clean = y_true[mask], y_pred[mask]
        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions for metric calculation'}
        return {
            'mae': np.mean(np.abs(y_true_clean - y_pred_clean)),
            'rmse': np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2)),
            'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100,
            'smape': np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
        }
    except Exception as e:
        return {'error': f'Error calculating metrics: {str(e)}'}

def calculate_metrics(cv_df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate metrics from cross-validation results using utilsforecast's evaluate method."""
    try:
        if model_name not in cv_df.columns:
            return {'error': f'Model {model_name} not found in CV results'}
        
        # Clean column names (remove suffixes like '-median' if present)
        cv_df_clean = cv_df.copy()
        cv_df_clean.columns = cv_df_clean.columns.str.replace('-median', '')
        
        # Prepare dataframe for utilsforecast evaluate (drop 'cutoff' if it exists)
        eval_df = cv_df_clean.drop(columns=['cutoff']) if 'cutoff' in cv_df_clean.columns else cv_df_clean
        
        # Use utilsforecast evaluate method
        evaluation_df = evaluate(eval_df, metrics=[mae, rmse, mse])
        
        # Filter results for the specific model and aggregate across unique_ids
        model_metrics = evaluation_df[evaluation_df['metric'].isin(['mae', 'rmse', 'mse'])].copy()
        
        # Aggregate metrics across all unique_ids (mean)
        aggregated_metrics = {}
        for metric_name in ['mae', 'rmse', 'mse']:
            metric_data = model_metrics[model_metrics['metric'] == metric_name]
            if model_name in metric_data.columns:
                aggregated_metrics[metric_name] = metric_data[model_name].mean()
            else:
                aggregated_metrics[metric_name] = np.nan
        
        # Calculate MAPE and SMAPE manually since utilsforecast doesn't have them
        y_true, y_pred = cv_df_clean['y'].values, cv_df_clean[model_name].values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean, y_pred_clean = y_true[mask], y_pred[mask]
        
        if len(y_true_clean) > 0:
            aggregated_metrics['mape'] = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
            aggregated_metrics['smape'] = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
        else:
            aggregated_metrics['mape'] = np.nan
            aggregated_metrics['smape'] = np.nan
        
        return {
            'mae': aggregated_metrics['mae'],
            'rmse': aggregated_metrics['rmse'], 
            'mape': aggregated_metrics['mape'],
            'smape': aggregated_metrics['smape']
        }
        
    except Exception as e:
        return {'error': f'Error calculating metrics: {str(e)}'} 