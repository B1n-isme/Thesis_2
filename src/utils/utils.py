"""
Utility functions for neural forecasting pipeline.
"""
import logging
import random
import torch
import numpy as np
import pandas as pd
import ray
import yaml
from datetime import datetime
import os
import warnings
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from mlforecast.lag_transforms import ExpandingMean, ExponentiallyWeightedMean, RollingMean
from mlforecast.target_transforms import AutoDifferences, LocalStandardScaler
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse
from utilsforecast.compat import DataFrame
import optuna
import time
import pickle
from config.base import PI_N_WINDOWS_FOR_CONFORMAL, RESULTS_DIR, HORIZON
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast

if TYPE_CHECKING:
    from neuralforecast import NeuralForecast
    from neuralforecast.utils import PredictionIntervals


def get_horizon_directories():
    """Get the appropriate directories based on the HORIZON parameter."""
    from config.base import (
        CV_7D_DIR, CV_14D_DIR, CV_30D_DIR, CV_60D_DIR, CV_90D_DIR,
        FINAL_7D_DIR, FINAL_14D_DIR, FINAL_30D_DIR, FINAL_60D_DIR, FINAL_90D_DIR,
        PLOT_7D_DIR, PLOT_14D_DIR, PLOT_30D_DIR, PLOT_60D_DIR, PLOT_90D_DIR
    )
    
    if HORIZON == 7:
        return CV_7D_DIR, FINAL_7D_DIR, PLOT_7D_DIR
    elif HORIZON == 14:
        return CV_14D_DIR, FINAL_14D_DIR, PLOT_14D_DIR
    elif HORIZON == 30:
        return CV_30D_DIR, FINAL_30D_DIR, PLOT_30D_DIR
    elif HORIZON == 60:
        return CV_60D_DIR, FINAL_60D_DIR, PLOT_60D_DIR
    elif HORIZON == 90:
        return CV_90D_DIR, FINAL_90D_DIR, PLOT_90D_DIR
    else:
        raise ValueError(f"Unsupported HORIZON value: {HORIZON}. Supported values are 7, 14, 30, 60, 90.")


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

def calculate_metrics_1(df: pd.DataFrame) -> Dict:
    """Calculate metrics from cross-validation results."""
    y_true, y_pred = df['y'].values, df['pred'].values
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


# Hit Ratio (Directional Accuracy)
# Compares the direction (up/down) of the point forecast to the actual movement. Only the forecasted and actual values are needed to determine if the direction was predicted correctly.
# Profit/Loss (PnL)
# Simulates trading outcomes using the point forecast as the trading signal. The actual and predicted prices are used to compute hypothetical returns.
# Sharpe Ratio
# Uses the series of returns generated from point forecasts (as trading signals or portfolio weights) to calculate risk-adjusted return. Only point forecasts and actual prices are needed.
# Maximum Drawdown (MDD)
# Evaluates the largest peak-to-trough loss in a simulated portfolio based on point forecast-driven trades or allocations.
# Heteroskedasticity-Adjusted MSE (HMSE)
# This is a variant of MSE that weights errors by realized volatility, but still requires only the point forecast and actual value for each period, plus a volatility estimate.
def calculate_metrics(cv_df: pd.DataFrame, model_names: List[str]) -> Dict[str, Dict]:
    """Calculate metrics from cross-validation results using utilsforecast's evaluate method for multiple models."""
    try:
        # If a single model name is passed as string, convert to list for backward compatibility
        if isinstance(model_names, str):
            model_names = [model_names]
        
        # Clean column names (remove suffixes like '-median' if present)
        cv_df_clean = cv_df.copy()
        cv_df_clean.columns = cv_df_clean.columns.str.replace('-median', '')
        
        # Prepare dataframe for utilsforecast evaluate (drop 'cutoff' if it exists)
        eval_df = cv_df_clean.drop(columns=['cutoff']) if 'cutoff' in cv_df_clean.columns else cv_df_clean
        
        # Use utilsforecast evaluate method
        evaluation_df = evaluate(eval_df, metrics=[mae, rmse, mse])
        
        # Filter results for the specific metrics and aggregate across unique_ids
        model_metrics = evaluation_df[evaluation_df['metric'].isin(['mae', 'rmse', 'mse'])].copy()
        
        results = {}
        
        for model_name in model_names:
            if model_name not in cv_df_clean.columns:
                results[model_name] = {'error': f'Model {model_name} not found in CV results'}
                continue
            
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
            
            # --- New Metrics for Financial Time Series ---
            # Directional Accuracy and Theil's U
            if 'y_lag1' not in cv_df_clean.columns:
                cv_df_clean['y_lag1'] = cv_df_clean.sort_values(by=['unique_id', 'ds']).groupby('unique_id')['y'].shift(1)

            y_lag1 = cv_df_clean['y_lag1'].values
            
            # Create a mask to handle NaNs from shifting and in predictions
            da_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_lag1))
            y_true_da, y_pred_da, y_lag1_da = y_true[da_mask], y_pred[da_mask], y_lag1[da_mask]

            if len(y_true_da) > 0:
                # Directional Accuracy (DA)
                true_direction = np.sign(y_true_da - y_lag1_da)
                pred_direction = np.sign(y_pred_da - y_lag1_da)
                aggregated_metrics['da'] = np.mean(true_direction == pred_direction) * 100

                # Theil's U statistic
                rmse_model = aggregated_metrics.get('rmse')
                if rmse_model is not None and not np.isnan(rmse_model):
                    rmse_naive = np.sqrt(np.mean((y_true_da - y_lag1_da)**2))
                    if rmse_naive > 0:
                        aggregated_metrics['theil_u'] = rmse_model / rmse_naive
                    else:
                        aggregated_metrics['theil_u'] = np.nan
                else:
                    aggregated_metrics['theil_u'] = np.nan
            else:
                aggregated_metrics['da'] = np.nan
                aggregated_metrics['theil_u'] = np.nan

            results[model_name] = {
                'mae': aggregated_metrics['mae'],
                'rmse': aggregated_metrics['rmse'], 
                'mape': aggregated_metrics['mape'],
                'smape': aggregated_metrics['smape'],
                'da': aggregated_metrics.get('da'),
                'theil_u': aggregated_metrics.get('theil_u')
            }
        
        return results
        
    except Exception as e:
        # Return error for all models if calculation fails
        return {model_name: {'error': f'Error calculating metrics: {str(e)}'} for model_name in model_names}

def process_cv_results(consolidated_cv_df: pd.DataFrame, all_cv_metadata: Dict) -> pd.DataFrame:
    """
    Calculates metrics on a consolidated CV dataframe and returns a final results dataframe.
    """
    if consolidated_cv_df is None or consolidated_cv_df.empty:
        # Create a results df from metadata even if no CV was successful
        cv_results_df = pd.DataFrame(list(all_cv_metadata.values()))
        if 'model_name' not in cv_results_df.columns and all_cv_metadata:
            cv_results_df['model_name'] = list(all_cv_metadata.keys())
        return cv_results_df

    print("Processing consolidated CV results for metric calculation...")
    
    successful_models = [name for name, meta in all_cv_metadata.items() if meta['status'] == 'success']
    if successful_models:
        metrics_results = calculate_metrics(consolidated_cv_df, successful_models)
        for model_name, metrics in metrics_results.items():
            if model_name in all_cv_metadata:
                if 'error' in metrics:
                    all_cv_metadata[model_name]['status'] = 'metrics_error'
                    all_cv_metadata[model_name]['error'] = metrics['error']
                else:
                    all_cv_metadata[model_name].update(metrics)
    
    cv_results = []
    for model_name, meta in all_cv_metadata.items():
        result_entry = {
            'model_name': model_name,
            'training_time': meta.get('training_time', 0),
            'status': meta.get('status', 'unknown'),
            'error': meta.get('error', np.nan),
            'mae': meta.get('mae', np.nan), 'rmse': meta.get('rmse', np.nan),
            'mape': meta.get('mape', np.nan), 'smape': meta.get('smape', np.nan),
            'da': meta.get('da', np.nan), 'theil_u': meta.get('theil_u', np.nan),
        }
        cv_results.append(result_entry)
    
    return pd.DataFrame(cv_results)

def save_best_configurations(fitted_objects: Dict, save_dir: str = None) -> Dict[str, List]:
    """
    Extract and save best configurations from fitted auto models using built-in methods.
    Saves all configs to a single YAML file for easy comparison.
    
    Returns:
        Dictionary containing best configurations for each framework
    """
    if save_dir is None:
        from config.base import RESULTS_DIR
        save_dir = RESULTS_DIR / "auto_model_configs"
    
    # Ensure save directory exists
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    all_best_configs = {}
    yaml_configs = {}  # For YAML output structure
    
    def clean_config_value(value):
        """Clean config values for YAML serialization."""
        if hasattr(value, '__name__'):  # For loss functions etc.
            return value.__name__
        elif hasattr(value, 'item'):  # For numpy types
            return value.item()
        elif isinstance(value, (list, tuple)):
            return [clean_config_value(v) for v in value]
        else:
            return value
    
    # Extract neural model configs using built-in method
    if 'neural' in fitted_objects and fitted_objects['neural'] is not None:
        nf_auto = fitted_objects['neural']
        print(f"\nExtracting best configs from {len(nf_auto.models)} neural auto models...")
        
        neural_configs = []
        for i, model in enumerate(nf_auto.models):
            try:
                # Use built-in method to get best config
                best_config = model.results.get_best_result().config
                model_name = model.__class__.__name__
                
                # Clean config for YAML
                clean_config = {}
                for key, value in best_config.items():
                    # Skip non-hyperparameter keys
                    if key not in ['h', 'loss', 'valid_loss']:
                        clean_config[key] = clean_config_value(value)
                
                neural_configs.append({'model_name': model_name, 'config': clean_config})
                yaml_configs[model_name] = clean_config
                print(f"  ✓ Extracted config for {model_name}")
                
            except Exception as e:
                print(f"  ✗ Error extracting config for model {i}: {e}")
        
        all_best_configs['neural'] = neural_configs
        print(f"  ✓ Extracted {len(neural_configs)} neural model configurations")
    
    # Extract ML model configs using built-in method
    if 'ml' in fitted_objects and fitted_objects['ml'] is not None:
        ml_auto = fitted_objects['ml']
        print(f"\nExtracting best configs from ML auto models...")
        
        ml_configs = []
        if hasattr(ml_auto, 'results_') and ml_auto.results_:
            for model_name, results in ml_auto.results_.items():
                try:
                    # Use built-in method to get best config
                    best_config = results.best_trial.user_attrs['config']
                    
                    # Keep nested structure for YAML - don't flatten
                    clean_config = {}
                    for section, params in best_config.items():
                        if isinstance(params, dict):
                            clean_config[section] = {}
                            for key, value in params.items():
                                clean_config[section][key] = clean_config_value(value)
                        else:
                            clean_config[section] = clean_config_value(params)
                    
                    ml_configs.append({'model_name': model_name, 'config': clean_config})
                    yaml_configs[model_name] = clean_config
                    print(f"  ✓ Extracted config for {model_name}")
                    
                except Exception as e:
                    print(f"  ✗ Error extracting config for {model_name}: {e}")
        
        all_best_configs['ml'] = ml_configs
        print(f"  ✓ Extracted {len(ml_configs)} ML model configurations")
    
    # Statistical models don't have hyperparameters to save
    if 'stat' in fitted_objects and fitted_objects['stat'] is not None:
        print("\nStatistical models don't have hyperparameters to save")
        all_best_configs['stat'] = []
    
    # Save all configs to a single YAML file with timestamp
    if yaml_configs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        yaml_path = save_dir / f"best_configurations_comparison_{timestamp}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_configs, f, default_flow_style=False, indent=2, sort_keys=False)
        print(f"\n  ✓ All best configurations saved to {yaml_path}")
        print(f"  ✓ YAML contains {len(yaml_configs)} model configurations")
    else:
        print("\n  ⚠ No configurations to save")
    
    return all_best_configs

def extract_model_names_from_columns(columns: List[str]) -> List[str]:
    """
    Extract model names from dataframe columns, handling both Auto and normal models.
    Handles prediction interval suffixes like '-lo-90', '-hi-95', '-median', etc.
    
    Args:
        columns: List of column names from dataframe
        
    Returns:
        List of unique model names (Auto or normal)
    """
    model_names = set()
    ignore_cols = {'unique_id', 'ds', 'y', 'cutoff'}
    
    for col in columns:
        if col in ignore_cols:
            continue
            
        # Extract base model name by removing suffixes
        if '-lo-' in col:
            base_name = col.split('-lo-')[0]
        elif '-hi-' in col:
            base_name = col.split('-hi-')[0]
        elif '-median' in col:
            base_name = col.split('-median')[0]
        else:
            base_name = col
            
        # Only add if it's either Auto or non-ignored normal model
        if base_name.startswith('Auto') or (not base_name.startswith('Auto') and base_name not in ignore_cols):
            model_names.add(base_name)
    
    return sorted(list(model_names))

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)

def save_dict_to_json(data_dict: Dict, file_path: Path):
    """
    Saves a dictionary to a JSON file, with special handling for numpy types.

    Args:
        data_dict (Dict): The dictionary to save.
        file_path (Path): The path to the output JSON file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, cls=NumpyEncoder, indent=4)

def load_json_to_dict(file_path: Path) -> Dict:
    """
    Loads a JSON file into a dictionary.

    Args:
        file_path (Path): The path to the input JSON file.

    Returns:
        Dict: The loaded dictionary.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def load_yaml_to_dict(file_path: Path) -> Dict[str, Any]:
    """
    Loads a YAML file into a dictionary.

    Args:
        file_path (Path): The path to the input YAML file.

    Returns:
        Dict: The loaded dictionary.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def rolling_forecast_neural_all_models(
    nf_model: "NeuralForecast",
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_length: int,
    predict_level: List[int],
    refit_frequency: int = 1  # Refit every N windows (1 = every window, 3 = every 3 windows)
) -> pd.DataFrame:
    """
    More efficient rolling forecast with configurable refit frequency.
    
    Args:
        refit_frequency: How often to refit models (1 = every window, 2 = every 2 windows, etc.)
                        Set to 0 to disable refitting (fastest but less adaptive)
    """
    print(f"      Performing efficient rolling forecast (refit every {refit_frequency} windows)...")
    all_forecasts_list = []
    current_train_df = train_df.copy()

    # Prepare column names for future exogenous features
    exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
    futr_cols = ['unique_id', 'ds'] + exog_cols

    test_duration_steps = len(test_df)
    if horizon_length <= 0:
        raise ValueError("horizon_length must be positive for rolling forecast.")

    n_windows = (test_duration_steps + horizon_length - 1) // horizon_length
    print(f"        Rolling windows: {n_windows}, Horizon per window: {horizon_length}")

    for window_idx in range(n_windows):
        print(f"        Processing rolling window {window_idx + 1}/{n_windows}...")
        
        start_idx = window_idx * horizon_length
        end_idx = min(start_idx + horizon_length, test_duration_steps)
        current_test_window_actuals_df = test_df.iloc[start_idx:end_idx].copy()

        if current_test_window_actuals_df.empty:
            print(f"        Window {window_idx + 1} data is empty, skipping.")
            continue
            
        # Prepare futr_df for prediction
        futr_df_for_predict = current_test_window_actuals_df[futr_cols].copy()
        if futr_df_for_predict.empty:
            if not exog_cols and not current_test_window_actuals_df[['unique_id', 'ds']].empty:
                futr_df_for_predict = current_test_window_actuals_df[['unique_id', 'ds']].copy()
            else:
                print(f"        futr_df for prediction is empty for window {window_idx + 1}. Skipping.")
                continue

        print(f"        Predicting for window {window_idx + 1} (futr_df rows: {len(futr_df_for_predict)}).")
        window_forecast_df = nf_model.predict(futr_df=futr_df_for_predict, level=predict_level)
        all_forecasts_list.append(window_forecast_df)
        
        # Update training data with ACTUAL VALUES (not predictions) and conditionally refit
        if window_idx < n_windows - 1:
            if 'y' not in current_test_window_actuals_df.columns:
                print(f"        Warning: Column 'y' not in test data for window {window_idx + 1}.")
            
            # Always append actual values to training data
            actuals_to_append = current_test_window_actuals_df
            print(f"        Appending {len(actuals_to_append)} actual observations to training data.")
            current_train_df = pd.concat([current_train_df, actuals_to_append], ignore_index=True)
            
            # Conditional refit based on frequency
            should_refit = (refit_frequency > 0 and (window_idx + 1) % refit_frequency == 0)
            
            if should_refit:
                print(f"        Re-fitting models (window {window_idx + 1}, train size: {len(current_train_df)})...")
                fit_val_size = horizon_length if horizon_length > 0 and len(current_train_df) > horizon_length * 2 else None
                nf_model.fit(
                    current_train_df, 
                    val_size=fit_val_size, 
                    verbose=False, 
                    prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
                )
                print(f"        Models re-fitted.")
            else:
                print(f"        Skipping refit (will refit at window {((window_idx // refit_frequency) + 1) * refit_frequency}).")
            
    if not all_forecasts_list:
        print("        Warning: No forecasts were generated during the rolling forecast process.")
        return pd.DataFrame()

    final_concatenated_forecasts_df = pd.concat(all_forecasts_list, ignore_index=True)
    return final_concatenated_forecasts_df

def my_init_config(trial: optuna.Trial):
    lag_transforms = [
        ExponentiallyWeightedMean(alpha=0.3),
        RollingMean(window_size=7, min_samples=1),  # 7 days instead of 24*7 hours
    ]
    lag_to_transform = trial.suggest_categorical('lag_to_transform', [1, 2])  # in days
    return {
        'lags': [i for i in range(1, 7)],  # 1 to 6 days
        'lag_transforms': {lag_to_transform: lag_transforms},
    }


def my_fit_config(trial: optuna.Trial):
    if trial.suggest_int('use_id', 0, 1):
        static_features = ['unique_id']
    else:
        static_features = None
    return {
        'static_features': static_features
    }

def custom_mae_loss(df: DataFrame, train_df: DataFrame) -> float:
    """
    Calculates Mean Absolute Error.
    'df' contains predictions in a column named 'model' and actuals in 'target_col'.
    'train_df' is the training data for the current window (can be ignored for simple MAE).
    """
    actuals = df['y'].to_numpy()
    predictions = df['model'].to_numpy()
    return np.mean(np.abs(actuals - predictions))
