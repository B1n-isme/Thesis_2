"""
Module for model evaluation (cross-validation and metrics) in the Auto Models workflow.
"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict

# Third-party import
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

# Local import
from config.base import *
from src.utils.utils import calculate_metrics

def perform_cross_validation(auto_models: List, stat_models: List, train_df: pd.DataFrame) -> pd.DataFrame:
    """Perform cross-validation for all models."""
    cv_results = []
    
    # Cross-validate Auto Neural Models
    if auto_models:
        print(f"Cross-validating {len(auto_models)} Auto Neural models...")
        for i, model in enumerate(auto_models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(auto_models)}] CV {model_name}...")
            start_time = time.time()
            try:
                nf = NeuralForecast(models=[model], freq=FREQUENCY, local_scaler_type=LOCAL_SCALER_TYPE)
                cv_df = nf.cross_validation(
                    df=train_df,
                    n_windows=CV_N_WINDOWS,
                    step_size=CV_STEP_SIZE,
                    val_size=HORIZON,
                    refit=True,
                    verbose=False,
                    prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL),
                    level= LEVELS
                )
                if cv_df.empty:
                    raise ValueError("Cross-validation returned empty results")
                metrics = calculate_metrics(cv_df, model_name)
                training_time = time.time() - start_time
                cv_results.append({
                    'model_name': model_name, 'framework': 'neuralforecast',
                    'training_time': training_time, 'evaluation_method': 'cross_validation',
                    'is_auto': True, 'status': 'success', **metrics
                })
                print(f"  ✓ {model_name} CV completed (MAE: {metrics.get('mae', 'N/A'):.4f})")
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in CV for {model_name}: {error_msg}")
                cv_results.append({
                    'model_name': model_name, 'framework': 'neuralforecast',
                    'training_time': training_time, 'error': error_msg,
                    'status': 'failed', 'is_auto': True,
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })
                
    # Cross-validate Auto Statistical Models
    if stat_models:
        print(f"Cross-validating {len(stat_models)} statistical models...")
        df_stat = train_df[['unique_id', 'ds', 'y']].copy()
        for i, model in enumerate(stat_models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(stat_models)}] CV {model_name}...")
            start_time = time.time()
            try:
                sf = StatsForecast(models=[model], freq=FREQUENCY, verbose=True)
                cv_df = sf.cross_validation(
                    df=df_stat, 
                    h=HORIZON, 
                    n_windows=CV_N_WINDOWS, 
                    step_size=CV_STEP_SIZE,
                    refit=True,
                    prediction_intervals=ConformalIntervals(h=HORIZON, n_windows=PI_N_WINDOWS_FOR_CONFORMAL), 
                    level=LEVELS
                )
                if cv_df.empty:
                    raise ValueError("Cross-validation returned empty results")
                metrics = calculate_metrics(cv_df, model_name)
                training_time = time.time() - start_time
                cv_results.append({
                    'model_name': model_name, 'framework': 'statsforecast',
                    'training_time': training_time, 'evaluation_method': 'cross_validation',
                    'is_auto': False, 'status': 'success', **metrics
                })
                print(f"  ✓ {model_name} CV completed (MAE: {metrics.get('mae', 'N/A'):.4f})")
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in CV for {model_name}: {error_msg}")
                cv_results.append({
                    'model_name': model_name, 'framework': 'statsforecast',
                    'training_time': training_time, 'error': error_msg,
                    'status': 'failed', 'is_auto': False,
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })
                
    cv_results_df = pd.DataFrame(cv_results)
    successful_cv = cv_results_df[cv_results_df['status'] == 'success']
    print(f"Cross-validation completed: {len(successful_cv)}/{len(cv_results_df)} models successful")
    return cv_results_df
