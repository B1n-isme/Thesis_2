"""
Module for model evaluation (cross-validation and metrics) in the Auto Models workflow.
"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict

# Configuration and local imports
from config.base import FREQUENCY, LOCAL_SCALER_TYPE, CV_N_WINDOWS, CV_STEP_SIZE, HORIZON
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast

# Import utilsforecast for cross-validation evaluation
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, rmse

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
                    verbose=False
                )
                if cv_df.empty:
                    raise ValueError("Cross-validation returned empty results")
                metrics = calculate_cv_metrics_with_utilsforecast(cv_df, model_name)
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
                
    # Cross-validate Statistical Models
    if stat_models:
        print(f"Cross-validating {len(stat_models)} statistical models...")
        df_stat = train_df[['unique_id', 'ds', 'y']].copy()
        for i, model in enumerate(stat_models, 1):
            model_name = model.__class__.__name__
            print(f"[{i}/{len(stat_models)}] CV {model_name}...")
            start_time = time.time()
            try:
                sf = StatsForecast(models=[model], freq=FREQUENCY, verbose=True)
                cv_df = sf.cross_validation(df=df_stat, h=HORIZON, n_windows=CV_N_WINDOWS, step_size=CV_STEP_SIZE)
                if cv_df.empty:
                    raise ValueError("Cross-validation returned empty results")
                metrics = calculate_cv_metrics_with_utilsforecast(cv_df, model_name)
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

def calculate_cv_metrics(cv_df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate metrics from cross-validation results."""
    try:
        if model_name not in cv_df.columns:
            return {'error': f'Model {model_name} not found in CV results'}
        y_true, y_pred = cv_df['y'].values, cv_df[model_name].values
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

def calculate_test_metrics(eval_df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate metrics from test evaluation results."""
    try:
        if model_name not in eval_df.columns:
            return {'error': f'Model {model_name} not found in evaluation results'}
        y_true, y_pred = eval_df['y'].values, eval_df[model_name].values
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

def calculate_cv_metrics_with_utilsforecast(cv_df: pd.DataFrame, model_name: str) -> Dict:
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