import pandas as pd
import numpy as np
import time
import json # For save_best_auto_model_config
from pathlib import Path
from typing import List, Dict, Tuple # Assuming these are already imported

# Assume these are defined elsewhere or passed as arguments:
# train_df, test_df, neural_models, stats_models, all_nf_forecasts_df, stat_forecasts_df
# auto_model_configs_dir, all_forecasts_dict, final_results
# use_rolling (boolean indicating if rolling forecast was used for neural models)
# sf (the StatsForecast object, for the isinstance check for AutoARIMA)
# from utilsforecast.losses import mae, rmse, mse # for calculate_metrics
# from utilsforecast.utils import evaluate # for calculate_metrics

# User-provided helper functions (ensure they are defined in your script)
# def save_best_auto_model_config(model_name: str, best_config: Dict, auto_model_configs_dir: Path): ...
# def calculate_metrics(eval_df: pd.DataFrame, model_name: str) -> Dict: ...
# (Make sure to replace 'calculate_test_metrics' with 'calculate_metrics' in the calling code)

# Define LEVELS based on what was used in predict calls, e.g.:
LEVELS = [90] # Example: if predict(level=[90]) was used. Adjust if multiple levels were used.

# --- Part of your perform_final_fit_predict function ---

# === Phase 3: Process results for each NEURAL model ===
print(f"  Processing individual neural model results...")
if all_nf_forecasts_df is None or all_nf_forecasts_df.empty:
    print("    ! No forecasts (all_nf_forecasts_df) available to process for neural models. Skipping Phase 3 for neural models.")
else:
    for i, model_instance in enumerate(neural_models, 1):
        model_name = model_instance.__class__.__name__
        per_model_start_time = time.time()
        print(f"  [{i}/{len(neural_models)}] Processing Neural Model: {model_name}...")
        
        try:
            # Extract this model's forecasts from the collective `all_nf_forecasts_df`
            cols_to_select = ['unique_id', 'ds']
            model_pred_cols = [col for col in all_nf_forecasts_df.columns if col.startswith(model_name)]
            
            # Ensure the main model prediction column is included if not caught by startswith (e.g. exact match)
            if model_name in all_nf_forecasts_df.columns and model_name not in model_pred_cols:
                 model_pred_cols.insert(0, model_name) # Add to start for clarity

            if not model_pred_cols:
                # If still no prediction columns found for the model name.
                print(f"    ! Warning: No prediction columns found for {model_name} in all_nf_forecasts_df. Columns available: {all_nf_forecasts_df.columns.tolist()}")
                # Attempt to add at least the model name if it's an exact match, to avoid erroring out immediately.
                if model_name in all_nf_forecasts_df.columns:
                    cols_to_select.append(model_name)
                else: # Skip this model if no columns found at all.
                    raise ValueError(f"No prediction columns found for {model_name} in all_nf_forecasts_df.")

            cols_to_select.extend(model_pred_cols)
            
            # Ensure unique columns, preserving order
            current_model_forecasts_df = all_nf_forecasts_df[list(dict.fromkeys(cols_to_select))].copy()
            
            evaluation_method = "rolling_forecast" if use_rolling else "direct_forecast"

            # Merge with actuals from test_df for evaluation
            eval_df = test_df.merge(current_model_forecasts_df, on=['unique_id', 'ds'], how='inner')
            if eval_df.empty:
                # Provide more context for debugging if needed
                print(f"      Warning: Merging test_df and current_model_forecasts_df for {model_name} resulted in an empty DataFrame.")
                print(f"      test_df columns: {test_df.columns.tolist()}, current_model_forecasts_df columns: {current_model_forecasts_df.columns.tolist()}")
                print(f"      Common unique_id in test_df: {test_df['unique_id'].unique()[:5]}, in current_model_forecasts_df: {current_model_forecasts_df['unique_id'].unique()[:5] if 'unique_id' in current_model_forecasts_df else 'N/A'}")
                print(f"      Min/Max ds in test_df: {test_df['ds'].min()} / {test_df['ds'].max()}, in current_model_forecasts_df: {current_model_forecasts_df['ds'].min() if 'ds' in current_model_forecasts_df else 'N/A'} / {current_model_forecasts_df['ds'].max() if 'ds' in current_model_forecasts_df else 'N/A'}")
                raise ValueError(f"No matching forecasts and actual values for evaluation for model {model_name}. Merged df is empty.")
            
            # Extract predictions (mean, lo, hi) for all_forecasts_dict
            mean_pred_series = eval_df[model_name] if model_name in eval_df else None # Get from eval_df after merge
            
            lo_preds_dict = {}
            hi_preds_dict = {}
            # Process prediction intervals for all configured LEVELS
            for level_val in LEVELS: # Ensure LEVELS is defined, e.g., [90]
                lo_col = f"{model_name}-lo-{level_val}"
                hi_col = f"{model_name}-hi-{level_val}"
                if lo_col in eval_df.columns: # Check in eval_df after merge
                    lo_preds_dict[str(level_val)] = eval_df[lo_col].values
                if hi_col in eval_df.columns: # Check in eval_df after merge
                    hi_preds_dict[str(level_val)] = eval_df[hi_col].values
            
            all_forecasts_dict[model_name] = {
                'framework': 'neural',
                'predictions': {
                    'mean': mean_pred_series.values if mean_pred_series is not None else np.full(len(eval_df), np.nan),
                    'lo': lo_preds_dict,
                    'hi': hi_preds_dict
                },
                'ds': eval_df['ds'].values,
                'actual': eval_df['y'].values,
                'forecast_method': evaluation_method
            }
            
            if hasattr(model_instance, 'results_') and model_instance.results_ is not None:
                save_best_auto_model_config(model_name, model_instance.results_.best_config, auto_model_configs_dir)

            # Use the user-provided calculate_metrics function
            # eval_df should contain 'y' (actuals) and 'model_name' (predictions) columns
            metrics = calculate_metrics(eval_df, model_name) 
            per_model_time = time.time() - per_model_start_time
            final_results.append({
                'model_name': model_name, 'framework': 'neuralforecast',
                'training_time': per_model_time, # This is per-model operation time
                'evaluation_method': evaluation_method,
                'is_auto': True, # Assuming all neural_models here are 'Auto' type
                'status': 'success' if 'error' not in metrics else 'metrics_error', 
                **metrics # Unpack MAE, RMSE, MAPE, etc.
            })
            print(f"    ✓ {model_name} processed (Test MAE: {metrics.get('mae', 'N/A'):.4f}) in {per_model_time:.2f}s.")
        
        except Exception as e_model:
            per_model_time = time.time() - per_model_start_time
            error_msg = str(e_model)
            print(f"    ✗ Error processing {model_name} (neural): {error_msg}")
            final_results.append({
                'model_name': model_name, 'framework': 'neuralforecast',
                'training_time': per_model_time, 'error': error_msg,
                'status': 'failed', 'is_auto': True,
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'smape': np.nan # Match keys from calculate_metrics
            })

# --- Process results for each STATISTICAL model ---
print(f"  Processing individual statistical model results...")
if 'stat_forecasts_df' not in locals() or stat_forecasts_df is None or stat_forecasts_df.empty: # Ensure stat_forecasts_df exists
    print("    ! No forecasts (stat_forecasts_df) available to process for statistical models. Skipping.")
else:
    for model_instance in stats_models:
        model_name = model_instance.__class__.__name__
        per_model_stat_start_time = time.time()
        print(f"  Processing Statistical Model: {model_name}...")
        try:
            if model_name not in stat_forecasts_df.columns:
                raise ValueError(f"Model prediction column '{model_name}' not found in StatForecast results columns: {stat_forecasts_df.columns.tolist()}")
            
            # Create a DataFrame for the current model's forecast by selecting relevant columns
            # from the comprehensive stat_forecasts_df
            current_model_cols_to_select = ['unique_id', 'ds', model_name]
            lo_preds_stat = {}
            hi_preds_stat = {}

            for level_val in LEVELS: # Ensure LEVELS is defined
                lo_pi_col = f"{model_name}-lo-{level_val}"
                hi_pi_col = f"{model_name}-hi-{level_val}"
                if lo_pi_col in stat_forecasts_df.columns:
                    current_model_cols_to_select.append(lo_pi_col)
                    # Values for all_forecasts_dict will be extracted after merge from eval_df_stat
                if hi_pi_col in stat_forecasts_df.columns:
                    current_model_cols_to_select.append(hi_pi_col)
                    # Values for all_forecasts_dict will be extracted after merge from eval_df_stat
            
            current_model_stat_forecast_df = stat_forecasts_df[list(dict.fromkeys(current_model_cols_to_select))].copy()

            eval_df_stat = test_df.merge(current_model_stat_forecast_df, on=['unique_id', 'ds'], how='inner')
            if eval_df_stat.empty:
                 raise ValueError(f"No matching StatForecast forecasts and actual values for evaluation for model {model_name}. Merged df is empty.")

            mean_pred_stat_series = eval_df_stat[model_name] if model_name in eval_df_stat else None
            
            # Re-extract PI values from eval_df_stat to ensure they are aligned with 'y'
            for level_val in LEVELS:
                lo_pi_col = f"{model_name}-lo-{level_val}"
                hi_pi_col = f"{model_name}-hi-{level_val}"
                if lo_pi_col in eval_df_stat.columns:
                    lo_preds_stat[str(level_val)] = eval_df_stat[lo_pi_col].values
                if hi_pi_col in eval_df_stat.columns:
                    hi_preds_stat[str(level_val)] = eval_df_stat[hi_pi_col].values
                    
            all_forecasts_dict[model_name] = {
                'framework': 'statistical',
                'predictions': {
                    'mean': mean_pred_stat_series.values if mean_pred_stat_series is not None else np.full(len(eval_df_stat), np.nan),
                    'lo': lo_preds_stat,
                    'hi': hi_preds_stat
                },
                'ds': eval_df_stat['ds'].values,
                'actual': eval_df_stat['y'].values,
                'forecast_method': 'direct_forecast' # StatsForecast generally does direct
            }
            
            # Use the user-provided calculate_metrics function
            metrics_stat = calculate_metrics(eval_df_stat, model_name)
            per_model_stat_time = time.time() - per_model_stat_start_time
            
            # Determine if the model is an "auto" model (example for AutoARIMA)
            is_auto_stat = False
            if 'sf' in locals() and sf is not None: # Check if sf object exists
                 is_auto_stat = isinstance(model_instance, getattr(sf, 'AutoARIMA', type(None))) # Add other auto classes if needed
            
            final_results.append({
                'model_name': model_name, 'framework': 'statsforecast',
                'training_time': per_model_stat_time, 
                'evaluation_method': 'direct_forecast',
                'is_auto': is_auto_stat, 
                'status': 'success' if 'error' not in metrics_stat else 'metrics_error', 
                **metrics_stat
            })
            print(f"    ✓ {model_name} processed (Test MAE: {metrics_stat.get('mae', 'N/A'):.4f}) in {per_model_stat_time:.2f}s.")
        except Exception as e_stat_model:
            per_model_stat_time = time.time() - per_model_stat_start_time
            error_msg_stat = str(e_stat_model)
            print(f"    ✗ Error processing {model_name} (statistical): {error_msg_stat}")
            final_results.append({
                'model_name': model_name, 'framework': 'statsforecast',
                'training_time': per_model_stat_time, 'error': error_msg_stat,
                'status': 'failed', 
                'is_auto': isinstance(model_instance, getattr(sf if 'sf' in locals() else None, 'AutoARIMA', type(None))) if 'sf' in locals() and sf is not None else False,
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'smape': np.nan # Match keys from calculate_metrics
            })