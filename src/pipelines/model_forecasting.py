"""
Module for final model fitting, prediction, and rolling forecasts in the Auto Models workflow.
"""
import time
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Configuration and local imports
from config.base import *
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from src.utils.utils import calculate_metrics

def save_best_auto_model_config(model_name: str, best_config: Dict, auto_model_configs_dir: Path):
    """Save the best hyperparameter configuration for an AutoModel."""
    if best_config is None:
        print(f"  INFO: No best_config found for {model_name} to save.")
        return

    config_path = auto_model_configs_dir / f"{model_name}_best_config.json"
    
    try:
        serializable_config = {}
        for key, value in best_config.items():
            if hasattr(value, '__name__'):
                serializable_config[key] = value.__name__
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)

        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)
        print(f"  ✓ Best config for {model_name} saved to {config_path}")
    except Exception as e:
        print(f"  ✗ Error saving best config for {model_name}: {e}")

def rolling_forecast_neural_all_models(
    nf_model: NeuralForecast, # NeuralForecast object containing potentially multiple models
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_length: int, # HORIZON value
    predict_level: List[int] # e.g., [90] for prediction intervals
) -> pd.DataFrame:
    """
    Perform rolling forecast for ALL neural models contained in nf_model.
    Retrains all models in nf_model at each step using new data from test_df.
    """
    print(f"      Performing rolling forecast for all models in the NeuralForecast object.")
    all_forecasts_list = [] # List to store forecast DataFrames from each window
    current_train_df = train_df.copy() # This df will grow with actuals from test_df

    # Prepare column names for future exogenous features
    exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
    futr_cols = ['unique_id', 'ds'] + exog_cols

    # Determine the number of steps for rolling.
    test_duration_steps = len(test_df)

    if horizon_length <= 0:
        raise ValueError("horizon_length must be positive for rolling forecast.")

    n_windows = (test_duration_steps + horizon_length - 1) // horizon_length
    print(f"        Rolling windows: {n_windows}, Horizon per window: {horizon_length}, Test duration steps: {test_duration_steps}")

    for window_idx in range(n_windows):
        print(f"        Processing rolling window {window_idx + 1}/{n_windows}...")
        
        start_idx = window_idx * horizon_length
        end_idx = min(start_idx + horizon_length, test_duration_steps)
        
        # This .iloc slicing assumes test_df is structured such that rows map directly to time steps.
        current_test_window_actuals_df = test_df.iloc[start_idx:end_idx].copy()

        if current_test_window_actuals_df.empty:
            print(f"        Window {window_idx + 1} data is empty, skipping.")
            continue
            
        # Prepare futr_df for prediction (contains 'unique_id', 'ds', and exog_cols for the window)
        futr_df_for_predict = current_test_window_actuals_df[futr_cols].copy()
        
        # If futr_df_for_predict is empty (e.g. futr_cols were missing from test_df slice),
        # it might indicate an issue or a need to predict using h.
        if futr_df_for_predict.empty:
            if not exog_cols and not current_test_window_actuals_df[['unique_id', 'ds']].empty:
                 # If no exogenous features, futr_df only needs unique_id and ds
                futr_df_for_predict = current_test_window_actuals_df[['unique_id', 'ds']].copy()
            else:
                print(f"        futr_df for prediction is empty for window {window_idx + 1}. Cannot predict for this window.")
                continue # Skip to next window if no futr_df can be made

        print(f"        Predicting for window {window_idx + 1} (futr_df rows: {len(futr_df_for_predict)}).")
        # nf_model.predict() will use all models within nf_model and selected level for PIs
        window_forecast_df = nf_model.predict(futr_df=futr_df_for_predict, level=predict_level)
        all_forecasts_list.append(window_forecast_df)
        
        # Update training data with actuals from the current window for the next iteration's fit.
        # Do not refit after the last window's prediction.
        if window_idx < n_windows - 1:
            # Ensure 'y' (actuals) are present in the slice to be appended for meaningful retraining.
            if 'y' not in current_test_window_actuals_df.columns:
                print(f"        Warning: Column 'y' not in test data for window {window_idx + 1}. Models will be refit on data potentially missing new actuals.")
                # Or raise ValueError if 'y' is strictly required.
            
            actuals_to_append = current_test_window_actuals_df # Contains unique_id, ds, y (if available), and exog_cols
            print(f"        Appending {len(actuals_to_append)} observations from window {window_idx + 1} to training data.")
            current_train_df = pd.concat([current_train_df, actuals_to_append], ignore_index=True)
            
            print(f"        Re-fitting models for the next window (current train size: {len(current_train_df)})...")
            fit_val_size = horizon_length if horizon_length > 0 and len(current_train_df) > horizon_length * 2 else None # Ensure val_size is reasonable
            
            # nf_model.fit() will re-train all models within nf_model.
            # Pass PredictionIntervals() configuration for consistency
            nf_model.fit(
                        current_train_df, 
                        val_size=fit_val_size, 
                        verbose=False, 
                        prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
                    ) 
            print(f"        Models re-fitted.")
            
    if not all_forecasts_list:
        print("        Warning: No forecasts were generated during the rolling forecast process.")
        return pd.DataFrame()

    final_concatenated_forecasts_df = pd.concat(all_forecasts_list, ignore_index=True)
    return final_concatenated_forecasts_df

def rolling_forecast_neural_all_models_efficient(
    nf_model: NeuralForecast,
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

def perform_final_fit_predict(
    neural_models: List,
    stats_models: List,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    auto_model_configs_dir: Path,
    all_forecasts_dict: Dict  
) -> Tuple[pd.DataFrame, Dict]:
    """Perform final fit and predict on test data for all models."""
    final_results = []
    test_length = len(test_df)

    use_rolling = ENABLE_ROLLING_FORECAST and HORIZON < test_length

    if use_rolling:
        print(f"Using rolling forecast (horizon={HORIZON} < test_length={test_length})")
    else:
        print(f"Using direct multi-step forecast (horizon={HORIZON})")

    # --- Fit-Predict Auto Neural Models ---
    if neural_models:
        print(f"Processing {len(neural_models)} Auto Neural models...")
        nf = None
        all_nf_forecasts_df = None # This will store results from either direct or rolling predict

        try:
            # === Phase 1: Collective Fit ===
            print(f"  Fitting {len(neural_models)} Auto Neural models collectively...")
            nf = NeuralForecast(models=neural_models, freq=FREQUENCY, local_scaler_type=LOCAL_SCALER_TYPE)
            
            fit_start_time = time.time()
            # Initial fit with PredictionIntervals configuration
            nf.fit(
                train_df, 
                val_size=HORIZON if HORIZON > 0 else None, 
                prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
            )
            total_neural_fit_time = time.time() - fit_start_time
            print(f"    ✓ All {len(neural_models)} neural models fitted in {total_neural_fit_time:.2f}s.")

            # === Phase 2: Collective Predict (Direct OR Rolling) ===
            if use_rolling:
                print(f"  Performing rolling forecast for all {len(neural_models)} neural models...")
                rolling_predict_start_time = time.time()
                
                # Call the efficient rolling forecast function with configurable refit frequency
                all_nf_forecasts_df = rolling_forecast_neural_all_models_efficient(
                    nf_model=nf,
                    train_df=train_df.copy(),
                    test_df=test_df.copy(),
                    horizon_length=HORIZON,
                    predict_level=LEVELS,
                    refit_frequency=ROLLING_REFIT_FREQUENCY  # Use configuration setting
                )
                total_neural_predict_time = time.time() - rolling_predict_start_time
                if all_nf_forecasts_df is not None and not all_nf_forecasts_df.empty:
                    print(f"    ✓ Rolling forecasts for all neural models generated in {total_neural_predict_time:.2f}s.")
                else:
                    print(f"    ! Rolling forecasts generated an empty or None result in {total_neural_predict_time:.2f}s.")
                    if all_nf_forecasts_df is None: all_nf_forecasts_df = pd.DataFrame() # Ensure it's an empty df
            else: # Direct multi-step forecast
                print(f"  Predicting with all {len(neural_models)} fitted neural models (direct multi-step forecast)...")
                predict_start_time = time.time()
                exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
                futr_df = test_df[['unique_id', 'ds'] + exog_cols].copy() if not test_df.empty else None

                if futr_df is not None and not futr_df.empty:
                    all_nf_forecasts_df = nf.predict(futr_df=futr_df, level=LEVELS)
                elif HORIZON > 0 :
                     all_nf_forecasts_df = nf.predict(h=HORIZON, level=LEVELS)
                else: # Should not happen if HORIZON is well-defined
                    all_nf_forecasts_df = pd.DataFrame() 
                    print("    ! Cannot perform direct predict: test_df is empty and HORIZON is not positive.")
                
                total_neural_predict_time = time.time() - predict_start_time
                if not all_nf_forecasts_df.empty:
                    print(f"    ✓ Direct predictions for all neural models generated in {total_neural_predict_time:.2f}s.")
                else:
                     print(f"    ! Direct predictions generated an empty result in {total_neural_predict_time:.2f}s.")


            # === Phase 3: Process results for each model ===
            print(f"  Processing individual model results...")
            if all_nf_forecasts_df is None or all_nf_forecasts_df.empty:
                print("    ! No forecasts (all_nf_forecasts_df) available to process for neural models. Skipping Phase 3.")
            else:
                for i, model_instance in enumerate(neural_models, 1):
                    model_name = model_instance.__class__.__name__
                    per_model_start_time = time.time()
                    print(f"  [{i}/{len(neural_models)}] Processing {model_name}...")
                    
                    try:
                        # Extract this model's forecasts from the collective `all_nf_forecasts_df`
                        cols_to_select = ['unique_id', 'ds']
                        # Model columns can be model_name itself or model_name-lo-XX, model_name-hi-XX
                        model_pred_cols = [col for col in all_nf_forecasts_df.columns if col.startswith(model_name)]

                        if not model_pred_cols and model_name in all_nf_forecasts_df.columns: # for just 'ModelName' column
                            cols_to_select.append(model_name)
                        cols_to_select.extend(model_pred_cols)
                        
                        # Ensure unique columns, preserving order
                        current_model_forecasts_df = all_nf_forecasts_df[list(dict.fromkeys(cols_to_select))].copy()
                        
                        evaluation_method = "rolling_forecast" if use_rolling else "direct_forecast"

                        # Merge with actuals for evaluation
                        eval_df = test_df.merge(current_model_forecasts_df, on=['unique_id', 'ds'], how='inner')
                        if eval_df.empty:
                            # (Provide more context for debugging if needed)
                            raise ValueError(f"No matching forecasts and actual values for evaluation for model {model_name}. Merged df is empty.")
                        
                        # Extract predictions (mean, lo, hi) for all_forecasts_dict
                        mean_pred_series = current_model_forecasts_df[model_name] if model_name in current_model_forecasts_df else None
                        
                        lo_preds_dict = {}
                        hi_preds_dict = {}
                        # Process prediction intervals for all configured levels
                        for level in LEVELS:
                            lo_col = f"{model_name}-lo-{level}"
                            hi_col = f"{model_name}-hi-{level}"
                            if lo_col in current_model_forecasts_df.columns:
                                lo_preds_dict[str(level)] = current_model_forecasts_df[lo_col].values
                            if hi_col in current_model_forecasts_df.columns:
                                hi_preds_dict[str(level)] = current_model_forecasts_df[hi_col].values
                        
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

                        metrics = calculate_metrics(eval_df, model_name)
                        per_model_time = time.time() - per_model_start_time
                        final_results.append({
                            'model_name': model_name, 'framework': 'neuralforecast',
                            'training_time': per_model_time,
                            'evaluation_method': evaluation_method,
                            'is_auto': True, 'status': 'success', **metrics
                        })
                        print(f"    ✓ {model_name} processed (Test MAE: {metrics.get('mae', 'N/A'):.4f}) in {per_model_time:.2f}s.")
                    
                    except Exception as e_model:
                        # (Error handling for individual model processing as before)
                        per_model_time = time.time() - per_model_start_time
                        error_msg = str(e_model)
                        print(f"    ✗ Error processing {model_name}: {error_msg}")
                        final_results.append({
                            'model_name': model_name, 'framework': 'neuralforecast',
                            'training_time': per_model_time, 'error': error_msg,
                            'status': 'failed', 'is_auto': True,
                            'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                        })

        except Exception as e_global_nf:
            print(f"  ✗ Global error in NeuralForecast fit/predict phase: {str(e_global_nf)}")
            for model_instance in neural_models: # Log failure for all neural models if global part fails
                model_name = model_instance.__class__.__name__
                final_results.append({
                    'model_name': model_name, 'framework': 'neuralforecast', 
                    'training_time': 0.0, 
                    'error': f"Global NeuralForecast error: {str(e_global_nf)}", 
                    'status': 'failed', 'is_auto': True,
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })
    
    # --- Fit-Predict Statistical Models (Your existing logic, seems correct) ---
    if stats_models:
        print(f"Fit-predicting {len(stats_models)} statistical models...")
        # Ensure train_df for stats models only contains y and no exogenous variables if models don't support them
        df_stat_train = train_df[['unique_id', 'ds', 'y']].copy()
        try:
            sf = StatsForecast(models=stats_models, freq=FREQUENCY, n_jobs=-1) # Added n_jobs for potential speedup
            
            # --- Fit models ---
            fit_start_time_stats = time.time()
            sf.fit(df_stat_train, prediction_intervals=ConformalIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL))
            total_stats_fit_time = time.time() - fit_start_time_stats
            print(f"  ✓ All {len(stats_models)} statistical models fitted in {total_stats_fit_time:.2f}s.")

            # --- Predict ---
            predict_start_time_stats = time.time()
            # `h` should be the forecast horizon. If test_df represents this horizon:
            h = len(test_df['ds'].unique()) if not test_df.empty else HORIZON
            if h == 0: raise ValueError("Horizon 'h' for StatsForecast cannot be zero.")

            # For StatsForecast, futr_df is passed to `predict` if models use exogenous vars
            exog_cols_stats = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
            X_futr = test_df[['unique_id', 'ds'] + exog_cols_stats].copy() if exog_cols_stats and not test_df.empty else None
            
            stat_forecasts_df = sf.predict(h=h, X_df=X_futr, level=LEVELS) # Use LEVELS configuration
            total_stats_predict_time = time.time() - predict_start_time_stats
            print(f"  ✓ Predictions for all statistical models generated in {total_stats_predict_time:.2f}s.")
            
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
        except Exception as e_global_sf:
            print(f"  ✗ Global error in StatsForecast fit/predict: {str(e_global_sf)}")
            for model_instance in stats_models:
                model_name = model_instance.__class__.__name__
                final_results.append({
                    'model_name': model_name, 'framework': 'statsforecast', 'training_time': 0.0,
                    'error': f"Global StatsForecast error: {str(e_global_sf)}", 'status': 'failed', 
                    'is_auto': False, # Or determine based on model type
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })

    final_results_df = pd.DataFrame(final_results)
    successful_final = final_results_df[final_results_df['status'] == 'success']
    print(f"Final fit-predict completed: {len(successful_final)}/{len(final_results_df)} models successful")
    return final_results_df, all_forecasts_dict 