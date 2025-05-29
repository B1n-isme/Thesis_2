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
from config.base import FREQUENCY, LOCAL_SCALER_TYPE, HORIZON, ENABLE_ROLLING_FORECAST
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from src.pipelines.model_evaluation import calculate_test_metrics # Import from new location

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

def rolling_forecast_neural(nf_model: NeuralForecast, train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Perform rolling forecast for neural models."""
    print(f"    Performing rolling forecast for {model_name}")
    all_forecasts = []
    current_train_df = train_df.copy()
    exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
    futr_cols = ['unique_id', 'ds'] + exog_cols
    test_length = len(test_df)
    n_windows = (test_length + HORIZON - 1) // HORIZON
    
    for window in range(n_windows):
        start_idx = window * HORIZON
        end_idx = min(start_idx + HORIZON, test_length)
        current_test_window = test_df.iloc[start_idx:end_idx].copy()
        futr_df = current_test_window[futr_cols].copy()
        window_forecast = nf_model.predict(futr_df=futr_df)
        all_forecasts.append(window_forecast)
        
        if window < n_windows - 1:
            window_actual = current_test_window.copy()
            current_train_df = pd.concat([current_train_df, window_actual], ignore_index=True)
            nf_model.fit(current_train_df, val_size=HORIZON, verbose=False)
            
    return pd.concat(all_forecasts, ignore_index=True)

def perform_final_fit_predict(
    auto_models: List, 
    stat_models: List, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    auto_model_configs_dir: Path,
    all_forecasts_dict: Dict # To store forecasts for plotting
) -> Tuple[pd.DataFrame, Dict]:
    """Perform final fit and predict on test data for all models."""
    final_results = []
    test_length = len(test_df)
    use_rolling = ENABLE_ROLLING_FORECAST and HORIZON < test_length
    
    if use_rolling:
        print(f"Using rolling forecast (horizon={HORIZON} < test_length={test_length})")
    else:
        print(f"Using direct multi-step forecast")

    # Fit-Predict Auto Neural Models
    if auto_models:
        print(f"Fit-predicting {len(auto_models)} Auto Neural models...")
        for i, model_instance in enumerate(auto_models, 1):
            model_name = model_instance.__class__.__name__
            print(f"[{i}/{len(auto_models)}] Fit-Predict {model_name}...")
            start_time = time.time()
            try:
                nf = NeuralForecast(models=[model_instance], freq=FREQUENCY, local_scaler_type=LOCAL_SCALER_TYPE)
                nf.fit(train_df, val_size=HORIZON, verbose=False)
                
                if use_rolling:
                    forecasts = rolling_forecast_neural(nf, train_df, test_df, model_name)
                    evaluation_method = "rolling_forecast"
                else:
                    exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
                    futr_df = test_df[['unique_id', 'ds'] + exog_cols].copy()
                    forecasts = nf.predict(futr_df=futr_df)
                    evaluation_method = "direct_forecast"
                
                eval_df = test_df.merge(forecasts, on=['unique_id', 'ds'], how='inner')
                if eval_df.empty:
                    raise ValueError("No matching forecasts and actual values for evaluation")
                
                all_forecasts_dict[model_name] = {
                    'framework': 'neural', 'predictions': forecasts[model_name].values,
                    'ds': forecasts['ds'].values, 'actual': eval_df['y'].values,
                    'forecast_method': evaluation_method
                }
                
                if hasattr(model_instance, 'results_') and model_instance.results_ is not None:
                    save_best_auto_model_config(model_name, model_instance.results_.best_config, auto_model_configs_dir)

                metrics = calculate_test_metrics(eval_df, model_name)
                training_time = time.time() - start_time
                final_results.append({
                    'model_name': model_name, 'framework': 'neuralforecast',
                    'training_time': training_time, 'evaluation_method': evaluation_method,
                    'is_auto': True, 'status': 'success', **metrics
                })
                print(f"  ✓ {model_name} completed (Test MAE: {metrics.get('mae', 'N/A'):.4f})")
            except Exception as e:
                training_time = time.time() - start_time
                error_msg = str(e)
                print(f"  ✗ Error in fit-predict for {model_name}: {error_msg}")
                final_results.append({
                    'model_name': model_name, 'framework': 'neuralforecast',
                    'training_time': training_time, 'error': error_msg,
                    'status': 'failed', 'is_auto': True,
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })

    # Fit-Predict Statistical Models
    if stat_models:
        print(f"Fit-predicting {len(stat_models)} statistical models...")
        df_stat_train = train_df[['unique_id', 'ds', 'y']].copy()
        try:
            sf = StatsForecast(models=stat_models, freq=FREQUENCY, verbose=True)
            sf.fit(df_stat_train)
            stat_forecasts = sf.predict(h=len(test_df))
            
            for model_instance in stat_models:
                model_name = model_instance.__class__.__name__
                start_time = time.time()
                try:
                    if model_name not in stat_forecasts.columns:
                        raise ValueError(f"Model {model_name} not found in forecast results")
                    eval_df = test_df[['unique_id', 'ds', 'y']].copy()
                    eval_df[model_name] = stat_forecasts[model_name].values
                    
                    all_forecasts_dict[model_name] = {
                        'framework': 'statistical', 'predictions': stat_forecasts[model_name].values,
                        'ds': test_df['ds'].values, 'actual': test_df['y'].values,
                        'forecast_method': 'direct_forecast'
                    }
                    metrics = calculate_test_metrics(eval_df, model_name)
                    training_time = time.time() - start_time
                    final_results.append({
                        'model_name': model_name, 'framework': 'statsforecast',
                        'training_time': training_time, 'evaluation_method': 'direct_forecast',
                        'is_auto': False, 'status': 'success', **metrics
                    })
                    print(f"  ✓ {model_name} completed (Test MAE: {metrics.get('mae', 'N/A'):.4f})")
                except Exception as e:
                    training_time = time.time() - start_time
                    error_msg = str(e)
                    print(f"  ✗ Error processing {model_name}: {error_msg}")
                    final_results.append({
                        'model_name': model_name, 'framework': 'statsforecast',
                        'training_time': training_time, 'error': error_msg,
                        'status': 'failed', 'is_auto': False,
                        'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                    })
        except Exception as e:
            print(f"  ✗ Error in statistical models fit-predict: {str(e)}")
            for model_instance in stat_models:
                model_name = model_instance.__class__.__name__
                final_results.append({
                    'model_name': model_name, 'framework': 'statsforecast', 'training_time': 0.0,
                    'error': str(e), 'status': 'failed', 'is_auto': False,
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })
                
    final_results_df = pd.DataFrame(final_results)
    successful_final = final_results_df[final_results_df['status'] == 'success']
    print(f"Final fit-predict completed: {len(successful_final)}/{len(final_results_df)} models successful")
    return final_results_df, all_forecasts_dict 