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
from mlforecast.auto import AutoMLForecast
from mlforecast.utils import PredictionIntervals as MLPredictionIntervals

from src.utils.utils import (
    calculate_metrics, 
    save_best_configurations, 
    extract_model_names_from_columns,
    rolling_forecast_neural_all_models,
    save_dict_to_json, 
    load_json_to_dict, 
    process_cv_results,
    get_horizon_directories
)

# Get dynamic directories based on HORIZON
CV_DIR, FINAL_DIR, _ = get_horizon_directories()

def perform_final_fit_predict(
    stats_models: List,
    ml_models: Dict,  
    neural_models: List,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    final_model_config_dir: Path,
    final_plot_results_dict: Dict,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Perform final fit and predict on test data for all models.
    For ML models, this function also handles the CV and processing of CV results.
    """
    final_results = []
    test_length = len(test_df)
    
    ml_cv_results_df = pd.DataFrame()

    use_rolling = ENABLE_ROLLING_FORECAST and HORIZON < test_length

    if use_rolling:
        print(f"Using rolling forecast (horizon={HORIZON} < test_length={test_length})")
    else:
        print(f"Using direct multi-step forecast (horizon={HORIZON})")

    # Initialize forecast dataframes and fitted objects for config saving
    stat_forecasts_df = None
    ml_forecasts_df = None 
    neural_forecasts_df = None
    fitted_objects = {}

    # Convert 'unique_id' to category dtype if it is not already
    for df in [train_df, test_df]:
        if 'unique_id' in df.columns and df['unique_id'].dtype == 'object':
            df['unique_id'] = df['unique_id'].astype('category')

    # --- Forecast Statistical Models ---
    if stats_models:
        print(f"Forecasting {len(stats_models)} statistical models...")
        try:
            sf = StatsForecast(models=stats_models, freq='D', verbose=True)
            
            # --- Forecast (fit + predict in one step) ---
            forecast_start_time_stats = time.time()
            h = len(test_df['ds'].unique()) if not test_df.empty else HORIZON
            if h == 0: 
                raise ValueError("Horizon 'h' for StatsForecast cannot be zero.")

            # Prepare future exogenous variables (exclude 'y' column)
            exog_cols_stats = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
            X_df_stats = test_df[['unique_id', 'ds'] + exog_cols_stats].copy() if exog_cols_stats and not test_df.empty else None
            
            stat_forecasts_df = sf.forecast(
                h=h,
                df=train_df,
                X_df=X_df_stats,
                level=LEVELS,
                prediction_intervals=ConformalIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
            )
            total_stats_forecast_time = time.time() - forecast_start_time_stats
            print(f"  ✓ Forecasts for all statistical models generated in {total_stats_forecast_time:.2f}s.")
            
        except Exception as e_global_sf:
            print(f"  ✗ Global error in StatsForecast fit/predict: {str(e_global_sf)}")
            for model_instance in stats_models:
                model_name = model_instance.__class__.__name__
                final_results.append({
                    'model_name': model_name, 'training_time': 0.0,
                    'error': f"Global StatsForecast error: {str(e_global_sf)}", 'status': 'failed',
                    'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
                })

    # --- Fit-Predict ML Models ---
    if ml_models:
        print(f"Fit-predicting {len(ml_models)} Auto ML models...")
        fit_start_time_ml = time.time()
        ml_model_metadata = {}
        try:
            from src.utils.utils import my_init_config, my_fit_config, custom_mae_loss
            
            ml = AutoMLForecast(
                models=ml_models,
                freq='D',
                season_length=7,
                init_config=my_init_config,
                # fit_config=my_fit_config,
                fit_config=lambda trial: {'static_features': ['unique_id']},
                num_threads=4
            )

            train_df['unique_id'] = train_df['unique_id'].astype('category')
            
            # --- Fit models (with CV/hyperparameter tuning) ---
            ml.fit(
                train_df,
                n_windows=CV_N_WINDOWS,
                h=HORIZON,
                num_samples=NUM_SAMPLES_PER_MODEL,
                step_size=CV_STEP_SIZE,
                fitted=True,
                refit=True,
                optimize_kwargs={'timeout': 60},
                loss=custom_mae_loss,
                prediction_intervals=MLPredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL, h=HORIZON)
            )


            total_ml_fit_time = time.time() - fit_start_time_ml
            print(f"  ✓ All {len(ml_models)} ML models fitted (incl. CV) in {total_ml_fit_time:.2f}s.")

            # === Process In-sample CV predictions for ML models ===
            print("  ✓ Generating and processing in-sample (CV) predictions for ML models...")
            ml_cv_df = ml.forecast_fitted_values(level=LEVELS)

            print(ml_cv_df)
            
            auto_model_names = extract_model_names_from_columns(ml_cv_df.columns.tolist())
            if not auto_model_names:
                print("  ! Warning: No 'Auto' model columns found in ML CV dataframe.")
            else:
                time_per_model = total_ml_fit_time / len(auto_model_names)
                for model_name in auto_model_names:
                    ml_model_metadata[model_name] = {'training_time': time_per_model, 'status': 'success'}
                
                # Process the CV results for ML models
                ml_cv_results_df = process_cv_results(ml_cv_df, ml_model_metadata)
                print(f"  ✓ CV results processed for {len(auto_model_names)} ML models.")

            # --- Predict on Test Set---
            predict_start_time_ml = time.time()
            exog_cols_ml = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
            X_df = test_df[['unique_id', 'ds'] + exog_cols_ml].copy() if exog_cols_ml and not test_df.empty else None
            
            ml_forecasts_df = pd.DataFrame()
            ml_forecasts_df = ml.predict(
                h=HORIZON,
                X_df=X_df,
                level=LEVELS
            )
            print(ml_forecasts_df)
            total_ml_predict_time = time.time() - predict_start_time_ml
            print(f"  ✓ Predictions for all ML models generated in {total_ml_predict_time:.2f}s.")
            
            # Store ML object for configuration saving
            fitted_objects['ml'] = ml

            
        except Exception as e_global_ml:
            print(f"  ✗ Global error in MLForecast fit/predict batch: {str(e_global_ml)}")
            fit_time = time.time() - fit_start_time_ml
            ml_model_metadata['MLForecast_Error'] = {
                'training_time': fit_time,
                'error': f"Global MLForecast error: {str(e_global_ml)}", 
                'status': 'failed'
            }
            # Also create a failed entry in the results df
            ml_cv_results_df = pd.DataFrame([{
                'model_name': 'MLForecast_Error', 'training_time': fit_time, 
                'error': str(e_global_ml), 'status': 'failed'
            }])

    # --- Fit-Predict Auto Neural Models ---
    if neural_models:
        print(f"Processing {len(neural_models)} Auto Neural models...")
        try:
            # === Phase 1: Collective Fit ===
            print(f"  Fitting {len(neural_models)} Auto Neural models collectively...")
            nf = NeuralForecast(models=neural_models, freq='D', local_scaler_type='robust')
            
            fit_start_time = time.time()
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
                
                neural_forecasts_df = rolling_forecast_neural_all_models(
                    nf_model=nf,
                    train_df=train_df.copy(),
                    test_df=test_df.copy(),
                    horizon_length=HORIZON,
                    predict_level=LEVELS,
                    refit_frequency=ROLLING_REFIT_FREQUENCY
                )
                total_neural_predict_time = time.time() - rolling_predict_start_time
                if neural_forecasts_df is not None and not neural_forecasts_df.empty:
                    print(f"    ✓ Rolling forecasts for all neural models generated in {total_neural_predict_time:.2f}s.")
                else:
                    print(f"    ! Rolling forecasts generated an empty or None result in {total_neural_predict_time:.2f}s.")
                    if neural_forecasts_df is None: 
                        neural_forecasts_df = pd.DataFrame()
            else:
                print(f"  Predicting with all {len(neural_models)} fitted neural models (direct multi-step forecast)...")
                predict_start_time = time.time()
                exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
                futr_df = test_df[['unique_id', 'ds'] + exog_cols].copy() if not test_df.empty else None

                if futr_df is not None and not futr_df.empty:
                    neural_forecasts_df = nf.predict(futr_df=futr_df, level=LEVELS)
                elif HORIZON > 0:
                    neural_forecasts_df = nf.predict(h=HORIZON, level=LEVELS)
                else:
                    neural_forecasts_df = pd.DataFrame() 
                    print("    ! Cannot perform direct predict: test_df is empty and HORIZON is not positive.")
                
                total_neural_predict_time = time.time() - predict_start_time
                if not neural_forecasts_df.empty:
                    print(f"    ✓ Direct predictions for all neural models generated in {total_neural_predict_time:.2f}s.")
                else:
                    print(f"    ! Direct predictions generated an empty result in {total_neural_predict_time:.2f}s.")

            # Store neural object for configuration saving
            fitted_objects['neural'] = nf

        except Exception as e_global_nf:
            print(f"  ✗ Global error in NeuralForecast fit/predict batch: {str(e_global_nf)}")
            # Log a single error for the whole batch
            final_results.append({
                'model_name': 'NeuralForecast_Error', 
                'training_time': 0.0, 
                'error': f"Global NeuralForecast error: {str(e_global_nf)}", 
                'status': 'failed',
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan
            })

    # === CONSOLIDATED PROCESSING: Merge all forecasts and process together ===
    print("Consolidating and processing all forecasts...")
    
    # Start with test data as base (preserve original test_df with actuals)
    consolidated_forecasts_df = test_df[['unique_id', 'ds', 'y']].copy()
    
    # Merge forecasts from all frameworks
    if stat_forecasts_df is not None and not stat_forecasts_df.empty:
        print(f"  Merging StatsForecast results ({len(stat_forecasts_df)} rows)")
        consolidated_forecasts_df = consolidated_forecasts_df.merge(stat_forecasts_df, how='left', on=['unique_id', 'ds'])
    
    if ml_forecasts_df is not None and not ml_forecasts_df.empty:
        print(f"  Merging MLForecast results ({len(ml_forecasts_df)} rows)")
        consolidated_forecasts_df = consolidated_forecasts_df.merge(ml_forecasts_df, how='left', on=['unique_id', 'ds'])
    
    if neural_forecasts_df is not None and not neural_forecasts_df.empty:
        print(f"  Merging NeuralForecast results ({len(neural_forecasts_df)} rows)")
        consolidated_forecasts_df = consolidated_forecasts_df.merge(neural_forecasts_df, how='left', on=['unique_id', 'ds'])

    print(consolidated_forecasts_df)
    print(f"  Consolidated forecast dataframe shape: {consolidated_forecasts_df.shape}")
    
    # Store consolidated forecasts for visualization
    final_plot_results_dict['common'] = {
        'ds': consolidated_forecasts_df['ds'].values,
        'actual': consolidated_forecasts_df['y'].values
    }
    final_plot_results_dict['models'] = {}

    # Process all models at once
    evaluation_method = "rolling_forecast" if use_rolling else "direct_forecast"
    
    # Extract Auto model names directly from consolidated dataframe (all models have Auto prefix)
    auto_model_names = extract_model_names_from_columns(consolidated_forecasts_df.columns.tolist())
    
    print(f"  Found {len(auto_model_names)} Auto models in consolidated dataframe: {auto_model_names}")
    
    # Process each Auto model
    for model_name in auto_model_names:
        start_time = time.time()
        try:
            # Check if model prediction column exists
            if model_name not in consolidated_forecasts_df.columns:
                print(f"    ! Model '{model_name}' prediction column not found, skipping")
                continue
            
            # Extract prediction intervals
            lo_preds = {}
            hi_preds = {}
            for level in LEVELS:
                lo_col = f"{model_name}-lo-{level}"
                hi_col = f"{model_name}-hi-{level}"
                if lo_col in consolidated_forecasts_df.columns:
                    lo_preds[str(level)] = consolidated_forecasts_df[lo_col].values
                if hi_col in consolidated_forecasts_df.columns:
                    hi_preds[str(level)] = consolidated_forecasts_df[hi_col].values
            
            
            # Store in final_plot_results_dict
            final_plot_results_dict['models'][model_name] = {
                'predictions': {
                    'mean': consolidated_forecasts_df[model_name].values,
                    'lo': lo_preds,
                    'hi': hi_preds
                },
                'forecast_method': evaluation_method
            }
            
            # Calculate metrics
            metrics = calculate_metrics(consolidated_forecasts_df, [model_name])
            # Extract metrics for this specific model
            model_metrics = metrics.get(model_name, {})
            processing_time = time.time() - start_time
            
            
            final_results.append({
                'model_name': model_name,
                'training_time': processing_time,
                'evaluation_method': evaluation_method,
                'status': 'success' if 'error' not in model_metrics else 'metrics_error',
                **model_metrics
            })
            
            print(f"    ✓ {model_name} processed (Test MAE: {model_metrics.get('mae', 'N/A'):.4f}) in {processing_time:.2f}s.")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"    ✗ Error processing {model_name}: {error_msg}")
            
            
            final_results.append({
                'model_name': model_name,
                'training_time': processing_time,
                'evaluation_method': evaluation_method,
                'error': error_msg,
                'status': 'failed',
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'smape': np.nan
            })
    
    # # Save best configurations to the specified directory if it exists
    # if final_model_config_dir:
    #     save_best_configurations(fitted_objects, final_model_config_dir)

    final_results_df = pd.DataFrame(final_results)
    
    if not final_results_df.empty:
        successful_final = final_results_df[final_results_df['status'] == 'success']
        print(f"Final fit-predict completed: {len(successful_final)}/{len(final_results_df)} models successful")
    else:
        print("Final fit-predict completed: 0 models processed, no results to show.")

    

    print(f"✅ Final fit-predict step completed. Returning {len(final_results_df)} results.")
    return final_results_df, final_plot_results_dict, ml_cv_results_df

if __name__ == "__main__":
    from src.models.statsforecast.models import get_statistical_models
    from src.models.mlforecast.models import get_ml_models
    from src.models.neuralforecast.models import get_normal_neural_models
    
    from src.pipelines.model_evaluation import prepare_pipeline_data

    # Get data and models
    train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data(horizon=HORIZON, test_length_multiplier=TEST_LENGTH_MULTIPLIER)
    # stat_models = get_statistical_models(season_length=7)
    # ml_models = get_ml_models()
    # neural_models = get_neural_models(horizon=HORIZON, num_samples=NUM_SAMPLES_PER_MODEL, hist_exog_list=hist_exog_list)
    print(CV_DIR)
    neural_models = get_normal_neural_models(horizon=HORIZON, config_path=CV_DIR / "best_configurations_comparison_nf2.yaml", hist_exog_list=hist_exog_list)
    
    # Perform final fit and predict
    final_results_df, final_plot_results_dict, ml_cv_results_df = perform_final_fit_predict(
        stats_models= [],
        ml_models=[],
        neural_models= neural_models,
        train_df=train_df,
        test_df=test_df,
        final_model_config_dir=FINAL_DIR,
        final_plot_results_dict={}
    )
    # Generate timestamp for filenames
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # Save final results to CSV with timestamp
    final_results_path = FINAL_DIR / f"metrics_results_{timestamp}.csv"
    print(f"Saving final results to {final_results_path}...")
    final_results_df.to_csv(final_results_path, index=False)
    print(f"  ✓ Final results saved successfully.")

    # Save final plot results to JSON with timestamp
    plot_results_path = FINAL_DIR / f"final_plot_results_{timestamp}.json"
    print(f"Saving final plot results to {plot_results_path}...")
    try:
        save_dict_to_json(final_plot_results_dict, plot_results_path)
        print("  ✓ Final plot results saved successfully.")
    except Exception as e:
        print(f"  ✗ Error saving plot results: {e}")

    # save ml_cv_results_df to csv
    if ml_cv_results_df is not None and not ml_cv_results_df.empty:
        ml_cv_results_df.to_csv(CV_DIR / f"cv_metrics_{timestamp}.csv", index=False)
    else:
        print("  ✗ No ML CV results to save.")

 