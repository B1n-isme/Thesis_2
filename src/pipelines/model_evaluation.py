"""
Module for model evaluation (cross-validation and best config extraction) in the Auto Models workflow.
"""
import time
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Tuple
from pathlib import Path

# Third-party import
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast
from mlforecast.utils import PredictionIntervals as MLPredictionIntervals

# Local import
from config.base import *
from src.utils.utils import (
    calculate_metrics,
    extract_auto_model_names_from_columns,
)
from src.dataset.data_preparation import prepare_pipeline_data

from src.utils.utils import save_best_configurations, process_cv_results

CV_DIR = Path(CV_DIR)


def perform_cross_validation(stat_models: List, neural_models: List, train_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], Dict]:
    """Perform cross-validation for statistical and neural models and return their CV dataframes and metadata."""
    all_cv_dfs = []
    model_metadata = {}
    fitted_objects = {'neural': None, 'stat': None}

    # Cross-validate All Statistical Models at once
    if stat_models:
        print(f"Cross-validating {len(stat_models)} statistical models...")
        start_time = time.time()
        try:
            sf = StatsForecast(models=stat_models, freq='D', verbose=True)
            cv_df = sf.cross_validation(
                h=HORIZON,
                df=train_df,
                n_windows=CV_N_WINDOWS,
                step_size=CV_STEP_SIZE,
                input_size=INPUT_SIZE,
                prediction_intervals=ConformalIntervals(h=HORIZON, n_windows=PI_N_WINDOWS_FOR_CONFORMAL),
                level=LEVELS
            )
            if cv_df.empty:
                raise ValueError("Cross-validation returned an empty dataframe.")
            
            # Extract model names exclusively from dataframe columns
            auto_model_names = extract_auto_model_names_from_columns(cv_df.columns.tolist())
            if not auto_model_names:
                raise ValueError("No model columns with 'Auto' prefix found in the CV dataframe.")

            fitted_objects['stat'] = sf
            all_cv_dfs.append(cv_df)
            training_time = time.time() - start_time
            
            # Populate metadata using only names from the dataframe
            for model_name in auto_model_names:
                model_metadata[model_name] = {'training_time': training_time / len(auto_model_names), 'status': 'success'}
            print(f"  ✓ CV completed for {len(auto_model_names)} statistical models: {auto_model_names}")
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)
            print(f"  ✗ Error during CV for the statistical models batch: {error_msg}")
            # Log a single error for the whole batch since we cannot get individual model names without a dataframe
            model_metadata['StatsForecast_Error'] = {
                'training_time': training_time,
                'error': error_msg,
                'status': 'failed'
            }

    # Cross-validate All Neural Models at once
    if neural_models:
        print(f"Cross-validating {len(neural_models)} Auto Neural models...")
        start_time = time.time()
        try:
            nf = NeuralForecast(models=neural_models, freq='D', local_scaler_type='robust')
            cv_df = nf.cross_validation(
                df=train_df,
                n_windows=CV_N_WINDOWS,
                step_size=CV_STEP_SIZE,
                val_size=HORIZON,
                refit=True,
                prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL),
                level=LEVELS,
            )

            if cv_df.empty:
                raise ValueError("Cross-validation returned an empty dataframe.")
            
            # Extract model names exclusively from dataframe columns
            auto_model_names = extract_auto_model_names_from_columns(cv_df.columns.tolist())
            if not auto_model_names:
                raise ValueError("No model columns with 'Auto' prefix found in the CV dataframe.")
            
            fitted_objects['neural'] = nf
            all_cv_dfs.append(cv_df)
            training_time = time.time() - start_time

            # Populate metadata using only names from the dataframe
            for model_name in auto_model_names:
                model_metadata[model_name] = {'training_time': training_time / len(auto_model_names), 'status': 'success'}
            print(f"  ✓ CV completed for {len(auto_model_names)} neural models: {auto_model_names}")
                    
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)
            print(f"  ✗ Error during CV for the neural models batch: {error_msg}")
            # Log a single error for the whole batch
            model_metadata['NeuralForecast_Error'] = {
                'training_time': training_time,
                'error': error_msg,
                'status': 'failed'
            }
    best_configs = save_best_configurations(fitted_objects, CV_DIR)

    print("Best configs saved for:", list(best_configs.keys()))
    
    print(f"\nCross-validation for Stat/Neural models completed. Returning {len(all_cv_dfs)} CV dataframes.")
    return all_cv_dfs, model_metadata


if __name__ == "__main__":
    from src.models.statsforecast.models import get_statistical_models
    from src.models.neuralforecast.models import get_neural_models
    
    # This main block now demonstrates only the Stat/Neural CV part
    train_df, _, hist_exog_list, _ = prepare_pipeline_data()
    stat_models = get_statistical_models(season_length=HORIZON)
    neural_models = get_neural_models(horizon=HORIZON, num_samples=NUM_SAMPLES_PER_MODEL, hist_exog_list=hist_exog_list)
    
    print("\n--- Running Standalone Stat/Neural Cross-Validation ---")
    cv_dfs, metadata = perform_cross_validation(
        stat_models=stat_models, 
        neural_models=[], 
        train_df=train_df
    )
    
    # To see the results, we can consolidate and process them here
    if cv_dfs:
        consolidated_df = pd.concat(cv_dfs, ignore_index=True)
        cv_results_df = process_cv_results(consolidated_df, metadata)
        print("\n--- CV Results (Stat/Neural) ---")
        print(cv_results_df.head())
    else:
        print("\nNo CV dataframes were generated.")

    # export the dataframes with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if cv_dfs:
        for i, df in enumerate(cv_dfs):
            df.to_csv(CV_DIR / f'cv_df_{i}_{timestamp}.csv', index=False)
        cv_results_df.to_csv(CV_DIR / f'cv_metrics_{timestamp}.csv', index=False)