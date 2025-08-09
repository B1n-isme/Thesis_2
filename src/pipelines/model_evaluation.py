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
    extract_model_names_from_columns,
    save_best_configurations, 
    process_cv_results,
    get_horizon_directories
)
from src.dataset.data_preparation import prepare_pipeline_data

# Get dynamic directories based on HORIZON
CV_DIR, _, _ = get_horizon_directories()


def back_transform_log_returns(cv_df: pd.DataFrame, original_prices: pd.DataFrame, model_names: List[str]):
    """
    Back-transforms log return forecasts to price forecasts recursively.
    """
    print("Back-transforming log returns to prices...")
    
    cv_df_transformed = cv_df.copy()
    
    # Prepare original prices dataframe for merging. This contains the actual prices.
    price_reference = original_prices[['ds', 'y']].rename(columns={'y': 'price_at_cutoff'})
    
    # Ensure date columns are in datetime format for a reliable merge
    cv_df_transformed['cutoff'] = pd.to_datetime(cv_df_transformed['cutoff'])
    cv_df_transformed['ds'] = pd.to_datetime(cv_df_transformed['ds'])
    price_reference['ds'] = pd.to_datetime(price_reference['ds'])

    # Merge to get the last known true price (at the cutoff date) for each forecast window.
    cv_df_transformed = pd.merge(
        cv_df_transformed,
        price_reference,
        how='left',
        left_on='cutoff',
        right_on='ds'
    ).drop(columns='ds_y').rename(columns={'ds_x': 'ds'})

    # Sort to ensure correct order for cumulative calculations
    cv_df_transformed = cv_df_transformed.sort_values(by=['unique_id', 'cutoff', 'ds']).reset_index(drop=True)

    # Back-transform 'y' (true log returns) to true prices.
    # P_true(t) = P(cutoff) * exp(cumsum of y_log_returns from cutoff+1 to t)
    cv_df_transformed['y'] = cv_df_transformed.groupby(['unique_id', 'cutoff'])['y'].transform(
        lambda x: cv_df_transformed.loc[x.index, 'price_at_cutoff'] * np.exp(x.cumsum())
    )

    # Back-transform model predictions (forecasted log returns) to forecasted prices
    for model in model_names:
        # P_hat(t) = P(cutoff) * cumprod(exp(y_hat_log_return)) from cutoff+1 to t
        # We use a more numerically stable equivalent: P_hat(t) = P(cutoff) * exp(cumsum(y_hat_log_return))
        # This avoids underflow from multiplying many small numbers from exp(large_negative_log_return).
        group_transform = cv_df_transformed.groupby(['unique_id', 'cutoff'])[model].transform(
            lambda x: cv_df_transformed.loc[x.index, 'price_at_cutoff'] * np.exp(np.clip(x, -15, 15).cumsum())
        )
        cv_df_transformed[model] = group_transform

    # Drop the helper column
    cv_df_transformed = cv_df_transformed.drop(columns=['price_at_cutoff'])
    
    print("✅ Back-transformation complete.")
    return cv_df_transformed


def back_transform_forecasts_from_log_returns(
    forecast_df: pd.DataFrame, 
    original_df: pd.DataFrame, 
    model_names: List[str]
) -> pd.DataFrame:
    """
    Back-transforms final forecasted log returns to prices for a hold-out test set.

    This function is designed for a final evaluation scenario, where predictions
    are made for a contiguous block of future time steps. It uses the last known
    price from before the forecast period to recursively calculate predicted prices.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing model predictions in log returns.
                                    It must have a 'ds' column and columns for each model.
                                    The 'y' column should contain the actual prices for the forecast period.
        original_df (pd.DataFrame): The complete original DataFrame with untransformed prices.
                                    Used to find the last price before the forecast starts.
        model_names (List[str]): A list of model names corresponding to columns in `forecast_df`.

    Returns:
        pd.DataFrame: The forecast DataFrame with model predictions transformed to price scale.
                      The 'y' column remains unchanged.
    """
    print("Back-transforming final forecasts from log returns to prices...")
    transformed_df = forecast_df.copy()

    # Find the last date in the training data from the original dataframe, which should be
    # one day before the first prediction date.
    first_forecast_date = transformed_df['ds'].min()
    last_train_date = first_forecast_date - pd.Timedelta(days=1)
    
    # Get the last actual price from the training period.
    last_price_row = original_df.loc[original_df['ds'] == last_train_date]
    
    if last_price_row.empty:
        raise ValueError(f"Could not find the last price at {last_train_date.date()} in the original dataframe to start back-transformation.")
        
    last_price = last_price_row['y'].iloc[0]
    print(f"  Starting back-transformation with last known price: {last_price:.2f} at {last_train_date.date()}")

    # Back-transform each model's predicted log returns to prices.
    for model in model_names:
        # P_hat(t) = P_last * exp(cumsum(y_hat_log_return))
        log_returns = transformed_df[model].values
        # Clip to prevent overflow from extreme predicted values
        clipped_log_returns = np.clip(log_returns, a_min=-15, a_max=15)
        # Calculate prices recursively from the last known price
        transformed_df[model] = last_price * np.exp(np.cumsum(clipped_log_returns))
    
    # The 'y' column in the input forecast_df already contains the true prices,
    # so no transformation is needed for it.
    
    print("✅ Final forecast back-transformation complete.")
    return transformed_df


def perform_cross_validation(stat_models: List, neural_models: List, train_df: pd.DataFrame, original_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], Dict]:
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
            auto_model_names = extract_model_names_from_columns(cv_df.columns.tolist())
            if not auto_model_names:
                raise ValueError("No model columns with 'Auto' prefix found in the CV dataframe.")

            # Back-transform predictions from log-return to price
            cv_df = back_transform_log_returns(cv_df, original_df, auto_model_names)

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
                refit=True,
                prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL),
                level=LEVELS,
            )

            if cv_df.empty:
                raise ValueError("Cross-validation returned an empty dataframe.")
            
            # Extract model names exclusively from dataframe columns
            auto_model_names = extract_model_names_from_columns(cv_df.columns.tolist())
            if not auto_model_names:
                raise ValueError("No model columns with 'Auto' prefix found in the CV dataframe.")

            # Back-transform predictions from log-return to price
            cv_df = back_transform_log_returns(cv_df, original_df, auto_model_names)
            
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
    # from src.models.mlforecast.models import get_ml_models
    from src.models.neuralforecast.models import get_neural_models
    
    # This main block now demonstrates only the Stat/Neural CV part
    train_df, _, hist_exog_list, _, original_df = prepare_pipeline_data(apply_transformations=True)
    stat_models = get_statistical_models(season_length=7)
    neural_models = get_neural_models(horizon=HORIZON, num_samples=NUM_SAMPLES_PER_MODEL, hist_exog_list=hist_exog_list)
    
    print("\n--- Running Standalone Stat/Neural Cross-Validation ---")
    cv_dfs, metadata = perform_cross_validation(
        stat_models=stat_models, 
        neural_models=neural_models, 
        train_df=train_df,
        original_df=original_df
    )
    
    # To see the results, we can consolidate and process them here
    if cv_dfs:
        consolidated_df = pd.concat(cv_dfs, ignore_index=True)
        cv_results_df = process_cv_results(consolidated_df, original_df)
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