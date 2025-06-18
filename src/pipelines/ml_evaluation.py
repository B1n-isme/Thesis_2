"""
Module for ML model evaluation (cross-validation and best config extraction).
"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Third-party import
from mlforecast.auto import AutoMLForecast

# Local import
from config.base import (
    HORIZON,
    CV_N_WINDOWS,
    CV_STEP_SIZE,
    LEVELS,
)
from src.utils.utils import (
    extract_model_names_from_columns,
    save_best_configurations,
    get_horizon_directories,
    calculate_metrics,
    save_dict_to_json,
    process_cv_results,
)

# Get dynamic directories based on HORIZON
CV_DIR, FINAL_DIR, _ = get_horizon_directories()


def perform_ml_evaluation(
    ml_models: List, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Dict | None]:
    """
    Perform cross-validation and holdout forecasting for ML models.
    """
    model_metadata = {}
    fitted_objects = {"ml": None}
    cv_df, cv_results_df, final_results_df, final_plot_results_dict = None, None, None, None

    if not ml_models:
        return None, None, None, None

    train_df["unique_id"] = train_df["unique_id"].astype("category")
    print(f"Cross-validating {len(ml_models)} Auto ML models...")
    start_time = time.time()

    try:
        amf = AutoMLForecast(
            models=ml_models,
            freq="D",
            season_length=7,
            fit_config=lambda trial: {"static_features": ["unique_id"]},
            num_threads=4,
        )
        amf.fit(train_df, n_windows=CV_N_WINDOWS, h=HORIZON, num_samples=1)
        
        training_time = time.time() - start_time
        fitted_objects["ml"] = amf

        cv_dfs = []
        all_forecasts_dfs = []

        exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
        X_df = test_df[['unique_id', 'ds'] + exog_cols].copy() if exog_cols else None

        for model_name, model_obj in amf.models_.items():
            print(f"Running CV for {model_name}...")
            cv_df_model = model_obj.cross_validation(
                df=train_df,
                h=HORIZON,
                n_windows=CV_N_WINDOWS,
                step_size=CV_STEP_SIZE,
                static_features=['unique_id']
            )
            if not cv_df_model.empty:
                cv_dfs.append(cv_df_model.set_index(['unique_id', 'ds', 'cutoff', 'y']))
            else:
                print(f"Warning: CV for {model_name} returned an empty dataframe.")
            
            print(f"Generating forecasts with {model_name}...")
            forecast_df = model_obj.predict(h=HORIZON, X_df=X_df, level=LEVELS)
            if not forecast_df.empty:
                all_forecasts_dfs.append(forecast_df.set_index(['unique_id', 'ds']))
            else:
                print(f"Warning: Forecast with {model_name} returned an empty dataframe.")

        # --- Process CV Results ---
        if cv_dfs:
            cv_df = pd.concat(cv_dfs, axis=1).reset_index()
            auto_model_names_cv = extract_model_names_from_columns(cv_df.columns.tolist())
            for model_name in auto_model_names_cv:
                model_metadata[model_name] = {'training_time': training_time / len(auto_model_names_cv), 'status': 'success'}
            cv_results_df = process_cv_results(cv_df, model_metadata)
            print(f"  ✓ CV completed for {len(auto_model_names_cv)} ML models: {auto_model_names_cv}")
        else:
            print("Warning: Cross-validation for all ML models resulted in empty dataframes.")

        # --- Process Holdout Forecasts ---
        if all_forecasts_dfs:
            final_results = []
            final_plot_results_dict = {}
            consolidated_forecasts_df = pd.concat(all_forecasts_dfs, axis=1).reset_index()
            consolidated_forecasts_df = consolidated_forecasts_df.merge(test_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

            final_plot_results_dict['common'] = {'ds': consolidated_forecasts_df['ds'].values, 'actual': consolidated_forecasts_df['y'].values}
            final_plot_results_dict['models'] = {}

            auto_model_names_fc = extract_model_names_from_columns(consolidated_forecasts_df.columns.tolist())
            
            for model_name in auto_model_names_fc:
                lo_preds, hi_preds = {}, {}
                for level in LEVELS:
                    lo_preds[str(level)] = consolidated_forecasts_df[f"{model_name}-lo-{level}"].values
                    hi_preds[str(level)] = consolidated_forecasts_df[f"{model_name}-hi-{level}"].values

                final_plot_results_dict['models'][model_name] = {
                    'predictions': {'mean': consolidated_forecasts_df[model_name].values, 'lo': lo_preds, 'hi': hi_preds},
                    'forecast_method': 'direct_forecast'
                }
                
                metrics = calculate_metrics(consolidated_forecasts_df, [model_name])
                model_metrics = metrics.get(model_name, {})
                
                final_results.append({
                    'model_name': model_name,
                    'training_time': model_metadata.get(model_name, {}).get('training_time', np.nan),
                    'evaluation_method': 'direct_forecast',
                    'status': 'success',
                    **model_metrics
                })
            final_results_df = pd.DataFrame(final_results)

    except Exception as e:
        training_time = time.time() - start_time
        error_msg = str(e)
        print(f"  ✗ Error during ML models evaluation: {error_msg}")
        model_metadata['MLForecast_Error'] = {'training_time': training_time, 'error': error_msg, 'status': 'failed'}
        return None, process_cv_results(pd.DataFrame(), model_metadata), None, None

    save_best_configurations(fitted_objects, CV_DIR)
    return cv_df, cv_results_df, final_results_df, final_plot_results_dict


if __name__ == "__main__":
    from src.models.mlforecast.models import get_ml_models
    from src.dataset.data_preparation import prepare_pipeline_data

    train_df, test_df, _, _ = prepare_pipeline_data()
    ml_models = get_ml_models()

    print("\n--- Running Standalone ML Evaluation Workflow ---")
    cv_df, cv_results_df, final_results_df, final_plot_results_dict = perform_ml_evaluation(
        ml_models=ml_models, train_df=train_df, test_df=test_df
    )

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if cv_df is not None and not cv_df.empty:
        print("\n--- CV Results (ML) ---")
        print(cv_results_df.head())
        cv_df.to_csv(CV_DIR / f"cv_df_ml_{timestamp}.csv", index=False)
        cv_results_df.to_csv(CV_DIR / f"cv_metrics_ml_{timestamp}.csv", index=False)
    else:
        print("\nNo CV dataframe was generated for ML models.")
        
    if final_results_df is not None and not final_results_df.empty:
        print("\n--- Holdout Forecast Results (ML) ---")
        print(final_results_df.head())
        final_results_df.to_csv(FINAL_DIR / f"metrics_results_ml_{timestamp}.csv", index=False)
        plot_results_path = FINAL_DIR / f"final_plot_results_ml_{timestamp}.json"
        save_dict_to_json(final_plot_results_dict, plot_results_path)
        print(f"Saved final plot results to {plot_results_path}")
    else:
        print("\nNo holdout forecast results were generated for ML models.") 