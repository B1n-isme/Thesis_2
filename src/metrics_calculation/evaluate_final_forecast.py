import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from functools import partial

from src.dataset.data_preparation import prepare_pipeline_data
from config.base import DATA_PATH, HORIZON
from src.utils.utils import _calculate_financial_metrics
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mase

def load_and_transform_final_results(json_path: Path) -> Tuple[pd.DataFrame, float, List[str]]:
    """
    Loads final forecast results from JSON and transforms them into a DataFrame.

    Args:
        json_path (Path): Path to the JSON file.

    Returns:
        Tuple[pd.DataFrame, float, List[str]]: A tuple containing the forecast DataFrame,
        the last value of the training set, and a list of model names.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame({
        'ds': pd.to_datetime(data['common']['ds'], unit='ns'),
        'y': data['common']['actual']
    })
    
    model_names = []
    for model, model_data in data['models'].items():
        df[model] = model_data['predictions']['mean']
        model_names.append(model)
    
    # The Naive forecast for the first step is the last value of the training set.
    y_last_train = data['models']['Naive']['predictions']['mean'][0]
    
    return df, y_last_train, model_names

def calculate_final_metrics(
    forecast_df: pd.DataFrame,
    y_last_train: float,
    model_names: List[str],
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates evaluation metrics for the final forecast using utilsforecast.

    Args:
        forecast_df (pd.DataFrame): DataFrame with forecasts.
        y_last_train (float): The last actual value from the training set.
        model_names (List[str]): A list of model names to evaluate.
        train_df (pd.DataFrame): The training dataframe for MASE calculation.

    Returns:
        pd.DataFrame: A DataFrame with the calculated metrics for each model.
    """
    # Ensure 'unique_id' column exists for the evaluate function
    if 'unique_id' not in forecast_df.columns:
        uid = train_df['unique_id'].unique()[0] if 'unique_id' in train_df.columns and train_df['unique_id'].nunique() > 0 else 'ts_1'
        forecast_df['unique_id'] = uid
        if 'unique_id' not in train_df.columns:
            train_df['unique_id'] = uid

    # Use utilsforecast for standard metrics
    evaluation_df = evaluate(
        forecast_df,
        metrics=[mae, rmse, partial(mase, seasonality=7)],
        train_df=train_df,
    )

    # Transform results to wide format
    metrics_df = evaluation_df.drop(columns=['unique_id']).set_index('metric').T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'model_name'})

    # Calculate RMSE/MAE ratio
    if 'rmse' in metrics_df.columns and 'mae' in metrics_df.columns:
        metrics_df['rmse_mae_ratio'] = np.where(
            metrics_df['mae'] == 0, np.nan, metrics_df['rmse'] / metrics_df['mae']
        )
    else:
        metrics_df['rmse_mae_ratio'] = np.nan

    # Prepare for and calculate financial metrics (Directional Accuracy)
    forecast_df['y_lag1'] = forecast_df['y'].shift(1)
    forecast_df.loc[forecast_df.index[0], 'y_lag1'] = y_last_train

    da_scores = {}
    y_true = forecast_df['y'].values
    y_lag1_vals = forecast_df['y_lag1'].values
    for model in model_names:
        y_pred = forecast_df[model].values
        financial_metrics = _calculate_financial_metrics(y_true, y_pred, y_lag1_vals)
        da_scores[model] = financial_metrics.get('da', np.nan)

    da_df = pd.DataFrame.from_dict(da_scores, orient='index', columns=['da'])
    da_df = da_df.reset_index().rename(columns={'index': 'model_name'})

    # Merge standard and financial metrics
    final_results_df = pd.merge(metrics_df, da_df, on='model_name')

    return final_results_df

def main():
    """Main function to execute the evaluation pipeline."""
    # Define paths
    results_dir = Path('results') / f'results_{HORIZON}d' / 'final'
    json_path = results_dir / 'final_plot_results.json'
    output_path = results_dir / 'metrics_results.csv'
    
    # Load forecast results
    forecast_df, y_last_train, model_names = load_and_transform_final_results(json_path)
    
    # Load training data for MASE calculation
    # We use `apply_transformations=False` to get the original values for the naive error calculation
    train_df, _, _, _, _ = prepare_pipeline_data(
        data_path=DATA_PATH, 
        apply_transformations=False
    )

    # Calculate metrics
    metrics_df = calculate_final_metrics(forecast_df, y_last_train, model_names, train_df)
    
    # Reorder columns to be similar to CV results
    metric_cols = ['mae', 'rmse', 'rmse_mae_ratio', 'mase', 'da']
    ordered_cols = ['model_name'] + [col for col in metric_cols if col in metrics_df.columns]
    other_cols = [col for col in metrics_df.columns if col not in ordered_cols]
    final_df = metrics_df[ordered_cols + other_cols]

    # # Save results
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # final_df.to_csv(output_path, index=False)
    
    print(f"Final metrics calculated and saved to {output_path}")
    print("\nMetrics:")
    print(final_df.round(4))

if __name__ == '__main__':
    main() 