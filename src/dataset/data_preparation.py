"""
Data preparation module for neural forecasting pipeline.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from config.base import (
    DATA_PATH, DATE_COLUMN, TARGET_COLUMN, TARGET_RENAMED, 
    DATE_RENAMED, UNIQUE_ID_VALUE, HORIZON, TEST_LENGTH_MULTIPLIER
)
from src.utils.utils import get_historical_exogenous_features, print_data_info
from statsmodels.tsa.stattools import adfuller


def load_and_prepare_data(data_path=None):
    """Load and prepare the dataset for forecasting."""
    print("Loading and preparing data...")
    
    # Use provided data_path or default to DATA_PATH
    if data_path is None:
        data_path = DATA_PATH
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Rename columns
    df = df.rename(columns={DATE_COLUMN: DATE_RENAMED, TARGET_COLUMN: TARGET_RENAMED})
    
    # Add unique_id and convert date
    df['unique_id'] = UNIQUE_ID_VALUE
    df[DATE_RENAMED] = pd.to_datetime(df[DATE_RENAMED])
    
    df.reset_index(drop=True, inplace=True)
    
    return df


def difference_non_stationary_features(df: pd.DataFrame, exog_list: List[str]) -> pd.DataFrame:
    """Checks for stationarity in exogenous features and applies differencing if needed."""
    print("Checking for non-stationary exogenous features...")
    df_transformed = df.copy()
    for col in exog_list:
        # Cannot test stationarity on columns with NaNs, fill them before test
        if df_transformed[col].isnull().any():
            # Using forward fill as a simple imputation method
            df_transformed[col] = df_transformed[col].fillna(method='ffill').fillna(method='bfill')
            if df_transformed[col].isnull().any():
                print(f"  - Skipping stationarity test for '{col}' due to persistent NaN values after imputation.")
                continue

        p_value = adfuller(df_transformed[col].dropna())[1] # dropna just in case
        if p_value > 0.05:
            print(f"  - Feature '{col}' is non-stationary (p-value: {p_value:.4f}). Applying differencing.")
            df_transformed[col] = df_transformed[col].diff()
    
    print("✅ Stationarity check complete.")
    return df_transformed


def transform_target_to_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the target column 'y' to its log return."""
    print("Transforming target 'y' to log returns...")
    df_transformed = df.copy()
    
    # Ensure 'y' is positive before taking the log
    if (df_transformed['y'] <= 0).any():
        print("Warning: Non-positive values found in target 'y'. Log transform may fail.")
        # Replace non-positive values with a small epsilon or handle as per domain knowledge
        # For now, we will proceed, but this should be reviewed.
        
    df_transformed['y'] = np.log(df_transformed['y']).diff()
    df_transformed = df_transformed.dropna(subset=['y']).reset_index(drop=True)
    
    print("✅ Target transformed to log returns.")
    return df_transformed


def split_data(df, horizon, test_length_multiplier):
    """Split data into train and test sets."""
    
    test_length = horizon * test_length_multiplier
    
    # print(f"Forecast horizon (h) set to: {horizon} days")
    
    # Validate data length
    if len(df) <= test_length:
        raise ValueError(
            "Not enough data to create a test set of the desired length. "
            "Decrease test_length or get more data."
        )
    
    # Split data
    train_df = df.iloc[:-test_length].copy()
    test_df = df.iloc[-test_length:].copy()
    
    # Print information
    print_data_info(df, train_df, test_df)
    
    return train_df, test_df


def prepare_data(horizon, test_length_multiplier, data_path=None, apply_transformations: bool = True):
    """Complete data preparation pipeline.
    
    Args:
        horizon (int): Forecast horizon in days
        test_length_multiplier (int): Multiplier to determine test set length
        data_path (str, optional): Path to data file. Defaults to DATA_PATH.
        apply_transformations (bool): If True, applies log-return and differencing.
    """
    # Load and prepare data
    df = load_and_prepare_data(data_path=data_path)
    
    # Store original df for back-transformation reference, which is always untransformed
    original_df_for_reference = df.copy()

    if apply_transformations:
        print("Applying transformations (log-return and differencing)...")
        # Get historical exogenous features from the original data
        hist_exog_list = get_historical_exogenous_features(df)
        
        # Handle non-stationarity in exogenous features
        df_stationary = difference_non_stationary_features(df, hist_exog_list)

        # Transform target to log returns using the stationarized df
        df_to_split = transform_target_to_log_return(df_stationary)
        print("✅ Transformations applied.")
    else:
        print("Skipping transformations. Loading data as is.")
        df_to_split = df

    # Split data
    train_df, test_df = split_data(df_to_split, horizon, test_length_multiplier)
    
    # Reorder columns for consistency, using the final list of exogenous features
    final_exog_list = get_historical_exogenous_features(df_to_split)
    train_df = train_df[['unique_id', 'ds', 'y'] + final_exog_list]
    test_df = test_df[['unique_id', 'ds', 'y'] + final_exog_list]
    
    return train_df, test_df, final_exog_list, original_df_for_reference


def prepare_pipeline_data(
    horizon: int = HORIZON, 
    test_length_multiplier: int = TEST_LENGTH_MULTIPLIER, 
    data_path: str = DATA_PATH, 
    apply_transformations: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict, pd.DataFrame]:
    """
    Enhanced data preparation for the pipeline with detailed logging and metadata.
    
    Args:
        horizon (int, optional): Forecast horizon in days. Defaults to HORIZON.
        test_length_multiplier (int, optional): Multiplier for test set length. Defaults to TEST_LENGTH_MULTIPLIER.
        data_path (str, optional): Path to data file. Defaults to DATA_PATH.
        apply_transformations (bool, optional): If False, skips log-return and differencing. Defaults to True.
    
    Returns:
        Tuple of (train_df, test_df, hist_exog_list, data_info_dict, original_df)
    """
    print("\n📊 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    train_df, test_df, hist_exog_list, original_df = prepare_data(
        horizon=horizon,
        test_length_multiplier=test_length_multiplier,
        data_path=data_path,
        apply_transformations=apply_transformations
    )
    
    data_info = {
        'total_samples': len(train_df) + len(test_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': hist_exog_list,
        'horizon': horizon
    }
    
    print(f"✅ Data prepared successfully")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Test samples: {len(test_df):,}")
    print(f"   • Features: {len(hist_exog_list)} exogenous variables")
    print(f"   • Forecast horizon: {horizon} days")
    
    return train_df, test_df, hist_exog_list, data_info, original_df


def main():
    """Main function to demonstrate prepare_pipeline_data usage."""
    print("--- Running data preparation with transformations (default) ---")
    train_df, test_df, _, _, _ = prepare_pipeline_data(apply_transformations=True)
    print("\nTransformed training data head:")
    print(train_df.head())
    
    print("\n" + "="*50 + "\n")

    print("--- Running data preparation without transformations ---")
    train_df_no_transform, _, _, _, _ = prepare_pipeline_data(apply_transformations=False)
    print("\nUntransformed training data head:")
    print(train_df_no_transform.head())


if __name__ == "__main__":
    main()
