"""
Data preparation module for neural forecasting pipeline.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from config.base import (
    RAW_DATA_PATH, DATA_PATH, DATE_COLUMN, TARGET_COLUMN, TARGET_RENAMED, 
    DATE_RENAMED, UNIQUE_ID_VALUE, HORIZON, TEST_LENGTH_MULTIPLIER
)
from src.utils.utils import get_historical_exogenous_features, print_data_info
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox_normmax


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
    """
    Applies transformations to non-stationary features based on their category,
    as per the methodology in feature_transform.md.

    - Raw prices, blockchain data, and trend-following indicators are tested for
      stationarity. If non-stationary, they are transformed using log returns
      if appropriate (checked via Box-Cox), otherwise using simple differencing.
    - Oscillators, spreads, ratios, sentiment, and other bounded indicators are
      kept in their raw form.
    """
    print("Applying transformations to non-stationary exogenous features...")
    df_transformed = df.copy()

    # Features to skip transformation (Group 4 & 5 from feature_transform.md)
    # These are designed to be stationary or their raw value is meaningful.
    SKIP_SUFFIXES = (
        '_diff', '_ratio', '_slope', '_dist', '_dist_norm', 
        '_width', '_sentiment'
    )
    SKIP_EXACT = {
        'btc_rsi_14', 'Confidence_cbbi', 'Fear Greed',
        'btc_volatility_index', 'Gold_Volatility', 'Oil_Volatility', 'CBOE_Volatility'
    }

    for col in exog_list:
        # Check if the feature should be skipped based on its name
        if col in SKIP_EXACT or any(col.endswith(s) for s in SKIP_SUFFIXES):
            print(f"  - Skipping transformation for '{col}' (stationary by design).")
            continue

        # Fill NaNs for the purpose of testing.
        series = df_transformed[col]
        if series.isnull().any():
            series = series.fillna(method='ffill').fillna(method='bfill')
        
        if series.isnull().all():
            print(f"  - Skipping '{col}' due to all NaN values.")
            continue
        
        # Stationarity Test (ADF)
        p_value = adfuller(series.dropna())[1]

        if p_value > 0.05:
            print(f"  - Feature '{col}' is non-stationary (p-value: {p_value:.4f}). Applying transformation.")
            
            # Use the original series from the copied dataframe for transformation
            series_to_transform = df_transformed[col]
            
            # Check for non-positive values. Log/Box-Cox requires positive values.
            if (series_to_transform <= 0).any():
                print(f"    - Contains non-positive values. Applying simple differencing.")
                df_transformed[col] = series_to_transform.diff()
            else:
                # Check for variance stability with Box-Cox to decide on log transform
                try:
                    series_clean = series_to_transform.dropna()
                    # Box-Cox requires at least a few data points to work reliably.
                    if len(series_clean) < 4:
                        raise ValueError("Not enough data points for a reliable Box-Cox analysis.")

                    # Find optimal lambda for Box-Cox.
                    lambda_ = boxcox_normmax(series_clean)
                    
                    if abs(lambda_) < 0.5:  # Threshold for being "close to log"
                        print(f"    - Box-Cox lambda ({lambda_:.2f}) suggests log transform. Applying log returns.")
                        df_transformed[col] = np.log(series_to_transform).diff()
                    else:
                        print(f"    - Box-Cox lambda ({lambda_:.2f}) suggests non-log transform. Applying simple differencing.")
                        df_transformed[col] = series_to_transform.diff()
                except Exception as e:
                    print(f"    - Box-Cox calculation failed for '{col}': {e}. Applying heuristic fallback.")
                    # Heuristic: For all-positive series where Box-Cox fails, log-return is a robust choice
                    # to stabilize variance, which is common for price-like series.
                    series_clean = series_to_transform.dropna()
                    if (series_clean > 0).all():
                        print(f"    - Applying log returns as a robust fallback for the positive series.")
                        df_transformed[col] = np.log(series_to_transform).diff()
                    else:
                        print(f"    - Applying simple differencing as a fallback.")
                        df_transformed[col] = series_to_transform.diff()
        else:
            print(f"  - Feature '{col}' is stationary (p-value: {p_value:.4f}). No transformation needed.")

    print("✅ Feature transformation process complete.")
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


def load_and_process_data(
    data_path: str,
    horizon: int,
    test_length_multiplier: int,
    apply_transformations: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads data, splits it, and optionally applies transformations.
    
    Returns a tuple of (train_df, test_df, original_df_for_reference, hist_exog_list).
    """
    # Load and perform initial preparation (renaming, etc.)
    df = load_and_prepare_data(data_path=data_path)
    
    # Keep a copy of the original, untransformed data for reference (e.g., for back-transformation)
    original_df_for_reference = df.copy()

    # Determine the list of historical exogenous features from the loaded data
    hist_exog_list = get_historical_exogenous_features(df)
    
    df_for_split = df
    if apply_transformations:
        # Apply differencing to non-stationary exogenous features across the entire dataset
        # This is done before splitting to maintain consistency in feature definitions.
        df_for_split = difference_non_stationary_features(df, hist_exog_list)

    # Split the data into training and testing sets
    train_df, test_df = split_data(df_for_split, horizon, test_length_multiplier)
    
    # If transformations are enabled, apply log-return transformation to the training set's target
    if apply_transformations:
        train_df = transform_target_to_log_return(train_df)

    # Ensure consistent column order in the final dataframes
    final_exog_list = get_historical_exogenous_features(df_for_split)
    train_df = train_df[['unique_id', 'ds', 'y'] + final_exog_list]
    test_df = test_df[['unique_id', 'ds', 'y'] + final_exog_list]
    
    return train_df, test_df, original_df_for_reference, hist_exog_list


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
    
    train_df, test_df, original_df, hist_exog_list = load_and_process_data(
        data_path=data_path,
        horizon=horizon,
        test_length_multiplier=test_length_multiplier,
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
    # print("--- Running raw data preparation with transformations (default) ---")
    # train_df, test_df, _, _, _ = prepare_pipeline_data(
    #     data_path=RAW_DATA_PATH, 
    #     apply_transformations=True)
    # print("\nTransformed training data head:")
    # print(train_df.head())
    
    print("\n" + "="*50 + "\n")

    print("--- Running transformed data preparation ---")
    train_df_no_transform, _, _, _, _ = prepare_pipeline_data(
        data_path=DATA_PATH, 
        apply_transformations=True)
    print("\nTransformed training data head:")
    print(train_df_no_transform.head())


if __name__ == "__main__":
    main()
