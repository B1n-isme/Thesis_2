"""
Data preparation module for neural forecasting pipeline.
"""
import pandas as pd
from typing import Tuple, List, Dict
from config.base import (
    DATA_PATH, DATE_COLUMN, TARGET_COLUMN, TARGET_RENAMED, 
    DATE_RENAMED, UNIQUE_ID_VALUE, HORIZON, TEST_LENGTH_MULTIPLIER
)
from src.utils.utils import get_historical_exogenous_features, print_data_info


def load_and_prepare_data():
    """Load and prepare the dataset for forecasting."""
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_parquet(DATA_PATH)
    
    # Rename columns
    df = df.rename(columns={DATE_COLUMN: DATE_RENAMED, TARGET_COLUMN: TARGET_RENAMED})
    
    # Add unique_id and convert date
    df['unique_id'] = UNIQUE_ID_VALUE
    df[DATE_RENAMED] = pd.to_datetime(df[DATE_RENAMED])
    
    df.reset_index(drop=True, inplace=True)
    
    return df


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


def prepare_data(horizon, test_length_multiplier):
    """Complete data preparation pipeline.
    
    Args:
        horizon (int): Forecast horizon in days
        test_length_multiplier (int): Multiplier to determine test set length
    """
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get historical exogenous features
    hist_exog_list = get_historical_exogenous_features(df)
    
    # Split data
    train_df, test_df = split_data(df, horizon, test_length_multiplier)
    
    # Move hist_exog_list to end of df
    df = df[['unique_id', 'ds', 'y'] + hist_exog_list]
    train_df = train_df[['unique_id', 'ds', 'y'] + hist_exog_list]
    test_df = test_df[['unique_id', 'ds', 'y'] + hist_exog_list]
    
    return train_df, test_df, hist_exog_list


def prepare_pipeline_data(horizon: int = HORIZON, test_length_multiplier: int = TEST_LENGTH_MULTIPLIER) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Enhanced data preparation for the pipeline with detailed logging and metadata.
    
    Args:
        horizon (int, optional): Forecast horizon in days. Defaults to HORIZON.
        test_length_multiplier (int, optional): Multiplier for test set length. Defaults to TEST_LENGTH_MULTIPLIER.
    
    Returns:
        Tuple of (train_df, test_df, hist_exog_list, data_info_dict)
    """
    print("\n📊 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    train_df, test_df, hist_exog_list = prepare_data(
        horizon=horizon,
        test_length_multiplier=test_length_multiplier
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
    
    return train_df, test_df, hist_exog_list, data_info


if __name__ == "__main__":
    # Example usage with sample values
    df_dev, df_test, hist_exog = prepare_data(horizon=7, test_length_multiplier=2)
    print(f"\nHistorical exogenous features: {len(hist_exog)} features")
    print(f"Sample features: {hist_exog[:5] if len(hist_exog) >= 5 else hist_exog}") 
    print(f"df_dev.head(): {df_dev.head()}")
    print(f"df_test.head(): {df_test.head()}")
