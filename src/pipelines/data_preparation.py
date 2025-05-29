"""
Module for data preparation in the Auto Models workflow.
"""
import pandas as pd
from typing import Tuple, List, Dict

# Configuration and local imports
from config.base import HORIZON, TEST_LENGTH_MULTIPLIER
from src.dataset.data_preparation import prepare_data as util_prepare_data # Renamed

def prepare_pipeline_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Step 1: Data Preparation for the pipeline.
    
    Returns:
        Tuple of (train_df, test_df, hist_exog_list, data_info_dict)
    """
    print("\n📊 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    train_df, test_df, hist_exog_list = util_prepare_data(
        horizon=HORIZON,
        test_length_multiplier=TEST_LENGTH_MULTIPLIER
    )
    
    data_info = {
        'total_samples': len(train_df) + len(test_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': hist_exog_list,
        'horizon': HORIZON
    }
    
    print(f"✅ Data prepared successfully")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Test samples: {len(test_df):,}")
    print(f"   • Features: {len(hist_exog_list)} exogenous variables")
    print(f"   • Forecast horizon: {HORIZON} days")
    
    return train_df, test_df, hist_exog_list, data_info 