"""
Base module for the Feature Selection Pipeline.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import xgboost as xgb
import lightgbm as lgb
import shap
from collections import defaultdict, Counter

# Local imports from the new package structure
from .robust_selection import RobustSelectionMixin

# Local imports from the project
from config.base import HORIZON, TEST_LENGTH_MULTIPLIER, SEED
from src.dataset.data_preparation import prepare_pipeline_data
from src.utils.utils import seed_everything

warnings.filterwarnings('ignore')

class FeatureSelector(
    RobustSelectionMixin,
):
    """
    Advanced Feature Selection Pipeline for Time Series Forecasting.
    
    This class integrates a comprehensive suite of feature selection techniques
    through a modular, mixin-based architecture.
    """
    
    def __init__(self, random_state: int = SEED, verbose: bool = True, use_gpu: bool = False):
        """
        Initialize the FeatureSelector.
        
        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            use_gpu: Whether to attempt to use GPU for acceleration
        """
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.fitted_selectors = {}
        self.feature_rankings = {}
        self.selected_features = {}
        self.autoencoder_features = {}
        self.step_results = {}

        self.is_fitted = False
        self.fitted_parameters = {}
        
        seed_everything(random_state)
        
    def print_info(self, message: str):
        """Print information if verbose is enabled."""
        if self.verbose:
            print(f"[FeatureSelector] {message}")

    def run_complete_feature_selection_strategy(self, df: pd.DataFrame,
                                               target_col: str = 'y',
                                               step3_params: Optional[Dict] = None,
                                               results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates the robust feature selection strategy (formerly Step 3).
        This is the main entry point for the feature selection process.
        """
        self.print_info("Starting robust feature selection strategy...")

        # Data splitting for robust evaluation
        # Ensure 'ds' is sorted if not already
        df_sorted = df.sort_values('ds').reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.9)
        train_df = df_sorted.iloc[:split_idx]
        val_df = df_sorted.iloc[split_idx:]

        self.print_info(f"Data split into training ({len(train_df)} rows) and validation ({len(val_df)} rows).")

        # The robust_comprehensive_selection function now handles all steps and
        # returns the final results in the correct format.
        selection_results = self.robust_comprehensive_selection(
            train_df, val_df, results_dir=results_dir, use_gpu=self.use_gpu, **step3_params
        )
        
        self.print_info("Complete feature selection strategy finished.")
        return selection_results

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Returns a summary of feature selection results.
        """
        summary_data = []
        for method, results in self.feature_rankings.items():
            for feature, importance in results.items():
                summary_data.append({
                    "method": method,
                    "feature": feature,
                    "importance": importance
                })
        return pd.DataFrame(summary_data)

    def save_models(self, save_dir: str):
        """
        Saves fitted selectors and autoencoder models.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.fitted_selectors, save_path / "fitted_selectors.pkl")
        self.print_info(f"Models saved to {save_dir}")

    def get_optimal_features(self, method: str = 'auto') -> List[str]:
        """
        Retrieves the list of selected features for a given method.
        """
        if method == 'auto':
            # Use a consensus or default method if available
            method = 'robust_comprehensive' 
        return self.selected_features.get(method, [])

    def apply_optimal_selection(self, df: pd.DataFrame, method: str = 'auto') -> pd.DataFrame:
        """
        Applies the optimal feature set to a dataframe.
        """
        features_to_keep = self.get_optimal_features(method)
        
        meta_cols = [col for col in ['unique_id', 'ds', 'y'] if col in df.columns]
        
        final_cols = meta_cols + [f for f in features_to_keep if f in df.columns]
        
        return df[final_cols] 