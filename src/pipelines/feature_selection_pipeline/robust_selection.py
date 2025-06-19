import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.api import add_constant
from pathlib import Path
import torch
import warnings

from config.base import HORIZON

class RobustSelectionMixin:
    """Mixin for robust feature selection methods."""

    def _save_feature_list(self, features: List[str], filename: str, results_dir: Optional[str] = None):
        """Helper to save a list of features to a file."""
        if results_dir:
            output_path = Path(results_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"# Total features: {len(features)}\n")
                for feature in sorted(features):
                    f.write(f"{feature}\n")
            self.print_info(f"Saved feature list to {output_path}")

    def _prepare_aligned_data(self, df: pd.DataFrame, 
                            features: Optional[List[str]] = None,
                            horizon: int = 1,
                            drop_na: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Unified data preparation function with perfect alignment of features and target.
        This version uses the raw 'y' column as the target.
        """
        # Sort by date to ensure proper order
        df_sorted = df.sort_values('ds').copy()
        
        target_col = 'y'
        
        # Get feature columns
        if features is None:
            # All columns except metadata and the target are potential features
            feature_cols = [col for col in df_sorted.columns if col not in ['unique_id', 'ds', 'y']]
        else:
            feature_cols = features
        
        # Create working dataframe with only needed columns
        work_df = df_sorted[[target_col] + feature_cols].copy()
        
        # Create shifted target
        work_df['y_shifted'] = work_df[target_col].shift(-horizon)
        
        if drop_na:
            # Drop rows with any NaN values for perfect alignment
            work_df = work_df.dropna()
        
        # Extract aligned data
        X = work_df[feature_cols].values
        y = work_df['y_shifted'].values
        
        return X, y, feature_cols

    def stability_selection(self, train_df: pd.DataFrame,
                          method: str = 'xgboost',
                          n_bootstrap: int = 50,
                          selection_threshold: float = 0.7,
                          sample_fraction: float = 0.8,
                          use_gpu: bool = False,
                          **model_params) -> Dict[str, Any]:
        """
        Stability selection using multiple bootstrap samples.
        
        Args:
            train_df: Training dataframe
            method: Method to use for selection
            n_bootstrap: Number of bootstrap iterations
            selection_threshold: Minimum frequency for feature selection
            sample_fraction: Fraction of data to sample in each bootstrap
            
        Returns:
            Dictionary with stable features and selection frequencies
        """
        self.print_info(f"Starting stability selection with {n_bootstrap} bootstrap samples...")
        
        gpu_available = use_gpu and torch.cuda.is_available()
        if use_gpu and not gpu_available:
            self.print_info("GPU requested but not available. Falling back to CPU.")
        
        X, y, feature_names = self._prepare_aligned_data(
            train_df, horizon=HORIZON
        )
        
        # Track feature selection frequency
        feature_selections = defaultdict(int)
        
        for bootstrap_idx in range(n_bootstrap):
            if (bootstrap_idx + 1) % 10 == 0:
                self.print_info(f"Bootstrap iteration {bootstrap_idx + 1}/{n_bootstrap}")
            
            # Bootstrap sampling (maintaining temporal order)
            n_samples = int(len(X) * sample_fraction)
            # For time series, we take a contiguous chunk rather than random sampling
            if len(X) - n_samples + 1 <= 0:
                continue
            start_idx = np.random.randint(0, len(X) - n_samples + 1)
            end_idx = start_idx + n_samples
            
            try:
                # Direct model training on bootstrap sample using aligned data preparation
                bootstrap_df = train_df.iloc[start_idx:end_idx].copy()
                X_boot, y_boot, feature_cols_boot = self._prepare_aligned_data(
                    bootstrap_df, features=feature_names, horizon=HORIZON
                )
                
                if X_boot.shape[0] == 0:
                    continue
                
                # Extract base method name (remove suffixes like '_shap', '_stability')
                base_method = method.split('_')[0]  # 'xgboost_shap' -> 'xgboost'
                
                params = model_params.copy()
                if method == 'random_forest':
                    base_method = 'random_forest'

                if base_method == 'xgboost':
                    if gpu_available:
                        params['device'] = 'cuda'
                        params['tree_method'] = 'hist'
                    model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1, **params)
                elif base_method == 'lightgbm':
                    if gpu_available:
                        params['device'] = 'gpu'
                    model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbosity=-1, **params)
                elif base_method == 'random_forest':
                    model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **params)
                else:
                    raise ValueError(f"Unknown base method: {base_method} (from method: {method})")
                
                model.fit(X_boot, y_boot)
                
                # Get feature importances directly
                importances = model.feature_importances_
                
                # Select top features (e.g., top 50% by importance)
                importance_threshold = np.percentile(importances, 50)
                selected_indices = np.where(importances >= importance_threshold)[0]
                
                # Count selections and build selected features list
                selected_features_this_iteration = []
                for idx in selected_indices:
                    feature_name = feature_cols_boot[idx]
                    feature_selections[feature_name] += 1
                    selected_features_this_iteration.append(feature_name)
                
            except Exception as e:
                self.print_info(f"Bootstrap iteration {bootstrap_idx + 1} failed: {e}")
                continue
        
        # Calculate selection frequencies
        if len(feature_selections) == 0:
            self.print_info("No successful bootstrap iterations - falling back to all features")
            feature_names = [col for col in train_df.columns if col not in ['unique_id', 'ds', 'y']]
            selection_frequencies = {feature: 0.0 for feature in feature_names}
            stable_features = feature_names[:min(10, len(feature_names))]  # Return top 10 features as fallback
        else:
            selection_frequencies = {
                feature: count / n_bootstrap 
                for feature, count in feature_selections.items()
            }
            
            # Select stable features
            stable_features = [
                feature for feature, freq in selection_frequencies.items() 
                if freq >= selection_threshold
            ]
        
        self.print_info(f"Stability selection found {len(stable_features)} features "
                       f"with frequency >= {selection_threshold}")
        
        frequency_df = pd.DataFrame(
            selection_frequencies.items(), columns=['feature', 'frequency']
        ).sort_values('frequency', ascending=False)

        result = {
            'selected_features': stable_features,
            'selection_frequency': frequency_df,
            'method': f'{method}_stability'
        }
        
        self.selected_features[f'{method}_stability'] = stable_features
        self.feature_rankings[f'{method}_stability'] = frequency_df
        
        return result
    
    def handle_multicollinearity(self,
                               features_to_check: List[str],
                               stability_scores: pd.DataFrame,
                               corr_threshold: float = 0.9,
                               vif_threshold: float = 10.0,
                               data_for_vif: pd.DataFrame = None) -> List[str]:
        """
        Handles multicollinearity using a two-stage approach:
        1. High Correlation Clustering: Removes features from highly correlated pairs.
        2. VIF Check: Iteratively removes features with high VIF.
        
        The decision of which feature to remove is based on stability scores.
        """
        self.print_info("Starting multicollinearity reduction...")
        
        # Ensure stability_scores is indexed by feature for easy lookup
        if 'feature' in stability_scores.columns:
            stability_scores = stability_scores.set_index('feature')
            
        # --- Stage 1: Pairwise Correlation ---
        self.print_info(f"Filtering pairs with correlation > {corr_threshold}")
        
        corr_matrix = data_for_vif[features_to_check].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for column in upper.columns:
            # Find correlated features
            correlated_features = upper.index[upper[column] > corr_threshold].tolist()
            if not correlated_features:
                continue

            for feature2 in correlated_features:
                # If one of the pair is already marked for dropping, skip
                if column in to_drop or feature2 in to_drop:
                    continue

                # Compare stability scores to decide which to keep
                score1 = stability_scores.loc[column, 'frequency_stability_avg']
                score2 = stability_scores.loc[feature2, 'frequency_stability_avg']
                
                if score1 >= score2:
                    to_drop.add(feature2)
                    self.print_info(f"  - Dropping '{feature2}' (corr with '{column}'): score {score2:.2f} <= {score1:.2f}")
                else:
                    to_drop.add(column)
                    self.print_info(f"  - Dropping '{column}' (corr with '{feature2}'): score {score1:.2f} < {score2:.2f}")

        features_after_corr = [f for f in features_to_check if f not in to_drop]
        self.print_info(f"Features after correlation filtering: {len(features_after_corr)}")

        # --- Stage 2: VIF Calculation ---
        self.print_info(f"Iteratively removing features with VIF > {vif_threshold}")
        
        features_for_vif = features_after_corr.copy()
        
        while True:
            if not features_for_vif:
                break
                
            X_vif = add_constant(data_for_vif[features_for_vif].dropna())
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_vif.columns
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            
            # Exclude 'const' from VIF check
            vif_data = vif_data[vif_data['feature'] != 'const']
            
            max_vif = vif_data['VIF'].max()
            
            if max_vif > vif_threshold:
                feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
                features_for_vif.remove(feature_to_drop)
                self.print_info(f"  - Dropping '{feature_to_drop}' (VIF: {max_vif:.2f})")
            else:
                break
        
        self.print_info(f"Features after VIF filtering: {len(features_for_vif)}")
        
        return features_for_vif
    
    def robust_comprehensive_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                     tree_methods: List[str] = ['xgboost', 'lightgbm', 'random_forest'],
                                     min_consensus_level: int = 1,
                                     handle_multicollinearity_flag: bool = False,
                                     results_dir: Optional[str] = None,
                                     use_gpu: bool = False,
                                     **kwargs) -> Dict[str, Any]:
        """
        Orchestrates a simplified feature selection pipeline.
        
        Workflow:
        1.  Run Stability Selection with multiple tree-based models on the raw 'y' target.
        2.  Create a consensus list of features based on how many models selected them.
        3.  Optionally, perform multicollinearity reduction.
        """
        self.print_info("--- Starting Simplified Comprehensive Selection ---")
        
        # --- Step 1: Stability Selection ---
        stability_results = {}
        for method in tree_methods:
            params = kwargs.copy()
            # The 'stability_selection' method handles its own kwargs
            result = self.stability_selection(train_df, method=method, use_gpu=use_gpu, **params)
            stability_results[method] = result
            self._save_feature_list(result['selected_features'], f"1_{method}_stability_features_{HORIZON}.txt", results_dir)

        # --- Step 2: Create Consensus ---
        self.print_info("\n--- Step 2: Aggregating Stability Results & Building Consensus ---")
        
        # Get the list of stable features from each model
        stable_features_per_model = [res['selected_features'] for res in stability_results.values()]

        # Count how many models selected each feature as stable
        feature_counts = Counter()
        for feature_list in stable_features_per_model:
            feature_counts.update(feature_list)

        # Select features that meet the consensus level
        initial_consensus_features = [
            feature for feature, count in feature_counts.items()
            if count >= min_consensus_level
        ]
        self.print_info(f"Found {len(initial_consensus_features)} consensus features selected by at least {min_consensus_level} model(s).")
        self._save_feature_list(initial_consensus_features, f"1_initial_stable_features_{HORIZON}.txt", results_dir)
        
        final_features = initial_consensus_features
        
        # --- Step 3: Multicollinearity Reduction (Optional) ---
        if handle_multicollinearity_flag:
            self.print_info("\n--- Step 3: Handling Multicollinearity ---")
            
            # Combine all frequency dataframes for tie-breaking
            all_freq_dfs = [res['selection_frequency'].rename(columns={'frequency': f'frequency_{method}'}) 
                            for method, res in stability_results.items()]
            merged_freq_df = all_freq_dfs[0]
            for df in all_freq_dfs[1:]:
                merged_freq_df = pd.merge(merged_freq_df, df, on='feature', how='outer')
            merged_freq_df = merged_freq_df.fillna(0)
            freq_cols = [f'frequency_{method}' for method in tree_methods]
            merged_freq_df['frequency_stability_avg'] = merged_freq_df[freq_cols].mean(axis=1)

            # Prepare data for VIF (use the training set part)
            _, _, feature_names = self._prepare_aligned_data(train_df, features=initial_consensus_features)
            data_for_vif = train_df[feature_names].copy()
            
            features_after_multicollinearity = self.handle_multicollinearity(
                features_to_check=initial_consensus_features,
                stability_scores=merged_freq_df.set_index('feature'),
                data_for_vif=data_for_vif,
                corr_threshold=kwargs.get('corr_threshold', 0.9),
                vif_threshold=kwargs.get('vif_threshold', 10.0)
            )
            self._save_feature_list(features_after_multicollinearity, f"2_after_multicollinearity_features_{HORIZON}.txt", results_dir)
            final_features = features_after_multicollinearity
        else:
            self.print_info("\n--- Step 3: Skipping Multicollinearity Reduction ---")
        
        # --- Step 4: Final Recommendations and Reporting ---
        self.print_info("\n--- Finalizing Results ---")
        
        # Create a final recommendation DataFrame
        final_recommendation_df = pd.DataFrame({
            'feature': final_features
        })

        # Build final results package
        results_package = {
            'initial_stability_results': stability_results,
            'final_recommendations': {
                'consensus_features': final_features,
                'feature_counts': final_recommendation_df
            }
        }
        
        return results_package

    def run_robust_auto_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fully automated robust selection with sensible defaults.
        """
        return self.robust_comprehensive_selection(
            train_df=train_df,
            val_df=val_df,
            tree_methods=['xgboost', 'lightgbm', 'random_forest'],
            min_consensus_level=1,
            handle_multicollinearity_flag=True,
            use_gpu=True
        ) 