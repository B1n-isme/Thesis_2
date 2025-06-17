import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFECV
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.api import add_constant
from pathlib import Path

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
                            drop_na: bool = True,
                            use_stationary_target: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Unified data preparation function with perfect alignment of features and target.
        
        Args:
            df: Input dataframe with time series data
            features: Specific features to use (if None, use all non-meta columns)
            horizon: Forecasting horizon (number of steps ahead)
            drop_na: Whether to drop NaN values after alignment
            use_stationary_target: Whether to use stationary target for analysis (if available)
            
        Returns:
            Tuple of (X, y, feature_names) arrays with perfect alignment
        """
        # Sort by date to ensure proper order
        df_sorted = df.sort_values('ds').copy()
        
        # Choose target column based on availability and preference
        target_col = 'y'  # Default to original target
        if use_stationary_target and 'target_stationary' in df_sorted.columns:
            target_col = 'target_stationary'
        
        # Get feature columns (exclude metadata and target only)
        if features is None:
            # All columns except metadata and the original 'y' are potential features
            base_features = [col for col in df_sorted.columns if col not in ['unique_id', 'ds', 'y']]
        else:
            base_features = features
        
        # --- FIX: Ensure the selected target is not also in the feature list ---
        # This prevents a downstream error when the target column is duplicated.
        feature_cols = [f for f in base_features if f != target_col]
        
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

    def shap_based_feature_selection(self, train_df: pd.DataFrame,
                                   method: str = 'xgboost',
                                   cv_folds: int = 5,
                                   shap_threshold_percentile: float = 25.0,
                                   n_estimators: int = 200) -> Dict[str, Any]:
        """
        SHAP-based feature selection with time-aware cross-validation.
        
        Args:
            train_df: Training dataframe
            method: Tree method ('xgboost', 'lightgbm', 'random_forest')
            cv_folds: Number of time-series CV folds
            shap_threshold_percentile: Percentile threshold for SHAP values
            n_estimators: Number of estimators for the model
            
        Returns:
            Dictionary with SHAP-selected features and analysis
        """
        self.print_info(f"Starting SHAP-based {method} feature selection...")
        
        # Prepare data using RAW target to avoid data leakage
        X, y, feature_names = self._prepare_aligned_data(
            train_df, horizon=HORIZON, use_stationary_target=True
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Store SHAP values across all CV folds
        all_shap_values = []
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.print_info(f"Processing CV fold {fold_idx + 1}/{cv_folds}")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train model
            if method == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=-1
                )
            elif method == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=-1, verbosity=-1
                )
            elif method == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            model.fit(X_train_cv, y_train_cv)
            
            # Calculate SHAP values
            try:
                if method in ['xgboost', 'lightgbm']:
                    explainer = shap.TreeExplainer(model)
                else:  # random_forest
                    explainer = shap.Explainer(model, X_train_cv)
                
                # Use a sample for SHAP calculation to speed up
                sample_size = min(500, len(X_val_cv))
                if sample_size > 0:
                    sample_indices = np.random.choice(len(X_val_cv), sample_size, replace=False)
                    X_sample = X_val_cv[sample_indices]
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Calculate mean absolute SHAP values for each feature
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    all_shap_values.append(mean_abs_shap)
                
                # Calculate CV score
                y_pred = model.predict(X_val_cv)
                cv_score = mean_absolute_error(y_val_cv, y_pred)
                cv_scores.append(cv_score)
                
            except Exception as e:
                self.print_info(f"SHAP calculation failed for fold {fold_idx + 1}: {e}")
                continue
        
        if not all_shap_values:
            raise ValueError("SHAP calculation failed for all folds")
        
        # Average SHAP values across all folds
        mean_shap_values = np.mean(all_shap_values, axis=0)
        std_shap_values = np.std(all_shap_values, axis=0)
        
        # Create SHAP importance dataframe
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_shap_values,
            'std_abs_shap': std_shap_values,
            'cv_stability': 1 - (std_shap_values / (mean_shap_values + 1e-8))  # Higher is more stable
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Select features above threshold
        threshold_value = np.percentile(mean_shap_values, shap_threshold_percentile)
        selected_features = shap_importance_df[
            shap_importance_df['mean_abs_shap'] >= threshold_value
        ]['feature'].tolist()
        
        self.print_info(f"SHAP selected {len(selected_features)} features "
                       f"(threshold: {threshold_value:.6f})")
        
        result = {
            'selected_features': selected_features,
            'shap_importance': shap_importance_df,
            'threshold_percentile': shap_threshold_percentile,
            'threshold_value': threshold_value,
            'cv_scores': {
                'cv_mae_mean': np.mean(cv_scores),
                'cv_mae_std': np.std(cv_scores)
            },
            'method': f'{method}_shap'
        }
        
        self.fitted_selectors[f'{method}_shap'] = model  # Store last model
        self.selected_features[f'{method}_shap'] = selected_features
        self.feature_rankings[f'{method}_shap'] = shap_importance_df
        
        return result
    
    def stability_selection(self, train_df: pd.DataFrame,
                          method: str = 'xgboost',
                          n_bootstrap: int = 50,
                          selection_threshold: float = 0.7,
                          sample_fraction: float = 0.8) -> Dict[str, Any]:
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
        
        # Prepare data using RAW target to avoid data leakage
        X, y, feature_names = self._prepare_aligned_data(
            train_df, horizon=HORIZON, use_stationary_target=True
        )
        
        # Track feature selection frequency
        feature_selections = defaultdict(int)
        bootstrap_results = []
        
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
                X_boot, y_boot, feature_cols = self._prepare_aligned_data(
                    bootstrap_df, horizon=HORIZON, use_stationary_target=True
                )
                
                if len(X_boot) == 0:
                    continue
                
                # Extract base method name (remove suffixes like '_shap', '_stability')
                base_method = method.split('_')[0]  # 'xgboost_shap' -> 'xgboost'
                
                # Train base model directly on bootstrap sample
                if base_method == 'xgboost':
                    model = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
                elif base_method == 'lightgbm':
                    model = lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbosity=-1)
                elif base_method == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
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
                    feature_name = feature_cols[idx]
                    feature_selections[feature_name] += 1
                    selected_features_this_iteration.append(feature_name)
                
                bootstrap_results.append({
                    'iteration': bootstrap_idx,
                    'n_features_selected': len(selected_features_this_iteration),
                    'selected_features': selected_features_this_iteration
                })
                
            except Exception as e:
                self.print_info(f"Bootstrap iteration {bootstrap_idx + 1} failed: {e}")
                continue
        
        # Calculate selection frequencies
        if len(bootstrap_results) == 0:
            self.print_info("No successful bootstrap iterations - falling back to all features")
            feature_names = [col for col in train_df.columns if col not in ['unique_id', 'ds', 'y']]
            selection_frequencies = {feature: 0.0 for feature in feature_names}
            stable_features = feature_names[:min(10, len(feature_names))]  # Return top 10 features as fallback
        else:
            selection_frequencies = {
                feature: count / len(bootstrap_results) 
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
            'bootstrap_results': bootstrap_results,
            'method': f'{method}_stability'
        }
        
        self.selected_features[f'{method}_stability'] = stable_features
        self.feature_rankings[f'{method}_stability'] = frequency_df
        
        return result
    
    def permutation_importance_validation(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                        selected_features: List[str],
                                        method: str = 'xgboost',
                                        n_repeats: int = 10) -> Dict[str, Any]:
        """
        Validate feature selection using permutation importance.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe  
            selected_features: Features to validate
            method: Method to use for validation
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary with permutation importance results
        """
        self.print_info("Validating feature selection with permutation importance...")
        
        # Prepare data with selected features using unified function
        X_train, y_train, feature_cols = self._prepare_aligned_data(
            train_df, features=selected_features, horizon=HORIZON
        )
        X_val, y_val, _ = self._prepare_aligned_data(
            val_df, features=selected_features, horizon=HORIZON
        )
        
        if len(X_train) == 0 or len(X_val) == 0:
            self.print_info("Not enough data for permutation importance validation.")
            return {
                'selected_features': [],
                'permutation_importance': pd.DataFrame(),
                'baseline_score': 0,
                'n_repeats': n_repeats,
                'method': f'{method}_permutation'
            }

        # Train model
        if method == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1
            )
        elif method == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1, verbosity=-1
            )
        elif method == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        model.fit(X_train, y_train)
        
        # Calculate baseline score
        baseline_score = model.score(X_val, y_val)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_val, y_val, 
            n_repeats=n_repeats, 
            random_state=self.random_state,
            scoring='neg_mean_absolute_error'
        )
        
        # Create permutation importance dataframe
        perm_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'importance_score': -perm_importance.importances_mean  # Convert to positive (lower MAE = higher importance)
        }).sort_values('importance_score', ascending=False)
        
        # Identify truly important features (positive importance)
        important_features = perm_importance_df[
            perm_importance_df['importance_score'] > 0
        ]['feature'].tolist()
        
        self.print_info(f"Permutation importance validated {len(important_features)} features "
                       f"out of {len(selected_features)}")
        
        result = {
            'selected_features': important_features,
            'permutation_importance': perm_importance_df,
            'baseline_score': baseline_score,
            'n_repeats': n_repeats,
            'method': f'{method}_permutation'
        }
        
        self.selected_features[f'{method}_permutation'] = important_features
        
        return result
    
    def handle_multicollinearity(self, df: pd.DataFrame, selected_features: List[str],
                               recommendation_df: pd.DataFrame,
                               corr_threshold: float = 0.9,
                               vif_threshold: float = 10.0) -> List[str]:
        """
        Removes multicollinearity from a feature set using clustering and VIF.
        """
        self.print_info(f"Handling multicollinearity for {len(selected_features)} features...")
        if len(selected_features) <= 1:
            return selected_features

        # 1. Feature Clustering
        corr_matrix = df[selected_features].corr(method='spearman').abs()
        dist_matrix = 1 - corr_matrix
        linkage_matrix = linkage(squareform(dist_matrix), method='average')
        
        # Clusters are formed based on distance, so 1 - corr_threshold
        clusters = fcluster(linkage_matrix, t=(1 - corr_threshold), criterion='distance')
        
        feature_cluster_map = pd.DataFrame({'feature': selected_features, 'cluster': clusters})
        
        # 2. Select Representatives
        final_features = []
        for cluster_id in feature_cluster_map['cluster'].unique():
            cluster_features = feature_cluster_map[
                feature_cluster_map['cluster'] == cluster_id
            ]['feature'].tolist()
            
            if len(cluster_features) == 1:
                final_features.append(cluster_features[0])
                continue
            
            # Select the best feature from the cluster based on combined_score
            cluster_scores = recommendation_df[recommendation_df['feature'].isin(cluster_features)]
            best_feature = cluster_scores.sort_values('combined_score', ascending=False).iloc[0]['feature']
            final_features.append(best_feature)
            
        self.print_info(f"Reduced to {len(final_features)} features after clustering.")
        
        if len(final_features) <= 1:
            return final_features

        # 3. Final VIF Check
        features_for_vif = df[final_features]
        
        # Iteratively remove features with high VIF
        while True:
            vif_data = add_constant(features_for_vif)
            vif_scores = pd.Series(
                [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])],
                index=vif_data.columns,
                dtype=float
            ).drop('const') # Drop VIF for the constant

            if vif_scores.max() > vif_threshold:
                feature_to_remove = vif_scores.idxmax()
                features_for_vif = features_for_vif.drop(columns=[feature_to_remove])
                self.print_info(f"Removed '{feature_to_remove}' due to high VIF ({vif_scores.max():.2f})")
                if len(features_for_vif.columns) <= 1:
                    break
            else:
                break
        
        final_features = features_for_vif.columns.tolist()
        self.print_info(f"Reduced to {len(final_features)} features after VIF check.")
        
        return final_features

    def tree_based_rfecv_selection(self, train_df: pd.DataFrame,
                                   method: str = 'xgboost',
                                   cv_folds: int = 5,
                                   step: int = 1,
                                   min_features_to_select: int = 10,
                                   **model_params) -> Dict[str, Any]:
        """
        Performs Recursive Feature Elimination with Cross-Validation (RFECV) using tree-based models.

        Args:
            train_df: Training dataframe.
            method: Tree-based model to use ('xgboost', 'lightgbm', 'random_forest').
            cv_folds: Number of time-series CV folds.
            step: Number of features to remove at each iteration.
            min_features_to_select: The minimum number of features to be selected.
            **model_params: Additional parameters for the tree model.

        Returns:
            Dictionary with selected features, optimal number, and rankings.
        """
        # supress warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        
        self.print_info(f"Starting RFECV feature selection with {method}...")

        # Prepare data using stationary target
        X, y, feature_names = self._prepare_aligned_data(
            train_df, horizon=HORIZON, use_stationary_target=True
        )

        # Initialize model
        if method == 'xgboost':
            base_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1, **model_params)
        elif method == 'lightgbm':
            base_model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbosity=-1, **model_params)
        elif method == 'random_forest':
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **model_params)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Initialize RFECV
        rfecv = RFECV(
            estimator=base_model,
            step=step,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            min_features_to_select=min_features_to_select,
            n_jobs=-1
        )
        
        self.print_info(f"Fitting RFECV... This may take a while.")
        X_df = pd.DataFrame(X, columns=feature_names)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names"
            )
            rfecv.fit(X_df, y)
        self.print_info(f"RFECV fitting completed.")

        # Get selected features
        selected_mask = rfecv.support_
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        ranking_df = pd.DataFrame({
            'feature': feature_names,
            'ranking': rfecv.ranking_
        }).sort_values('ranking', ascending=True)

        self.print_info(f"RFECV selected {rfecv.n_features_} optimal features for {method}.")

        result = {
            'selected_features': selected_features,
            'optimal_n_features': rfecv.n_features_,
            'feature_ranking': ranking_df,
            'cv_scores': rfecv.cv_results_,
            'method': f'{method}_rfecv'
        }
        
        self.selected_features[f'{method}_rfecv'] = selected_features
        self.feature_rankings[f'{method}_rfecv'] = ranking_df

        return result

    def _run_shap_step(self, train_df, tree_methods, shap_percentile, tree_n_estimators):
        """Helper to run SHAP selection for multiple tree methods."""
        shap_results = {}
        for method in tree_methods:
            res = self.shap_based_feature_selection(
                train_df, method=method, shap_threshold_percentile=shap_percentile,
                n_estimators=tree_n_estimators
            )
            shap_results[f'{method}_shap'] = res
        return shap_results

    def _run_autoencoder_step(self, train_df, val_df, ae_method, ae_top_k, ae_epochs):
        """Helper to run Autoencoder selection."""
        ae_results = {}
        res = self.autoencoder_selection_with_reconstruction_error(
            train_df, val_df, ae_method=ae_method, top_k_features=ae_top_k, epochs=ae_epochs
        )
        ae_results[f'{ae_method}_autoencoder'] = res
        return ae_results

    def _run_rfecv_step(self, train_df, tree_methods, tree_n_estimators):
        """Helper to run RFECV selection for multiple tree methods."""
        rfecv_results = {}
        for method in tree_methods:
            res = self.tree_based_rfecv_selection(
                train_df, method=method, n_estimators=tree_n_estimators
            )
            rfecv_results[f'{method}_rfecv'] = res
        return rfecv_results

    def _run_stability_step(self, train_df, tree_methods):
        """Helper to run Stability selection for multiple tree methods."""
        stability_results = {}
        for method in tree_methods:
            res = self.stability_selection(train_df, method=method)
            stability_results[f'{method}_stability'] = res
        return stability_results

    def robust_comprehensive_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                     tree_methods: List[str] = ['xgboost', 'lightgbm'],
                                     use_shap: bool = True,
                                     use_rfecv: bool = True,
                                     use_stability_selection: bool = True,
                                     use_autoencoder: bool = True,
                                     ae_method: str = 'lstm',
                                     use_permutation: bool = True,
                                     remove_collinearity: bool = True,
                                     shap_percentile: float = 75.0,
                                     ae_top_k: int = 40,
                                     n_repeats_permutation: int = 10,
                                     tree_n_estimators: int = 200,
                                     ae_epochs: int = 50,
                                     results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main orchestration method for robust feature selection.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            tree_methods (List[str]): Tree-based models to use.
            use_shap (bool): Whether to use SHAP selection.
            use_rfecv (bool): Whether to use RFECV.
            use_stability_selection (bool): Whether to use Stability Selection.
            use_autoencoder (bool): Whether to use Autoencoder selection.
            ae_method (str): Autoencoder architecture ('lstm' or 'transformer').
            use_permutation (bool): Whether to run permutation importance validation.
            remove_collinearity (bool): Whether to handle multicollinearity.
            shap_percentile (float): SHAP value percentile for feature selection.
            ae_top_k (int): Number of top features from Autoencoder to select.
            n_repeats_permutation (int): Number of repeats for permutation importance.
            tree_n_estimators (int): Number of estimators for tree models.
            ae_epochs (int): Number of epochs for Autoencoder training.
            results_dir (Optional[str]): Directory to save intermediate results.

        Returns:
            Dict[str, Any]: A dictionary containing selection results.
        """
        self.print_info("Starting robust comprehensive selection...")
        all_results = {}
        all_selected_features = defaultdict(list)

        # Step 1: Run primary selection methods in parallel if possible
        # For simplicity, running them sequentially here.
        if use_shap:
            shap_results = self._run_shap_step(train_df, tree_methods, shap_percentile, tree_n_estimators)
            all_results.update(shap_results)
            for method, res in shap_results.items():
                self._save_feature_list(res['selected_features'], f"1_{method}_features.txt", results_dir)


        if use_autoencoder:
            ae_results = self._run_autoencoder_step(train_df, val_df, ae_method, ae_top_k, ae_epochs)
            all_results.update(ae_results)
            for method, res in ae_results.items():
                self._save_feature_list(res['selected_features'], f"1_{method}_features.txt", results_dir)

        if use_rfecv:
            rfecv_results = self._run_rfecv_step(train_df, tree_methods, tree_n_estimators)
            all_results.update(rfecv_results)
            for method, res in rfecv_results.items():
                self._save_feature_list(res['selected_features'], f"1_{method}_features.txt", results_dir)

        if use_stability_selection:
            stability_results = self._run_stability_step(train_df, tree_methods)
            all_results.update(stability_results)
            for method, res in stability_results.items():
                self._save_feature_list(res['selected_features'], f"1_{method}_features.txt", results_dir)

        # Step 2: Generate recommendations from the primary selections
        recommendations = self._generate_robust_recommendations(all_results)
        consensus_features = recommendations.get('consensus_features', [])
        self.print_info(f"Generated {len(consensus_features)} consensus features.")
        self._save_feature_list(consensus_features, "2_consensus_features.txt", results_dir)
        
        # Keep track of consensus results
        self.step_results['consensus'] = {
            'features': consensus_features,
            'details': recommendations.get('feature_counts')
        }
        
        features_after_consensus = consensus_features

        # Step 3: Handle multicollinearity
        if remove_collinearity and features_after_consensus:
            features_after_collinearity = self.handle_multicollinearity(
                train_df, features_after_consensus, recommendations['feature_counts']
            )
            self.print_info(f"Reduced to {len(features_after_collinearity)} features after handling multicollinearity.")
            self._save_feature_list(features_after_collinearity, "3_after_multicollinearity_features.txt", results_dir)
        else:
            features_after_collinearity = features_after_consensus
            
        self.step_results['collinearity'] = {'features': features_after_collinearity}

        # Step 4: Permutation Importance Validation
        if use_permutation and features_after_collinearity:
            pfi_results = self.permutation_importance_validation(
                train_df, val_df, features_after_collinearity, n_repeats=n_repeats_permutation
            )
            final_features = pfi_results['selected_features']
            self.print_info(f"Validated to {len(final_features)} features with PFI.")
            self._save_feature_list(final_features, "4_final_pfi_features.txt", results_dir)
            self.step_results['pfi'] = {
                'features': final_features,
                'importance': pfi_results['pfi_importance']
            }
        else:
            final_features = features_after_collinearity
            self.print_info("Skipping permutation importance, using features from previous step.")
            self._save_feature_list(final_features, "4_final_features_no_pfi.txt", results_dir)

        self.print_info("Robust comprehensive selection finished.")

        return {
            'initial_selections': all_results,
            'final_recommendations': {
                'consensus_features': final_features,
                'feature_counts': recommendations.get('feature_counts')
            },
            'step_by_step_results': self.step_results
        }

    def _generate_robust_recommendations(self, all_results: Dict[str, Dict], 
                                       min_votes: int = 2) -> Dict[str, Any]:
        """
        Generates feature recommendations based on a weighted consensus from multiple methods.

        Args:
            all_results: Dict containing the detailed results from each selection step.
            min_votes: The minimum number of methods that must select a feature.

        Returns:
            Dictionary with the consensus features and a DataFrame of votes and scores.
        """
        self.print_info(f"Aggregating results with weighted scoring...")
        feature_scores = defaultdict(float)
        feature_votes = Counter()

        # Helper for min-max normalization
        def normalize_series(s: pd.Series, ascending: bool = True) -> pd.Series:
            if s.max() == s.min(): return pd.Series(1.0, index=s.index)
            if ascending: return (s - s.min()) / (s.max() - s.min())
            return 1 - ((s - s.min()) / (s.max() - s.min()))

        # Process SHAP results
        if 'shap' in all_results:
            for result in all_results['shap'].values():
                df = result.get('shap_importance')
                if df is not None and not df.empty:
                    df['norm_score'] = normalize_series(df['mean_abs_shap'], ascending=True)
                    selected = result.get('selected_features', [])
                    feature_votes.update(selected)
                    for _, row in df.iterrows():
                        if row['feature'] in selected:
                            feature_scores[row['feature']] += row['norm_score']

        # Process RFECV results
        if 'rfecv' in all_results:
            for result in all_results['rfecv'].values():
                df = result.get('feature_ranking')
                if df is not None and not df.empty:
                    df['norm_score'] = normalize_series(df['ranking'], ascending=False) # Lower rank is better
                    selected = result.get('selected_features', [])
                    feature_votes.update(selected)
                    for _, row in df.iterrows():
                        if row['feature'] in selected:
                            feature_scores[row['feature']] += row['norm_score']

        # Process Autoencoder results
        if 'autoencoder' in all_results:
            result = all_results['autoencoder']
            df = result.get('feature_ranking')
            if df is not None and not df.empty:
                # Lower error is better
                df['norm_score'] = normalize_series(df['reconstruction_error'], ascending=False)
                selected = result.get('selected_features', [])
                feature_votes.update(selected)
                for _, row in df.iterrows():
                    if row['feature'] in selected:
                        feature_scores[row['feature']] += row['norm_score']
        
        # Process Stability Selection results
        if 'stability' in all_results:
            for result in all_results['stability'].values():
                df = result.get('selection_frequency')
                if df is not None and not df.empty:
                    df['norm_score'] = df['frequency'] # Already a score from 0-1
                    selected = result.get('selected_features', [])
                    feature_votes.update(selected)
                    for _, row in df.iterrows():
                        if row['feature'] in selected:
                            feature_scores[row['feature']] += row['norm_score']

        if not feature_votes:
            return {'consensus_features': [], 'feature_counts': pd.DataFrame()}

        # Create a detailed recommendation DataFrame
        recco_df = pd.DataFrame(
            feature_votes.items(), columns=['feature', 'votes']
        )
        recco_df['combined_score'] = recco_df['feature'].map(feature_scores)
        recco_df = recco_df.sort_values(
            by=['votes', 'combined_score'], ascending=[False, False]
        ).reset_index(drop=True)

        # Select features that meet the minimum vote threshold
        consensus_features = recco_df[
            recco_df['votes'] >= min_votes
        ]['feature'].tolist()
        
        self.print_info(f"Found {len(consensus_features)} features with at least {min_votes} votes.")
        
        return {
            'consensus_features': consensus_features,
            'feature_counts': recco_df
        }

    def run_robust_auto_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run robust automatic feature selection implementing Step 4 strategy.
        
        Args:
            train_df: Training dataframe (preprocessed from steps 1-2)
            val_df: Validation dataframe
            
        Returns:
            Comprehensive robust selection results
        """
        self.print_info("Running robust automatic feature selection pipeline (Step 4)...")
        
        return self.robust_comprehensive_selection(
            train_df=train_df,
            val_df=val_df,
            tree_methods=['xgboost', 'lightgbm', 'random_forest'],
            use_shap=True,
            use_autoencoder=True,
            ae_method='lstm',
            use_permutation=True,
            remove_collinearity=True,
            shap_percentile=75.0,
            ae_top_k=40,
            n_repeats_permutation=10
        ) 