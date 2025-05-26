from typing import List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

def get_mlforecast_models(self, model_names: List[str] = None) -> List[Tuple[str, Any]]:
        """
        Get MLForecast models with different algorithms.
        
        Parameters:
        -----------
        model_names : List[str], optional
            Specific model names to include. If None, returns all models.
            
        Returns:
        --------
        List[Tuple[str, Any]]: List of (model_name, sklearn_model) tuples
        """
        n_estimators = 200
        max_depth = 15
        
        all_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=self.config.n_jobs
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=self.config.n_jobs
            ),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'LinearRegression': LinearRegression(n_jobs=self.config.n_jobs),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        # Add XGBoost
        all_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=self.config.n_jobs,
            verbosity=0
        )
        
        
        if model_names is None:
            return [(name, model) for name, model in all_models.items()]
        
        return [(name, all_models[name]) for name in model_names if name in all_models]