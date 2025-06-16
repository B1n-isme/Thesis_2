import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import optuna
import pandas as pd
from typing import List, Dict, Any, Optional
from mlforecast.auto import AutoModel
from src.models.mlforecast.auto_cfg import lgb_auto_cfg, xgb_auto_cfg, cat_auto_cfg
from src.dataset.data_preparation import prepare_pipeline_data
import numpy as np

optuna.logging.set_verbosity(optuna.logging.ERROR)


def create_lgb_config(trial: optuna.Trial) -> dict:
    """Create LightGBM configuration using comprehensive hyperparameters."""
    return lgb_auto_cfg(trial)


def create_xgb_config(trial: optuna.Trial) -> dict:
    """Create XGBoost configuration using comprehensive hyperparameters."""
    return xgb_auto_cfg(trial)


def create_cat_config(trial: optuna.Trial) -> dict:
    """Create CatBoost configuration using comprehensive hyperparameters."""
    return cat_auto_cfg(trial)


def get_ml_models() -> Dict[str, Any]:
    """Returns a dictionary of ML models for AutoMLForecast."""
    return {
        'AutoXGBoost': AutoModel(model=xgb.XGBRegressor(), config=create_xgb_config),
        'AutoCatBoost': AutoModel(model=cat.CatBoostRegressor(), config=create_cat_config),
        'AutoLightGBM': AutoModel(model=lgb.LGBMRegressor(), config=create_lgb_config),
    }
