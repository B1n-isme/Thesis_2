import optuna


# def lgb_auto_cfg(trial: optuna.Trial) -> dict:
#     """Generate LightGBM-specific hyperparameter configuration using native parameter names only."""
#     return {
#         # "bagging_freq": 1,
#         "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-4, 0.3, log=True),
#         "verbosity": -1,  # Silent mode
#         # "n_estimators": trial.suggest_int("n_estimators", 100, 2500, log=True),
#         # "lambda_l1": trial.suggest_float("lgb_lambda_l1", 1e-8, 20.0, log=True),
#         # "lambda_l2": trial.suggest_float("lgb_lambda_l2", 1e-8, 20.0, log=True),
#         "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 256, log=True),
#         # "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.4, 1.0),
#         # "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.4, 1.0),
#         "objective": trial.suggest_categorical("objective", ["l1", "l2"]),
#     }

def lgb_auto_cfg(trial: optuna.Trial) -> dict:
    """
    Refined LightGBM hyperparameter configuration for volatile time series.
    """
    return {
        "objective": trial.suggest_categorical("objective", ["l1", "l2", "huber"]),
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 300, log=True),
        "lambda_l1": trial.suggest_float("lgb_lambda_l1", 1e-8, 15.0, log=True),
        "lambda_l2": trial.suggest_float("lgb_lambda_l2", 1e-8, 15.0, log=True),
        "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("lgb_bagging_freq", 1, 7),
        "verbosity": -1,  # Silent mode
    }

def xgb_auto_cfg(trial: optuna.Trial) -> dict:
    """
    Refined XGBoost hyperparameter configuration for volatile time series.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 14),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 70.0, log=True),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 70.0, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 25),
        "enable_categorical": True,
    }

def cat_auto_cfg(trial: optuna.Trial) -> dict:
    """
    Refined CatBoost hyperparameter configuration for volatile time series.
    """
    return {
        "cat_features": ["unique_id"],
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "depth": trial.suggest_int("cat_depth", 4, 12),
        "learning_rate": trial.suggest_float("cat_learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("cat_subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("cat_colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("cat_min_data_in_leaf", 5, 100),
        "silent": True,
    }