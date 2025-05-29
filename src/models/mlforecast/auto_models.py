import lightgbm as lgb
import optuna
import pandas as pd
from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from utilsforecast.plotting import plot_series

from mlforecast import MLForecast
from mlforecast.auto import (
    AutoLightGBM,
    AutoMLForecast,
    AutoModel,
    AutoRidge,
    ridge_space,
)
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean

optuna.logging.set_verbosity(optuna.logging.ERROR)
auto_mlf = AutoMLForecast(
    models={'lgb': AutoLightGBM(), 'ridge': AutoRidge()},
    freq=1,
    season_length=24,
)
auto_mlf.fit(
    train,
    n_windows=2,
    h=horizon,
    num_samples=2,  # number of trials to run
)

preds = auto_mlf.predict(horizon)
preds.head()