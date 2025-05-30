import pandas as pd# Load the training target dataset from the provided URL

Y_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet')

# Rename columns to match the Nixtlaverse's expectations
# The 'item_id' becomes 'unique_id' representing the unique identifier of the time series
# The 'timestamp' becomes 'ds' representing the time stamp of the data points
# The 'demand' becomes 'y' representing the target variable we want to forecast
Y_df = Y_df.rename(
    columns={
        'item_id': 'unique_id', 
        'timestamp': 'ds', 
        'demand': 'y'
    }
)

# Convert the 'ds' column to datetime format to ensure proper handling of date-related operations in subsequent steps
Y_df['ds'] = pd.to_datetime(Y_df['ds'])

Y_df = Y_df.query('unique_id.str.startswith("FOODS_3")').reset_index(drop=True)
Y_df['unique_id'] = Y_df['unique_id'].astype(str)

from utilsforecast.plotting import plot_series

# Feature: plot random series for EDA
plot_series(Y_df)

from statsforecast import StatsForecast
# Import necessary models from the statsforecast library
from statsforecast.models import (
    # SeasonalNaive: A model that uses the previous season's data as the forecast
    SeasonalNaive,
    # Naive: A simple model that uses the last observed value as the forecast
    Naive,
    # HistoricAverage: This model uses the average of all historical data as the forecast
    HistoricAverage,
    # CrostonOptimized: A model specifically designed for intermittent demand forecasting
    CrostonOptimized,
    # ADIDA: Adaptive combination of Intermittent Demand Approaches, a model designed for intermittent demand
    ADIDA,
    # IMAPA: Intermittent Multiplicative AutoRegressive Average, a model for intermittent series that incorporates autocorrelation
    IMAPA,
    # AutoETS: Automated Exponential Smoothing model that automatically selects the best Exponential Smoothing model based on AIC
    AutoETS
)

horizon = 28
models = [
    SeasonalNaive(season_length=7),
    Naive(),
    HistoricAverage(),
    CrostonOptimized(),
    ADIDA(),
    IMAPA(),
    AutoETS(season_length=7)
]

# Instantiate the StatsForecast class
sf = StatsForecast(
    models=models,  # A list of models to be used for forecasting
    freq='D',  # The frequency of the time series data (in this case, 'D' stands for daily frequency)
    n_jobs=-1,  # The number of CPU cores to use for parallel execution (-1 means use all available cores)
    verbose=True,  # Show progress
)

from time import time

# Get the current time before forecasting starts, this will be used to measure the execution time
init = time()

# Call the forecast method of the StatsForecast instance to predict the next 28 days (h=28) 
fcst_df = sf.forecast(df=Y_df, h=28)

# Get the current time after the forecasting ends
end = time()

# Calculate and print the total time taken for the forecasting in minutes
print(f'Forecast Minutes: {(end - init) / 60}')

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals

# Import the necessary models from various libraries

# LGBMRegressor: A gradient boosting framework that uses tree-based learning algorithms from the LightGBM library
from lightgbm import LGBMRegressor

# XGBRegressor: A gradient boosting regressor model from the XGBoost library
from xgboost import XGBRegressor

# LinearRegression: A simple linear regression model from the scikit-learn library
from sklearn.linear_model import LinearRegression

# Instantiate the MLForecast object
mlf = MLForecast(
    models=[LGBMRegressor(verbosity=-1), XGBRegressor(), LinearRegression()],  # List of models for forecasting: LightGBM, XGBoost and Linear Regression
    freq='D',  # Frequency of the data - 'D' for daily frequency
    lags=list(range(1, 7)),  # Specific lags to use as regressors: 1 to 6 days
    lag_transforms = {
        1: [ExpandingMean()],  # Apply expanding mean transformation to the lag of 1 day
    },
    date_features=['year', 'month', 'day', 'dayofweek', 'quarter', 'week'],  # Date features to use as regressors
)

# Start the timer to calculate the time taken for fitting the models
init = time()

# Fit the MLForecast models to the data
mlf.fit(Y_df)

# Calculate the end time after fitting the models
end = time()

# Print the time taken to fit the MLForecast models, in minutes
print(f'MLForecast Minutes: {(end - init) / 60}')

fcst_mlf_df = mlf.predict(28)

import ray.tune as tune
import torch

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss

nf = NeuralForecast(
    models=[
        AutoNHITS(h=48, config=config_nhits, loss=MQLoss(), backend='optuna', num_samples=5),
        AutoLSTM(h=48, config=config_lstm, loss=MQLoss(), backend='optuna', num_samples=2),
    ],
    freq=1,
)

nf.fit(df=Y_df)

fcst_nf_df = nf.predict()
fcst_nf_df.columns = fcst_nf_df.columns.str.replace('-median', '')

# Merge the forecasts from StatsForecast and NeuralForecast
fcst_df = fcst_df.merge(fcst_nf_df, how='left', on=['unique_id', 'ds'])

# Merge the forecasts from MLForecast into the combined forecast dataframe
fcst_df = fcst_df.merge(fcst_mlf_df, how='left', on=['unique_id', 'ds'])

# Cross-validation
# StatsForecast
sf.verbose = False
init = time()
cv_df = sf.cross_validation(df=Y_df, h=horizon, n_windows=3, step_size=horizon)
end = time()
print(f'CV Minutes: {(end - init) / 60}')

# MLForecast
init = time()
cv_mlf_df = mlf.cross_validation(
    df=Y_df, 
    h=horizon,
    n_windows=3,
)
end = time()
print(f'CV Minutes: {(end - init) / 60}')

# NeuralForecast
init = time()
cv_nf_df = nf.cross_validation(
    df=Y_df, 
    h=horizon,
    n_windows=3,
)
end = time()
print(f'CV Minutes: {(end - init) / 60}')

# Merge cross validation forecasts
cv_df = cv_df.merge(cv_nf_df.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
cv_df = cv_df.merge(cv_mlf_df.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])

agg_cv_df = cv_df.loc[:,~cv_df.columns.str.contains('hi|lo')].groupby(['ds', 'cutoff']).sum(numeric_only=True).reset_index()
agg_cv_df.insert(0, 'unique_id', 'agg_demand')

agg_Y_df = Y_df.groupby(['ds']).sum(numeric_only=True).reset_index()
agg_Y_df.insert(0, 'unique_id', 'agg_demand')

# Evaluation Metrics
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse, mae, smape

evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=[mse, mae, smape])
evaluation_df

# group value by metric
by_metric = evaluation_df.groupby('metric').mean(numeric_only=True)
by_metric

# Best models by metric
by_metric.idxmin(axis=1)

# Choose best model
# Choose the best model for each time series, metric, and cross validation window
evaluation_df['best_model'] = evaluation_df.idxmin(axis=1, numeric_only=True)
# count how many times a model wins per metric and cross validation window
count_best_model = evaluation_df.groupby(['metric', 'best_model']).size().rename('n').to_frame().reset_index()
# plot results
sns.barplot(count_best_model, x='n', y='best_model', hue='metric')