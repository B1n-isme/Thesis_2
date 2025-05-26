import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.arima import arima_string
from src.models.statsforecast.models import get_auto_statsmodels
from src.dataset.data_preparation import prepare_data
from config.base import HORIZON, TEST_LENGTH_MULTIPLIER


# Get data
train_df, test_df, hist_exog = prepare_data(horizon=HORIZON, test_length_multiplier=TEST_LENGTH_MULTIPLIER)


# Get models
models = get_auto_statsmodels(HORIZON)


# Instantiate StatsForecast
sf = StatsForecast(
    models=models,
    freq='D',
    n_jobs=-1,
    verbose=True
)

# forecasts_df = sf.forecast(df=train_df[['unique_id', 'ds', 'y']], X_df=test_df[['unique_id', 'ds']], h=HORIZON * TEST_LENGTH_MULTIPLIER)
forecasts_df = sf.forecast(df=train_df[['unique_id', 'ds', 'y'] + ['btc_sma_5', 'btc_trading_volume', 'Gold_Price']], X_df=test_df[['unique_id', 'ds'] + ['btc_sma_5', 'btc_trading_volume', 'Gold_Price']], h=HORIZON * TEST_LENGTH_MULTIPLIER)

print(forecasts_df.head())


fig = sf.plot(test_df, forecasts_df, models=["AutoARIMA", "AutoETS", "CES", "Naive"])

fig.savefig("results/forecasts/statsforecast.png")


