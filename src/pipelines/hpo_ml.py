from mlforecast import MLForecast
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from numba import njit
from window_ops.rolling import rolling_mean

# difference helps remove seasonality
target_transforms =[
    Differences([24]),
    LocalStandardScaler()
]

@njit
def rolling_mean_48(x):
    return rolling_mean(x, window_size=48)

fcst = MLForecast(
    models=[],
    freq=1,  # our series have integer timestamps, so we'll just add 1 in every timestep
    lags=[1, 24],
    target_transforms=target_transforms,
    lag_transforms={
        1: [ExpandingMean()],
        24: [RollingMean(window_size=48), rolling_mean_48],
    },
)

prep = fcst.preprocess(df)
prep