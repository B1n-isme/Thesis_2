"""
Bitcoin-optimized statistical models for forecasting.
"""
from typing import List, Any
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoCES, AutoTheta,
    Naive, RandomWalkWithDrift, WindowAverage,
    HistoricAverage, CrostonClassic, IMAPA, TSB, ADIDA,
    SeasonalNaive, SeasonalWindowAverage
)


def get_statistical_models(season_length: int = 7) -> List[Any]:
    """
    Get optimized statistical models for Bitcoin forecasting.
    
    Args:
        season_length: Seasonal period (default 7 for weekly patterns)
        
    Returns:
        List of statistical model instances
    """
    # Full model set optimized for Bitcoin characteristics
    all_models = [
        # PRIMARY: Best for Bitcoin's non-stationary, trending, volatile nature
        AutoARIMA(season_length=season_length, stationary=False, approximation=False),
        # AutoETS(season_length=season_length, model='ZZZ'),  # Auto-select
        # AutoETS(season_length=season_length, model='MMM'),  # Multiplicative
        # AutoETS(season_length=season_length, model='MAM'),  # Mixed
        
        # # THETA METHODS: Excellent for trending financial data
        # AutoTheta(season_length=season_length),
        
        # # COMPLEX SMOOTHING: Handles complex patterns
        # AutoCES(season_length=season_length),
        
        # # BASELINE MODELS: Simple but effective
        # RandomWalkWithDrift(),
        # WindowAverage(window_size=3),   # Very responsive
        # WindowAverage(window_size=7),   # Weekly patterns
        
        # # SEASONAL PATTERNS
        # SeasonalNaive(season_length=season_length),
        # SeasonalWindowAverage(season_length=season_length, window_size=3),
    ]
    
    return all_models


# Backward compatibility
def get_bitcoin_optimized_models(season_length: int = 7):
    """
    Backward compatibility function.
    """
    return get_statistical_models(season_length)