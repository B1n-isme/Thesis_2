"""
Unified Model Registry for Bitcoin Forecasting

This module provides a centralized registry for all models across
statsforecast, mlforecast, and neuralforecast frameworks, eliminating
redundancy and ensuring consistency.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Import from modular structure


# Framework imports for loss functions
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE

# Configuration
from config.base import HORIZON, FREQUENCY, LOCAL_SCALER_TYPE


class ModelRegistry:
    """
    Centralized registry for all forecasting models.
    
    This class manages model definitions across all three frameworks
    and provides consistent interfaces for model selection.
    """
    
    @staticmethod
    def get_statistical_models(season_length: int = 7) -> List[Any]:
        """
        Get optimized statistical models for Bitcoin forecasting.
        
        Args:
            season_length: Seasonal period (default 7 for weekly patterns)
            
        Returns:
            List of statistical model instances
        """
        from src.models.statsforecast.models import get_statistical_models as _get_statistical_models
        return _get_statistical_models(season_length)
    
    @staticmethod
    def get_neural_models(horizon: int, hist_exog_list: Optional[List[str]] = None, hyperparameters_json_path: Optional[str] = None) -> List[Any]:
        """
        Get neural forecasting models with fixed hyperparameters.
        
        Args:
            horizon: Forecast horizon
            hist_exog_list: List of historical exogenous features
            
        Returns:
            List of neural model instances
        """
        from src.models.neuralforecast.models import get_neural_models as _get_neural_models
        return _get_neural_models(horizon, hist_exog_list, hyperparameters_json_path)
    
    @staticmethod
    def get_auto_models(horizon: int, num_samples: int = 10,
                       hist_exog_list: Optional[List[str]] = None) -> List[Any]:
        """
        Get auto models for hyperparameter optimization.
        
        Args:
            horizon: Forecast horizon
            num_samples: Number of hyperparameter samples per model
            hist_exog_list: List of historical exogenous features
            
        Returns:
            List of auto model instances for HPO
        """
        from src.models.neuralforecast.auto_models import get_auto_models as _get_auto_models
        return _get_auto_models(horizon, num_samples, hist_exog_list)
    
    @staticmethod
    def get_model_summary() -> Dict[str, int]:
        """
        Get summary of available models.
        
        Returns:
            Dictionary with model counts by category
        """
        from src.models.statsforecast.models import get_statistical_models as _get_statistical_models
        from src.models.neuralforecast.models import get_neural_models as _get_neural_models
        from src.models.neuralforecast.auto_models import get_auto_models as _get_auto_models
        
        return {
            'statistical_models': len(_get_statistical_models()),
            'neural_models': len(_get_neural_models(HORIZON)),
            'auto_models': len(_get_auto_models(HORIZON)),
        }