"""
Enhanced model configurations optimized for Bitcoin forecasting.
"""
from typing import List, Optional, Any
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM, GRU, RNN, TCN, DeepAR
from neuralforecast.losses.pytorch import MAE
from config.base import LOCAL_SCALER_TYPE


def get_neural_models(horizon: int, hist_exog_list: Optional[List[str]] = None) -> List[Any]:
    """
    Get neural forecasting models with fixed hyperparameters.
    
    Args:
        horizon: Forecast horizon
        hist_exog_list: List of historical exogenous features
        
    Returns:
        List of neural model instances
    """
    # Base configuration optimized for Bitcoin
    base_config = {
        'h': horizon,
        'input_size': horizon * 4,  # Look back 4x horizon
        'loss': MAE(),
        'learning_rate': 1e-3,
        'max_steps': 10,
        'val_check_steps': 50,
        'batch_size': 32,
        'windows_batch_size': 256,
        'scaler_type': LOCAL_SCALER_TYPE,
    }
    
    # Include exogenous features if available
    if hist_exog_list:
        base_config['hist_exog_list'] = hist_exog_list
    
    # Model definitions
    all_models = [
        # NHITS: Excellent for complex temporal patterns
        NHITS(
            **base_config,
            stack_types=['identity', 'identity', 'identity'],
            n_blocks=[1, 1, 1],
            mlp_units=[[512, 512], [512, 512], [512, 512]],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            interpolation_mode='linear',
        ),
        
        # NBEATS: Good for trend and seasonality decomposition
        # NBEATS(
        #     **base_config,
        #     stack_types=['trend', 'seasonality'],
        #     n_blocks=[3, 3],
        #     mlp_units=[[512, 512], [512, 512]],
        # ),
        
        # # LSTM: Sequential pattern learning
        # LSTM(
        #     **base_config,
        #     encoder_n_layers=2,
        #     encoder_hidden_size=128,
        #     decoder_hidden_size=128,
        #     decoder_layers=2,
        # ),
        # TFT(
        #     **base_config,
        #     hidden_size=64,
        #     lstm_layers=2,
        #     num_attention_heads=4,
        #     add_relative_index=True,
        # ),
        
        # # GRU: Alternative RNN architecture
        # GRU(
        #     **base_config,
        #     encoder_n_layers=2,
        #     encoder_hidden_size=128,
        #     decoder_hidden_size=128,
        #     decoder_layers=2,
        # ),
    ]
    
    return all_models
