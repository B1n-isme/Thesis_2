from typing import Dict
from ray import tune


def neural_auto_model_cfg(h: int) -> Dict[str, Dict]:
    """
    Get model configurations optimized for Bitcoin forecasting.

    Args:
        h: Forecast horizon (number of time steps to forecast)

    Returns:
        Dictionary of model configurations with Ray Tune choices for model-specific parameters only.
        Common parameters are handled in base_auto_config.
    """

    # NHITS config - only model-specific parameters
    nhits_config = {
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 2], [16, 8, 1]]
        ),  # MaxPool's Kernelsize
        "n_freq_downsample": tune.choice(
            [[168, 24, 1], [24, 12, 1], [1, 1, 1]]
        ),  # Interpolation expressivity ratios
    }

    # NBEATS config - only model-specific parameters
    nbeats_config = {
        # 'stack_types': tune.choice([
        #     ['trend', 'seasonality'],
        #     ['trend', 'trend', 'seasonality'],
        #     ['generic', 'generic']
        # ]),
    }

    # LSTM config - only model-specific parameters
    lstm_config = {
        "encoder_n_layers": tune.choice([1, 2, 3]),
        "encoder_hidden_size": tune.choice([64, 128, 256]),
    }

    # TFT config - only model-specific parameters
    tft_config = {
        "hidden_size": tune.choice([32, 64, 128]),
        "n_rnn_layers": tune.choice([1, 2]),
        "n_head": tune.choice([2, 4, 8]),
    }

    return {
        "nhits": nhits_config,
        "nbeats": nbeats_config,
        "lstm": lstm_config,
        "tft": tft_config,
    }


# Legacy function for backward compatibility
def neural_auto_model_cfg_legacy(h: int) -> Dict[str, Dict]:
    """
    Get model configurations optimized for Bitcoin price forecasting.
    Bitcoin exhibits high volatility, trending behavior, and potential regime changes.
    """

    # Enhanced NHITS config for crypto volatility
    nhits_config = {
        "input_size": tune.choice(
            [h * 2, h * 3, h * 4, h * 6, h * 8]
        ),  # Longer lookback for crypto
        "max_steps": tune.choice([500, 1000, 1500, 2000]),
        "learning_rate": tune.choice([1e-4, 5e-4, 1e-3, 2e-3]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "n_blocks": tune.choice([[2, 2], [3, 3], [4, 4]]),
        "mlp_units": tune.choice([[256, 256], [512, 512], [256, 128]]),
        # 'dropout_prob_theta': tune.choice([0.1, 0.2, 0.3]),
        "activation": tune.choice(["ReLU", "GELU"]),
        "stack_types": tune.choice(
            [["identity", "identity"], ["trend", "seasonality"]]
        ),
        "scaler_type": tune.choice(["standard", "robust", "minmax"]),
        "random_seed": tune.randint(1, 6),
    }

    # Enhanced NBEATS config
    nbeats_config = {
        "input_size": tune.choice([h * 3, h * 4, h * 6, h * 8]),
        "max_steps": tune.choice([500, 1000, 1500]),
        "learning_rate": tune.choice([1e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([128, 256]),
        "stack_types": tune.choice([["trend", "seasonality"], ["generic", "generic"]]),
        "n_blocks": tune.choice([2, 3, 4]),
        "mlp_units": tune.choice([256, 512]),
        # 'dropout_prob_theta': tune.choice([0.1, 0.2]),
        "scaler_type": tune.choice(["standard", "robust"]),
        "random_seed": tune.randint(1, 6),
    }

    # TFT config for complex temporal patterns
    tft_config = {
        "input_size": tune.choice([h * 4, h * 6, h * 8]),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_rnn_layers": tune.choice([1, 2]),
        "n_head": tune.choice([4, 8]),
        # 'dropout': tune.choice([0.1, 0.2, 0.3]),
        "learning_rate": tune.choice([1e-4, 5e-4, 1e-3]),
        "max_steps": tune.choice([1000, 1500, 2000]),
        "batch_size": tune.choice([16, 32]),
        "windows_batch_size": tune.choice([128, 256]),
        "scaler_type": tune.choice(["standard", "robust"]),
        "random_seed": tune.randint(1, 6),
    }

    return {"nhits": nhits_config, "nbeats": nbeats_config, "tft": tft_config}
