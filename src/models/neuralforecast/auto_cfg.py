from typing import Dict
from ray import tune


def neural_auto_cfg(h: int) -> Dict[str, Dict]:
    """
    Get model configurations optimized for Bitcoin forecasting.

    Args:
        h: Forecast horizon (number of time steps to forecast)

    Returns:
        Dictionary of model configurations with Ray Tune choices for model-specific parameters only.
        Common parameters are handled in base_auto_config.
    """

    # NHITS config 
    nhits_config = {
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 2], [16, 8, 1]]
        ),  # MaxPool's Kernelsize
        "n_freq_downsample": tune.choice(
            [[168, 24, 1], [24, 12, 1], [1, 1, 1]]
        ),  # Interpolation expressivity ratios
    }

    # NBEATS config 
    nbeats_config = {}

    # LSTM config 
    lstm_config = {
        "encoder_n_layers": tune.choice([1, 2, 3]),
        "encoder_hidden_size": tune.choice([64, 128, 256]),
        "decoder_hidden_size": tune.choice([32, 64, 128, 256]),
    }

    # TFT config 
    tft_config = {
        "hidden_size": tune.choice([32, 64, 128]),
        "n_rnn_layers": tune.choice([1, 2]),
        "n_head": tune.choice([2, 4, 8]),
    }

    # Transformer config 
    transformer_config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([4, 8]),
    }

    # TSMixer config 
    tsmixer_config = {
        "n_block": tune.choice([2, 4, 6]),
        "ff_dim": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
    }

    # BiTCN config 
    bitcn_config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
    }

    # KAN config 
    kan_config = {
        "grid_size": tune.choice([5, 10, 15]),
        "spline_order": tune.choice([2, 3, 4]),
        "hidden_size": tune.choice([64, 128, 256]),
    }

    return {
        "nhits": nhits_config,
        "nbeats": nbeats_config,
        "lstm": lstm_config,
        "tft": tft_config,
        "transformer": transformer_config,
        "tsmixer": tsmixer_config,
        "bitcn": bitcn_config,
        "kan": kan_config,
    }
