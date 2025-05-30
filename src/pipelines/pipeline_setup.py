"""
Module for pipeline setup functionalities for the Auto Models workflow.
"""
from pathlib import Path
import ray
from typing import Dict

# Configuration and utilities
from config.base import RESULTS_DIR, RAY_ADDRESS, RAY_NUM_CPUS, RAY_NUM_GPUS, SEED
from utils.utils import setup_environment as util_setup_environment # Renamed to avoid conflict


def setup_pipeline_directories(base_results_dir: str) -> Dict[str, Path]:
    """Create necessary output directories and return their paths."""
    results_dir = Path(base_results_dir)
    plot_dir = results_dir / 'plot'
    final_dir = results_dir / 'final'
    models_dir = results_dir / 'models'
    auto_model_configs_dir = models_dir / 'auto_model_configs'
    
    directories = [
        results_dir,
        plot_dir,
        final_dir,
        models_dir,
        auto_model_configs_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    return {
        "results_dir": results_dir,
        "plot_dir": plot_dir,
        "final_dir": final_dir,
        "models_dir": models_dir,
        "auto_model_configs_dir": auto_model_configs_dir
    }

def setup_environment(pipeline_timestamp):
    """Setup the forecasting environment."""
    print("=" * 60)
    print("🚀 BITCOIN AUTO MODELS PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {pipeline_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Auto Models: Direct workflow (no separate HPO)")
    print("=" * 60)
    
    ray_config = {
        'address': RAY_ADDRESS,
        'num_cpus': RAY_NUM_CPUS,
        'num_gpus': RAY_NUM_GPUS
    }
    
    util_setup_environment(seed=SEED, ray_config=ray_config) 