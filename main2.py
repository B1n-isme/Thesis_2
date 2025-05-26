# enhanced_main.py
"""
Enhanced main orchestration script for Bitcoin forecasting pipeline.

This improved pipeline includes:
1. Comprehensive model comparison across frameworks
2. Bitcoin-specific feature engineering
3. Systematic model selection
4. Ensemble methods
5. Advanced evaluation metrics
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append('src')

# Import configurations
from config.base import *

# Import enhanced modules
from src.models.neuralforecast.models import get_neural_models
from src.models.neuralforecast.auto_models import (
    get_auto_models, get_loss_functions
)
from src.models.statsforecast.models import get_bitcoin_optimized_models
from src.pipelines.model_selection import ModelComparison, run_comprehensive_model_selection
from src.dataset.data_preparation import prepare_data
from src.utils.other import setup_environment, print_data_info

warnings.filterwarnings('ignore')

class EnhancedBitcoinForecastingPipeline:
    """Enhanced Bitcoin forecasting pipeline with comprehensive model comparison."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'horizon': HORIZON,
            'test_length_multiplier': TEST_LENGTH_MULTIPLIER,
            'seed': SEED,
            'cv_n_windows': CV_N_WINDOWS,
            'num_samples_per_model': NUM_SAMPLES_PER_MODEL,
            'max_features': 30,
            'include_neural_models': True,
            'include_stat_models': True,
            'fast_mode': False,  # Use fewer models and samples for quick testing
        }
    
    def setup_environment(self):
        """Setup the forecasting environment."""
        print("=== Setting up Environment ===")
        setup_environment(seed=self.config['seed'])
        
        # Create output directories
        for dir_path in [FORECASTS_DIR, METRICS_DIR, PLOTS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_neural_models_for_comparison(self) -> Tuple[List, List]:
        """Get models for comprehensive comparison."""
        neural_models = []
        stat_models = []
        
        horizon = self.config['horizon']
        num_samples = 3 if self.config['fast_mode'] else self.config['num_samples_per_model']
        
        # Neural models
        if self.config['include_neural_models']:
            if self.config['fast_mode']:
                # Quick models for fast testing
                neural_models = get_neural_models(horizon)
            else:
                # Comprehensive auto models
                loss_functions = get_loss_functions()
                
                # Try different loss functions
                for loss_name, loss_fn in list(loss_functions.items())[:2]:  # Limit for demo
                    models = get_comprehensive_auto_models(
                        horizon=horizon, 
                        loss_fn=loss_fn, 
                        num_samples_per_model=num_samples
                    )
                    neural_models.extend(models)
        
        # Statistical models
        if self.config['include_stat_models']:
            stat_models = get_bitcoin_optimized_models(season_length=7)
            
            if self.config['fast_mode']:
                # Reduce to essential statistical models
                essential_stat_models = [
                    model for model in stat_models 
                    if any(name in model.__class__.__name__ for name in 
                          ['AutoARIMA', 'AutoETS', 'Naive', 'WindowAverage'])
                ]
                stat_models = essential_stat_models[:5]  # Limit for fast mode
        
        print(f"Neural models to evaluate: {len(neural_models)}")
        print(f"Statistical models to evaluate: {len(stat_models)}")
        
        return neural_models, stat_models
    
    def run_model_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        """Run comprehensive model selection."""
        print("\n=== Model Selection and Comparison ===")
        
        # Get models
        neural_models, stat_models = self.get_neural_models_for_comparison()
        
        # Initialize comparison
        comparison = ModelComparison(results_dir=HPO_DIR)
        
        # Run comparison
        results_df = comparison.compare_all_models(
            neural_models=neural_models,
            stat_models=stat_models,
            df_train=train_df,
            df_val=val_df,
            horizon=self.config['horizon']
        )
        
        # Store results
        self.results['model_comparison'] = results_df
        
        return results_df
    
    def create_ensemble(self, results_df: pd.DataFrame, train_df: pd.DataFrame, 
                       top_n: int = 3) -> Dict:
        """Create ensemble from top performing models."""
        print(f"\n=== Creating Ensemble from Top {top_n} Models ===")
        
        if len(results_df) < top_n:
            print(f"Warning: Only {len(results_df)} models available, using all.")
            top_n = len(results_df)
        
        top_models = results_df.head(top_n)
        
        ensemble_info = {
            'models': top_models['model_name'].tolist(),
            'frameworks': top_models['framework'].tolist(),
            'weights': self._calculate_ensemble_weights(top_models),
            'individual_performance': top_models[['model_name', 'mae', 'rmse', 'mape']].to_dict('records')
        }
        
        print("Ensemble composition:")
        for i, (model, framework, weight) in enumerate(zip(
            ensemble_info['models'], 
            ensemble_info['frameworks'], 
            ensemble_info['weights']
        )):
            print(f"  {i+1}. {model} ({framework}) - Weight: {weight:.3f}")
        
        return ensemble_info
    
    def _calculate_ensemble_weights(self, top_models: pd.DataFrame) -> List[float]:
        """Calculate ensemble weights based on performance."""
        # Inverse MAE weighting (better models get higher weights)
        mae_values = top_models['mae'].values
        inverse_mae = 1 / mae_values
        weights = inverse_mae / inverse_mae.sum()
        return weights.tolist()
    
    def evaluate_final_performance(self, ensemble_info: Dict, 
                                 train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Evaluate final performance on holdout test set."""
        print("\n=== Final Performance Evaluation ===")
        
        # This is a simplified version - in practice, you'd need to:
        # 1. Retrain top models on full training data
        # 2. Generate predictions on test set
        # 3. Combine predictions using ensemble weights
        # 4. Calculate final metrics
        
        # For now, return placeholder results
        final_results = {
            'ensemble_mae': np.nan,  # Would calculate actual ensemble performance
            'ensemble_rmse': np.nan,
            'ensemble_mape': np.nan,
            'individual_results': ensemble_info['individual_performance'],
            'test_period': {
                'start_date': test_df['ds'].min(),
                'end_date': test_df['ds'].max(),
                'n_days': len(test_df)
            }
        }
        
        return final_results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            'pipeline_config': self.config,
            'data_info': {
                'total_features': len(self.results.get('enhanced_train_df', pd.DataFrame()).columns) - 3,
                'training_period': self.results.get('training_period'),
                'test_period': self.results.get('test_period')
            },
            'model_comparison': {
                'total_models_tested': len(self.results.get('model_comparison', pd.DataFrame())),
                'successful_models': len(self.results.get('model_comparison', pd.DataFrame()).dropna()),
                'best_model': self.results.get('model_comparison', pd.DataFrame()).iloc[0]['model_name'] if len(self.results.get('model_comparison', pd.DataFrame())) > 0 else None
            },
            'ensemble_info': self.results.get('ensemble_info', {}),
            'final_performance': self.results.get('final_performance', {})
        }
        
        return report
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete enhanced pipeline."""
        
        try:
            # 1. Setup
            self.setup_environment()
            
            # 2. Data preparation
            train_df, test_df, hist_exog_list = prepare_data(horizon=self.config['horizon'], 
                                            test_length_multiplier=self.config['test_length_multiplier']
                                        )
            self.results['enhanced_train_df'] = train_df
            self.results['enhanced_test_df'] = test_df
            self.results['hist_exog_features'] = hist_exog_list
            
            # 3. Model selection
            # Create validation split from training data for model selection
            val_split_size = min(len(train_df) // 4, self.config['horizon'] * 3)  # Use 25% or 3*horizon, whichever is smaller
            train_for_selection = train_df.iloc[:-val_split_size].copy()
            val_for_selection = train_df.iloc[-val_split_size:].copy()
            
            results_df = self.run_model_selection(train_for_selection, val_for_selection)
            
            # 4. Ensemble creation
            if len(results_df) > 0:
                ensemble_info = self.create_ensemble(results_df, train_df)
                self.results['ensemble_info'] = ensemble_info
                
                # 5. Final evaluation
                final_performance = self.evaluate_final_performance(ensemble_info, train_df, test_df)
                self.results['final_performance'] = final_performance
            
            # 6. Generate report
            report = self.generate_report()
            
            print("\n=== Pipeline Complete ===")
            print(f"Best model: {report['model_comparison']['best_model']}")
            print(f"Total models tested: {report['model_comparison']['total_models_tested']}")
            
            return report
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Bitcoin Forecasting Pipeline")
    parser.add_argument('--fast-mode', action='store_true', 
                       help='Run in fast mode with fewer models for quick testing')
    parser.add_argument('--no-features', action='store_true',
                       help='Skip feature engineering')
    parser.add_argument('--neural-only', action='store_true',
                       help='Only test neural models')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only test statistical models')
    parser.add_argument('--horizon', type=int, default=HORIZON,
                       help='Forecast horizon in days')
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = {
        'horizon': args.horizon,
        'test_length_multiplier': TEST_LENGTH_MULTIPLIER,
        'seed': SEED,
        'cv_n_windows': 3 if args.fast_mode else CV_N_WINDOWS,
        'num_samples_per_model': 2 if args.fast_mode else NUM_SAMPLES_PER_MODEL,
        'max_features': 15 if args.fast_mode else 30,
        'include_neural_models': not args.stats_only,
        'include_stat_models': not args.neural_only,
        'fast_mode': args.fast_mode,
    }
    
    print("=== Enhanced Bitcoin Forecasting Pipeline ===")
    print(f"Configuration: {config}")
    
    # Run pipeline
    pipeline = EnhancedBitcoinForecastingPipeline(config)
    report = pipeline.run_complete_pipeline()
    
    # Print summary
    if 'error' not in report:
        print("\n=== Final Summary ===")
        print(f"✓ Pipeline completed successfully")
        print(f"✓ Best performing model: {report['model_comparison']['best_model']}")
        print(f"✓ Models successfully tested: {report['model_comparison']['successful_models']}")
        print(f"✓ Enhanced features created: {report['data_info']['total_features']}")
        
        if 'ensemble_info' in report:
            print(f"✓ Ensemble created with {len(report['ensemble_info']['models'])} models")
    else:
        print(f"✗ Pipeline failed: {report['error']}")

if __name__ == "__main__":
    main()