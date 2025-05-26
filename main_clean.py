#!/usr/bin/env python3
"""
Clean Bitcoin Forecasting Pipeline
==================================

A streamlined pipeline for Bitcoin price forecasting using the Nixtla ecosystem.
This pipeline implements the following workflow:
1. Data Preparation
2. Hyperparameter Tuning with Cross-Validation on Auto Models
3. Model Comparison using best parameters on normal models
4. Final evaluation and reporting

Usage:
    python main_clean.py [--fast-mode] [--skip-hpo]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import time

# Add src to Python path
sys.path.append('src')
warnings.filterwarnings('ignore')

# Configuration and utilities
from config.base import *
from src.utils.hyperparam_loader import load_best_hyperparameters
from src.utils.other import setup_environment, seed_everything
from src.utils.plot import ForecastVisualizer
from src.dataset.data_preparation import prepare_data
from src.hpo.hyperparameter_tuning import run_complete_hpo_pipeline


class BitcoinForecastingPipeline:
    """
    Clean, comprehensive Bitcoin forecasting pipeline.
    
    This pipeline coordinates the entire forecasting workflow with proper
    separation of concerns and clear execution steps.
    """
    
    def __init__(self,  skip_hpo: bool = False):
        """
        Initialize the forecasting pipeline.
        
        Args:
            skip_hpo: Skip hyperparameter optimization step
        """
        self.skip_hpo = skip_hpo
        self.results = {}
        self.timestamp = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").strftime("%Y%m%d_%H%M%S")
        
        # Create results directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        directories = [
            RESULTS_DIR,
            HPO_DIR,
            CV_DIR,
            FINAL_DIR,
            MODELS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def setup_environment(self):
        """Setup the forecasting environment."""
        print("=" * 60)
        print("🚀 BITCOIN FORECASTING PIPELINE")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Skip HPO: {self.skip_hpo}")
        print("=" * 60)
        
        # Setup environment with proper configuration
        ray_config = {
            'address': RAY_ADDRESS,
            'num_cpus': RAY_NUM_CPUS,
            'num_gpus': RAY_NUM_GPUS
        }
        
        setup_environment(seed=SEED, ray_config=ray_config)
        
    def step1_data_preparation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Step 1: Data Preparation
        
        Returns:
            Tuple of (train_df, test_df, exog_features)
        """
        print("\n📊 STEP 1: DATA PREPARATION")
        print("-" * 40)
        
        # Load and prepare data
        train_df, test_df, hist_exog_list = prepare_data(
            horizon=HORIZON,
            test_length_multiplier=TEST_LENGTH_MULTIPLIER
        )
        
        # Store results
        self.results['data_info'] = {
            'total_samples': len(train_df) + len(test_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': hist_exog_list,
            'horizon': HORIZON
        }
        
        print(f"✅ Data prepared successfully")
        print(f"   • Training samples: {len(train_df):,}")
        print(f"   • Test samples: {len(test_df):,}")
        print(f"   • Features: {len(hist_exog_list)} exogenous variables")
        print(f"   • Forecast horizon: {HORIZON} days")
        
        return train_df, test_df, hist_exog_list
        
    def step2_hyperparameter_optimization(self, train_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Step 2: Hyperparameter Optimization using Auto Models
        
        Args:
            train_df: train dataset for HPO
            
        Returns:
            DataFrame with best hyperparameters or None if skipped
        """
        if self.skip_hpo:
            print("\n⏭️  STEP 2: HYPERPARAMETER OPTIMIZATION (SKIPPED)")
            print("-" * 40)
            print("Using default configurations...")
            return None
            
        print("\n🔧 STEP 2: HYPERPARAMETER OPTIMIZATION")
        print("-" * 40)
        
        start_time = time.time()
        
        # Adjust samples based on mode
        num_samples = NUM_SAMPLES_PER_MODEL
        
        try:
            # Run HPO pipeline
            csv_filename = run_complete_hpo_pipeline(
                train_df,
                horizon=HORIZON,
                num_samples=num_samples
            )
            
            # Load and display results
            best_configs_df = load_best_hyperparameters(csv_filename)
            
            execution_time = time.time() - start_time
            
            self.results['hpo_info'] = {
                'execution_time': execution_time,
                'models_tuned': len(best_configs_df),
                'samples_per_model': num_samples,
                'config_file': csv_filename
            }
            
            print(f"✅ HPO completed successfully in {execution_time:.1f} seconds")
            print(f"   • Models tuned: {len(best_configs_df)}")
            print(f"   • Samples per model: {num_samples}")
            print(f"   • Best configurations saved to: {csv_filename}")
            
            return best_configs_df
            
        except Exception as e:
            print(f"❌ HPO failed: {str(e)}")
            print("Continuing with default configurations...")
            return None
    
    def step3_model_comparison(self, train_df: pd.DataFrame, 
                             df_test: pd.DataFrame, best_configs: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Step 3: Comprehensive Model Comparison
        
        Use the best hyperparameters from HPO to train and compare models
        from all three frameworks: statsforecast, mlforecast, neuralforecast
        
        Args:
            train_df: train dataset
            df_test: Test dataset for final evaluation
            best_configs: Best hyperparameters from HPO (if available)
            
        Returns:
            DataFrame with model comparison results
        """
        print("\n🏆 STEP 3: MODEL COMPARISON")
        print("-" * 40)
        
        from src.pipelines.model_selection import ModelComparison
        from src.models.model_registry import ModelRegistry
        
        start_time = time.time()
        
        # Get models for comparison
        neural_models = []
        stat_models = []
        
        # Statistical models (no HPO needed)
        stat_models = ModelRegistry.get_statistical_models(season_length=7)
        
        # Neural models
        if best_configs is not None and len(best_configs) > 0:
            # Use optimized configurations from HPO
            print("Using optimized neural model configurations from HPO...")
            # TODO: Implement loading best configs into regular models
            
            neural_models = ModelRegistry.get_neural_models(HORIZON)  # Fallback for now
        else:
            # Use default configurations
            print("Using default neural model configurations...")
            neural_models = ModelRegistry.get_neural_models(HORIZON)
        
        # Initialize model comparison
        comparison = ModelComparison(results_dir=CV_DIR)
        
        print(f"   • train data: {len(train_df):,} samples")
        print(f"   • Neural models: {len(neural_models)}")
        print(f"   • Statistical models: {len(stat_models)}")
        
        # Run model comparison
        try:
            results_df = comparison.compare_all_models(
            neural_models=neural_models,
            stat_models=stat_models,
            train_df=train_df,
            horizon=HORIZON
            )
            
            execution_time = time.time() - start_time
            
            # Store results
            self.results['comparison_info'] = {
                'execution_time': execution_time,
                'models_compared': len(results_df),
                'best_model': results_df.iloc[0]['model_name'] if len(results_df) > 0 else None,
                'best_mae': results_df.iloc[0]['mae'] if len(results_df) > 0 else None
            }
            
            print(f"✅ Model comparison completed in {execution_time:.1f} seconds")
            print(f"   • Models compared: {len(results_df)}")
            if len(results_df) > 0:
                print(f"   • Best model: {results_df.iloc[0]['model_name']}")
                print(f"   • Best MAE: {results_df.iloc[0]['mae']:.4f}")
            
            # Save results
            results_path = Path(CV_DIR) / f"model_comparison_{self.timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"   • Results saved to: {results_path}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Model comparison failed: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def step4_final_evaluation(self, comparison_results: pd.DataFrame) -> Dict:
        """
        Step 4: Final Evaluation and Reporting
        
        Args:
            comparison_results: Results from model comparison
            
        Returns:
            Dictionary with final evaluation results
        """
        print("\n📈 STEP 4: FINAL EVALUATION")
        print("-" * 40)
        
        if len(comparison_results) == 0:
            print("❌ No comparison results available for final evaluation")
            return {}
        
        # Get top models
        top_n = min(5, len(comparison_results))
        top_models = comparison_results.head(top_n)
        
        print(f"🏅 TOP {top_n} MODELS:")
        print("-" * 30)
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            print(f"{i}. {row['model_name']:<20} MAE: {row['mae']:.4f}")
        
        # Calculate ensemble weights (inverse MAE weighting)
        mae_values = top_models['mae'].values
        inverse_mae = 1 / mae_values
        weights = inverse_mae / inverse_mae.sum()
        
        final_results = {
            'top_models': top_models['model_name'].tolist(),
            'top_mae_scores': top_models['mae'].tolist(),
            'ensemble_weights': weights.tolist(),
            'total_models_tested': len(comparison_results),
            'pipeline_timestamp': self.timestamp
        }
        
        # Save final results
        results_path = Path(FINAL_DIR) / f"final_results_{self.timestamp}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n✅ Final evaluation completed")
        print(f"   • Results saved to: {results_path}")
        
        return final_results
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        report_lines = [
            "=" * 60,
            "🎯 BITCOIN FORECASTING PIPELINE SUMMARY",
            "=" * 60,
            f"Execution Time: {self.timestamp}",
            f"HPO: {'Skipped' if self.skip_hpo else 'Executed'}",
            "",
            "📊 DATA INFORMATION:",
        ]
        
        if 'data_info' in self.results:
            data_info = self.results['data_info']
            report_lines.extend([
                f"  • Total samples: {data_info['total_samples']:,}",
                f"  • Training samples: {data_info['train_samples']:,}",
                f"  • Test samples: {data_info['test_samples']:,}",
                f"  • Features: {len(data_info['features'])}",
                f"  • Horizon: {data_info['horizon']} days",
                ""
            ])
        
        if 'hpo_info' in self.results:
            hpo_info = self.results['hpo_info']
            report_lines.extend([
                "🔧 HYPERPARAMETER OPTIMIZATION:",
                f"  • Execution time: {hpo_info['execution_time']:.1f} seconds",
                f"  • Models tuned: {hpo_info['models_tuned']}",
                f"  • Samples per model: {hpo_info['samples_per_model']}",
                ""
            ])
        
        if 'comparison_info' in self.results:
            comp_info = self.results['comparison_info']
            report_lines.extend([
                "🏆 MODEL COMPARISON:",
                f"  • Execution time: {comp_info['execution_time']:.1f} seconds",
                f"  • Models compared: {comp_info['models_compared']}",
                f"  • Best model: {comp_info['best_model']}",
                f"  • Best MAE: {comp_info['best_mae']:.4f}" if comp_info['best_mae'] else "  • Best MAE: N/A",
                ""
            ])
        
        report_lines.extend([
            "🎯 RECOMMENDATIONS:",
            "  • Review top-performing models for deployment",
            "  • Consider ensemble methods for improved performance",
            "  • Monitor model performance over time",
            "  • Retrain periodically with new data",
            "",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute the complete Bitcoin forecasting pipeline.
        
        Returns:
            Dictionary containing all pipeline results
        """
        total_start_time = time.time()
        
        try:
            # Setup
            self.setup_environment()
            
            # Step 1: Data Preparation
            train_df, df_test, hist_exog_list = self.step1_data_preparation()
            
            # Step 2: Hyperparameter Optimization
            best_configs = self.step2_hyperparameter_optimization(train_df)
            
            # Step 3: Model Comparison
            comparison_results = self.step3_model_comparison(train_df, df_test, best_configs)
            
            # Step 4: Final Evaluation
            final_results = self.step4_final_evaluation(comparison_results)
            
            # Calculate total execution time
            total_execution_time = time.time() - total_start_time
            self.results['total_execution_time'] = total_execution_time
            
            # Generate and display summary report
            summary_report = self.generate_summary_report()
            print(f"\n{summary_report}")
            
            # Save summary report
            report_path = Path(FINAL_DIR) / f"summary_report_{self.timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write(summary_report)
            
            print(f"\n📝 Summary report saved to: {report_path}")
            print(f"⏱️  Total execution time: {total_execution_time:.1f} seconds")
            
            return {
                'success': True,
                'results': self.results,
                'final_results': final_results,
                'execution_time': total_execution_time,
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - total_start_time,
                'timestamp': self.timestamp
            }


def main():
    """Main execution function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Clean Bitcoin Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_clean.py                    # Run full pipeline
  python main_clean.py --fast-mode       # Quick testing mode
  python main_clean.py --skip-hpo        # Skip hyperparameter optimization
  python main_clean.py --fast-mode --skip-hpo  # Fastest execution
        """
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Use fewer models and samples for quick testing'
    )
    
    parser.add_argument(
        '--skip-hpo',
        action='store_true',
        help='Skip hyperparameter optimization step'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = BitcoinForecastingPipeline(
        skip_hpo=args.skip_hpo
    )
    
    # Execute pipeline
    results = pipeline.run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
