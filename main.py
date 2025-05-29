"""
Bitcoin Forecasting Pipeline - Auto Models Direct Workflow
===========================================================

A streamlined pipeline for Bitcoin price forecasting using Auto Models directly.
This pipeline implements the following simplified workflow:
1. Data Preparation
2. Direct Auto Models Workflow (CV, Fit-Predict)
3. Visualization and Reporting

Usage:
    python main.py
"""

import sys
import time
import warnings
from pathlib import Path
import pandas as pd
from typing import Dict, List
# Configuration and Model
from config.base import *
from src.models.neuralforecast.models import get_neural_models
from src.models.statsforecast.models import get_statistical_models

# Pipeline Step Modules
from src.pipelines.pipeline_setup import setup_pipeline_directories, setup_environment
from src.pipelines.data_preparation import prepare_pipeline_data
from src.pipelines.model_evaluation import perform_cross_validation
from src.pipelines.model_forecasting import perform_final_fit_predict
from src.pipelines.results_processing import (
    combine_results,
    display_top_models,
    generate_summary_report
)
from src.pipelines.visualization import create_visualizations_step


class BitcoinAutoModelsPipeline:
    """
    Simplified Bitcoin forecasting pipeline using Auto Models directly.
    """
    
    def __init__(self):
        self.pipeline_results = {} # Stores various results and info from pipeline steps
        self.timestamp = pd.Timestamp.now(tz='Asia/Ho_Chi_Minh')
        self.all_forecasts_data = {}  # Stores raw forecasts for plotting
        
        # Setup directories and store paths in pipeline_results
        self.dir_paths = setup_pipeline_directories(RESULTS_DIR)
        self.pipeline_results['directory_paths'] = self.dir_paths

    def _run_auto_models_workflow(self, train_df: pd.DataFrame, test_df: pd.DataFrame, hist_exog_list: List[str]) -> pd.DataFrame:
        """Core workflow for Auto Models: CV, fit-predict, and result combination."""
        print("\n🤖 STEP 2: AUTO MODELS WORKFLOW")
        print("-" * 40)
        start_time = time.time()

        neural_models_instances = get_neural_models(
            horizon=HORIZON,
            num_samples=NUM_SAMPLES_PER_MODEL,
            hist_exog_list=hist_exog_list
        )
        stat_models_instances = get_statistical_models(season_length=7)

        print(f"Models loaded:")
        print(f"   • Neural models: {len(neural_models_instances)}")
        print(f"   • Statistical models: {len(stat_models_instances)}")

        print("\n--- Step 2a: Cross-Validation Model Evaluation ---")
        cv_results_df = perform_cross_validation(neural_models_instances, stat_models_instances, train_df)
        
        print("\n--- Step 2b: Final Fit-Predict on Test Data ---")
        final_results_df, self.all_forecasts_data = perform_final_fit_predict(
            neural_models_instances, 
            stat_models_instances, 
            train_df, 
            test_df, 
            self.dir_paths['auto_model_configs_dir'],
            self.all_forecasts_data # Pass dict to be updated
        )
        
        print("\n--- Step 2c: Combining Results ---")
        comparison_df = combine_results(cv_results_df, final_results_df)
        
        execution_time = time.time() - start_time
        self.pipeline_results['auto_models_workflow_info'] = {
            'execution_time': execution_time,
            'auto_models_count': len(neural_models_instances),
            'stat_models_count': len(stat_models_instances),
            'total_models': len(neural_models_instances) + len(stat_models_instances)
        }
        print(f"✅ Auto Models workflow completed in {execution_time:.1f} seconds")
        return comparison_df

    def run_complete_pipeline(self) -> Dict:
        """Run the complete simplified Auto Models pipeline."""
        print("\n🚀 Starting Bitcoin Auto Models Pipeline...")
        
        try:
            setup_environment(self.timestamp)
            
            train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data()
            self.pipeline_results['data_preparation_info'] = data_info
            
            comparison_results_df = self._run_auto_models_workflow(train_df, test_df, hist_exog_list)
            
            top_model_names = display_top_models(comparison_results_df, "Final Model Rankings", top_n=10)
            
            create_visualizations_step(
                train_df, 
                test_df, 
                self.all_forecasts_data, 
                self.dir_paths['plot_dir'], 
                self.timestamp
            )
            
            summary_report_str = generate_summary_report(
                comparison_results_df, 
                self.timestamp,
                self.pipeline_results['data_preparation_info'], 
                self.pipeline_results['auto_models_workflow_info'], 
                self.dir_paths['results_dir']
            )
            
            results_file_path = self.dir_paths['final_dir'] / f'auto_pipeline_results_{self.timestamp.strftime("%Y%m%d_%H%M%S")}.csv'
            comparison_results_df.to_csv(results_file_path, index=False)
            print(f"📊 Results saved: {results_file_path}")
            
            print("\n" + "=" * 60)
            print("✅ AUTO MODELS PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return {
                'status': 'success',
                'results_dataframe': comparison_results_df,
                'top_models_list': top_model_names,
                'summary_report_text': summary_report_str,
                'pipeline_execution_details': self.pipeline_results
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'error_message': str(e),
                'pipeline_execution_details': self.pipeline_results
            }

def main():
    warnings.filterwarnings('ignore')
    pipeline = BitcoinAutoModelsPipeline()
    run_outcome = pipeline.run_complete_pipeline()
    
    if run_outcome['status'] == 'success':
        print("\n📋 PIPELINE SUMMARY:")
        print(run_outcome['summary_report_text'])
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline execution failed: {run_outcome['error_message']}")
        sys.exit(1)

if __name__ == "__main__":
    main() 