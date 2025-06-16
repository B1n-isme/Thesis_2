"""
Bitcoin Forecasting Pipeline - Reporting and Visualization
============================================================

A streamlined pipeline for generating reports and visualizations from existing
Bitcoin price forecasting model results. This script assumes that the model
cross-validation and forecasting have already been run.

It performs the following steps:
1. Load existing CV results and forecast data.
2. Display top-performing models.
3. Generate and save visualizations.
4. Generate and save a summary report.

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

from src.models.statsforecast.models import get_statistical_models
from src.models.mlforecast.models import get_ml_models
from src.models.neuralforecast.models import get_neural_models

# Pipeline Step Modules
from src.dataset.data_preparation import prepare_pipeline_data
from src.pipelines.results_processing import (
    display_top_models,
    generate_summary_report,
)
from src.pipelines.visualization import create_visualizations_step
from src.utils.utils import get_horizon_directories, load_json_to_dict


class BitcoinAnalysisPipeline:
    """
    Generates reports and visualizations from existing model results.
    """

    def __init__(self):
        self.pipeline_results = {}  # Stores various results and info from pipeline steps
        self.cv_dir = None
        self.final_dir = None
        self.plot_dir = None

    def _load_results(self):
        """Loads the latest holdout metrics and forecast results."""
        print("\n🔎 STEP 2: LOADING PRE-COMPUTED RESULTS")
        print("-" * 40)

        # Set directories based on horizon
        self.cv_dir, self.final_dir, self.plot_dir = get_horizon_directories()
        print(f"  ✓ Using directories for {HORIZON}d horizon:")
        print(f"    CV: {self.cv_dir}")
        print(f"    Final: {self.final_dir}")
        print(f"    Plot: {self.plot_dir}")

        # Load holdout metrics from the final results directory
        metrics_results_path = self.final_dir / "metrics_results.csv"
        metrics_df = pd.read_csv(metrics_results_path)
        print(f"  ✓ Holdout metrics loaded from: {metrics_results_path}")

        # Load forecast data for plotting
        final_plot_results_path = self.final_dir / f"final_plot_results.json"
        all_forecasts_data = load_json_to_dict(final_plot_results_path)
        print(f"  ✓ Plot data loaded from: {final_plot_results_path}")

        return metrics_df, all_forecasts_data

    def _get_workflow_info(self, hist_exog_list: List[str]) -> Dict:
        """Reconstructs the workflow info dictionary for the summary report."""
        stat_models_instances = get_statistical_models(season_length=HORIZON)
        ml_models_instances = get_ml_models()
        neural_models_instances = get_neural_models(
            horizon=HORIZON,
            num_samples=NUM_SAMPLES_PER_MODEL,
            hist_exog_list=hist_exog_list,
        )
        return {
            "stat_models_count": len(stat_models_instances),
            "ml_models_count": len(ml_models_instances),
            "neural_models_count": len(neural_models_instances),
            "total_models": len(stat_models_instances)
            + len(ml_models_instances)
            + len(neural_models_instances),
        }

    def run_analysis_pipeline(self) -> Dict:
        """Run the complete analysis and reporting pipeline."""
        print("\n🚀 Starting Bitcoin Analysis Pipeline...")
        start_time = time.time()

        try:
            # We still need the data for context and plotting
            train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data()
            self.pipeline_results["data_preparation_info"] = data_info

            # This info is needed for the report
            self.pipeline_results["auto_models_workflow_info"] = self._get_workflow_info(hist_exog_list)

            metrics_df, all_forecasts_data = self._load_results()
            
            print("\n🎨 STEP 3: CREATING VISUALIZATIONS")
            print("-" * 40)
            create_visualizations_step(
                train_df, test_df, all_forecasts_data, HORIZON, self.plot_dir
            )

            print("\n📝 STEP 4: GENERATING SUMMARY REPORT")
            print("-" * 40)
            summary_report_str = generate_summary_report(
                metrics_df,
                HORIZON,
                self.pipeline_results["data_preparation_info"],
                self.pipeline_results["auto_models_workflow_info"],
                RESULTS_DIR,
            )

            execution_time = time.time() - start_time
            print(f"\n✅ Analysis pipeline completed in {execution_time:.1f} seconds")
            print("=" * 60)

            return {
                "status": "success",
                "results_dataframe": metrics_df,
                "summary_report_text": summary_report_str,
                "pipeline_execution_details": self.pipeline_results,
            }

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback

            traceback.print_exc()
            return {
                "status": "failed",
                "error_message": str(e),
                "pipeline_execution_details": self.pipeline_results,
            }


def main():
    warnings.filterwarnings("ignore")
    pipeline = BitcoinAnalysisPipeline()
    run_outcome = pipeline.run_analysis_pipeline()

    if run_outcome["status"] == "success":
        print("\n📋 PIPELINE SUCCEEDED")
        print("Summary report and visualizations have been saved to the 'results' directory.")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline execution failed: {run_outcome['error_message']}")
        sys.exit(1)


if __name__ == "__main__":
    main() 