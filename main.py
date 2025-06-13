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
from src.pipelines.pipeline_setup import setup_environment
from src.dataset.data_preparation import prepare_pipeline_data
from src.pipelines.results_processing import (
    display_top_models,
    generate_summary_report,
)
from src.pipelines.visualization import create_visualizations_step
from src.utils.utils import load_json_to_dict


class BitcoinAnalysisPipeline:
    """
    Generates reports and visualizations from existing model results.
    """

    def __init__(self):
        self.pipeline_results = {}  # Stores various results and info from pipeline steps
        self.timestamp = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
        self.all_forecasts_data = {}  # Stores raw forecasts for plotting

    def _load_results(self):
        """Loads the latest CV and forecast results from the results directory."""
        print("\n🔎 STEP 1: LOADING PRE-COMPUTED RESULTS")
        print("-" * 40)

        # Load CV results
        # cv_dir_path = Path(CV_DIR)
        # list_of_cv_files = sorted(
        #     cv_dir_path.glob("cv_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        # )
        # if not list_of_cv_files:
        #     raise FileNotFoundError(f"No CV result files found in {CV_DIR}")
        # latest_cv_file = list_of_cv_files[0]
        # print(f"Loading latest CV results from: {latest_cv_file}")

        cv_metrics_df = pd.read_csv(CV_DIR / "cv_metrics.csv")

        # Load forecast data for plotting
        # list_of_plot_files = sorted(
        #     FINAL_DIR.glob("final_plot_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        # )
        # if not list_of_plot_files:
        #     raise FileNotFoundError(f"No plot result files found in {FINAL_DIR}")
        # latest_plot_file = list_of_plot_files[0]
        # print(f"Loading latest plot results from: {latest_plot_file}")

        self.all_forecasts_data = load_json_to_dict(FINAL_DIR / "final_plot_results.json")
        print("  ✓ Plot data loaded successfully.")

        return cv_metrics_df

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
            setup_environment(self.timestamp)

            # We still need the data for context and plotting
            train_df, test_df, hist_exog_list, data_info = prepare_pipeline_data()
            self.pipeline_results["data_preparation_info"] = data_info

            # This info is needed for the report
            self.pipeline_results["auto_models_workflow_info"] = self._get_workflow_info(hist_exog_list)

            cv_metrics_df = self._load_results()

            print("\n📈 STEP 2: DISPLAYING TOP MODELS")
            print("-" * 40)
            top_model_names = display_top_models(
                cv_metrics_df, "Cross-Validation Model Rankings", top_n=10
            )

            print("\n🎨 STEP 3: CREATING VISUALIZATIONS")
            print("-" * 40)
            create_visualizations_step(
                train_df, test_df, self.all_forecasts_data, HORIZON
            )

            print("\n📝 STEP 4: GENERATING SUMMARY REPORT")
            print("-" * 40)
            summary_report_str = generate_summary_report(
                cv_metrics_df,
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
                "results_dataframe": cv_metrics_df,
                "top_models_list": top_model_names,
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
        print("\n📋 PIPELINE SUMMARY:")
        print(run_outcome["summary_report_text"])
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline execution failed: {run_outcome['error_message']}")
        sys.exit(1)


if __name__ == "__main__":
    main() 