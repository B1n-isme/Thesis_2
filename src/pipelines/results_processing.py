"""
Module for processing and reporting results in the Auto Models workflow.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path

def combine_results(cv_results: pd.DataFrame, final_results: pd.DataFrame) -> pd.DataFrame:
    """Combine cross-validation and final results."""
    combined = cv_results.merge(
        final_results, 
        on=['model_name', 'framework'], 
        how='outer',
        suffixes=('_cv', '_test')
    )
    combined['is_auto'] = combined['is_auto_cv'].fillna(combined['is_auto_test'])
    combined['status_cv'] = combined['status_cv'].fillna('not_run')
    combined['status_test'] = combined['status_test'].fillna('not_run')
    combined = combined.sort_values('mae_test', ascending=True)
    return combined

def display_top_models(results_df: pd.DataFrame, title: str = "Top Models", top_n: int = 5) -> List[str]:
    """Display top models in a consistent format."""
    if len(results_df) == 0:
        print(f"❌ No results available for {title.lower()}")
        return []
    successful = results_df[results_df['status_test'] == 'success']
    if len(successful) == 0:
        print(f"❌ No successful models for {title.lower()}")
        return []
    actual_top_n = min(top_n, len(successful))
    top_models_df = successful.head(actual_top_n)
    print(f"\n🏆 {title} (by Test MAE):")
    print("-" * 60)
    for i, (_, row) in enumerate(top_models_df.iterrows(), 1):
        auto_indicator = " [AUTO]" if row.get('is_auto', False) else ""
        print(f"{i:2d}. {row['model_name']}{auto_indicator}")
        print(f"     Framework: {row['framework']}")
        print(f"     Test MAE: {row.get('mae_test', np.nan):.4f} | RMSE: {row.get('rmse_test', np.nan):.4f}")
        print(f"     Training Time: {row.get('training_time_test', 0):.2f}s")
        print()
    return top_models_df['model_name'].tolist()

def generate_summary_report(
    comparison_results: pd.DataFrame, 
    pipeline_timestamp: pd.Timestamp,
    data_info: Dict, 
    auto_models_info: Dict, 
    results_dir: Path
) -> str:
    """Generate comprehensive summary report."""
    successful_results = comparison_results[comparison_results['status_test'] == 'success']
    report = [
        "=" * 80, "🚀 BITCOIN AUTO MODELS PIPELINE - SUMMARY REPORT", "=" * 80,
        f"Execution Time: {pipeline_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pipeline: Direct Auto Models Workflow (No Separate HPO)", "",
        "📊 DATA INFORMATION", "-" * 40,
        f"Training Samples: {data_info.get('train_samples', 'N/A'):,}",
        f"Test Samples: {data_info.get('test_samples', 'N/A'):,}",
        f"Forecast Horizon: {data_info.get('horizon', 'N/A')} days",
        f"Exogenous Features: {len(data_info.get('features', []))}", "",
        "🤖 AUTO MODELS WORKFLOW", "-" * 40,
        f"Total Execution Time: {auto_models_info.get('execution_time', 0):.1f} seconds",
        f"Auto Neural Models: {auto_models_info.get('auto_models_count', 0)}",
        f"Statistical Models: {auto_models_info.get('stat_models_count', 0)}",
        f"Total Models: {auto_models_info.get('total_models', 0)}", "",
        "🏆 MODEL PERFORMANCE SUMMARY", "-" * 40
    ]
    if len(successful_results) > 0:
        top_5 = successful_results.head(5)
        report.append("Top 5 Models (by Test MAE):")
        report.append("")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            auto_indicator = " [AUTO]" if row.get('is_auto', False) else ""
            report.extend([
                f"{i}. {row['model_name']}{auto_indicator} ({row['framework']})",
                f"   Test MAE: {row.get('mae_test', np.nan):.4f} | RMSE: {row.get('rmse_test', np.nan):.4f}", ""
            ])
        report.extend([
            "Performance Statistics:",
            f"Best Test MAE: {successful_results['mae_test'].min():.4f}",
            f"Best Test RMSE: {successful_results['rmse_test'].min():.4f}",
            f"Average Test MAE: {successful_results['mae_test'].mean():.4f}",
            f"Average Test RMSE: {successful_results['rmse_test'].mean():.4f}"
        ])
    else:
        report.append("❌ No successful model results available")
    report.append("")
    report.extend(["📈 SUCCESS METRICS", "-" * 40])
    total_models = len(comparison_results)
    successful_models_count = len(successful_results)
    success_rate = (successful_models_count / total_models * 100) if total_models > 0 else 0
    report.extend([
        f"Total Models Evaluated: {total_models}",
        f"Successful Models: {successful_models_count}",
        f"Success Rate: {success_rate:.1f}%"
    ])
    if len(successful_results) > 0:
        report.extend([
            "", "Successful Models by Framework:",
            f"• Neural Models: {len(successful_results[successful_results['framework'] == 'neuralforecast'])}",
            f"• Statistical Models: {len(successful_results[successful_results['framework'] == 'statsforecast'])}",
            f"• Auto Models: {len(successful_results[successful_results['is_auto'] == True])}"
        ])
    report.extend(["", "=" * 80])
    report_text = "\n".join(report)
    report_path = results_dir / f'auto_pipeline_report_{pipeline_timestamp.strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"📄 Summary report saved: {report_path}")
    return report_text 