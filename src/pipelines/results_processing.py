"""
Module for processing and reporting results in the Auto Models workflow.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from datetime import datetime

def display_top_models(results_df: pd.DataFrame, title: str = "Top Models", top_n: int = 5) -> List[str]:
    """Display top models from cross-validation results in a consistent format."""
    if results_df.empty:
        print(f"❌ No results available for {title.lower()}")
        return []
        
    successful = results_df[results_df['status'] == 'success']
    if successful.empty:
        print(f"❌ No successful models for {title.lower()}")
        return []
        
    # Convert MAE to numeric and sort by MAE (ascending - lower is better)
    successful = successful.copy()
    successful['mae'] = pd.to_numeric(successful['mae'], errors='coerce')
    successful['rmse'] = pd.to_numeric(successful['rmse'], errors='coerce')
    
    # Remove rows where MAE conversion failed
    successful = successful.dropna(subset=['mae'])
    
    successful_sorted = successful.sort_values('mae', ascending=True)
    
    actual_top_n = min(top_n, len(successful_sorted))
    top_models_df = successful_sorted.head(actual_top_n)
    
    print(f"\n🏆 {title} (by CV MAE):")
    print("-" * 60)
    for i, (_, row) in enumerate(top_models_df.iterrows(), 1):
        print(f"{i:2d}. {row['model_name']}")
        print(f"     CV MAE: {row.get('mae', np.nan):.4f} | RMSE: {row.get('rmse', np.nan):.4f}")
        print(f"     Training Time: {row.get('training_time', 0):.2f}s")
        print()
        
    return top_models_df['model_name'].tolist()

def generate_summary_report(
    comparison_results: pd.DataFrame, 
    horizon: int,
    data_info: Dict, 
    auto_models_info: Dict, 
    results_dir: Path
) -> str:
    """Generate comprehensive summary report based on cross-validation results."""
    if comparison_results.empty:
        successful_results = pd.DataFrame()
    else:
        successful_results = comparison_results[comparison_results['status'] == 'success']
        
    report = [
        "=" * 80, "🚀 BITCOIN AUTO MODELS PIPELINE - SUMMARY REPORT", "=" * 80,
        f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        "🏆 MODEL PERFORMANCE SUMMARY (CROSS-VALIDATION)", "-" * 40
    ]
    
    if not successful_results.empty:
        # Convert to numeric and sort by MAE
        successful_results = successful_results.copy()
        successful_results['mae'] = pd.to_numeric(successful_results['mae'], errors='coerce')
        successful_results['rmse'] = pd.to_numeric(successful_results['rmse'], errors='coerce')
        successful_results = successful_results.dropna(subset=['mae'])
        successful_results = successful_results.sort_values('mae', ascending=True)
        
        top_5 = successful_results.head(5)
        report.append("Top 5 Models (by CV MAE):")
        report.append("")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            report.extend([
                f"{i}. {row['model_name']}",
                f"   CV MAE: {row.get('mae', np.nan):.4f} | RMSE: {row.get('rmse', np.nan):.4f}", ""
            ])
        report.extend([
            "Performance Statistics (CV):",
            f"Best CV MAE: {successful_results['mae'].min():.4f}",
            f"Best CV RMSE: {successful_results['rmse'].min():.4f}",
            f"Average CV MAE: {successful_results['mae'].mean():.4f}",
            f"Average CV RMSE: {successful_results['rmse'].mean():.4f}"
        ])
    else:
        report.append("❌ No successful model results available from cross-validation.")
        
    report.append("")
    report.extend(["📈 SUCCESS METRICS", "-" * 40])
    total_models = len(comparison_results)
    successful_models_count = len(successful_results)
    success_rate = (successful_models_count / total_models * 100) if total_models > 0 else 0
    
    report.extend([
        f"Total Models Evaluated: {total_models}",
        f"Successful Models (CV): {successful_models_count}",
        f"Success Rate: {success_rate:.1f}%"
    ])
    
    report.extend(["", "=" * 80])
    report_text = "\n".join(report)
    report_path = results_dir / f'report_{HORIZON}d.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"📄 Summary report saved: {report_path}")
    return report_text 