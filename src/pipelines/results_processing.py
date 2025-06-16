"""
Results processing module for the Bitcoin forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from config.base import HORIZON
from src.utils.utils import get_horizon_directories


def display_top_models(metrics_df: pd.DataFrame, title: str) -> None:
    """Display top performing models based on MAE."""
    print(f"\n📊 {title}")
    print("=" * 50)
    
    # Remove any rows with NaN MAE values and sort by MAE (ascending - lower is better)
    valid_results = metrics_df.dropna(subset=['mae']).copy()
    if valid_results.empty:
        print("⚠️  No valid results to display")
        return
    
    # Sort by MAE (lower is better) and display all models
    ranked_models = valid_results.sort_values('mae')
    
    for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
        model_name = row['model_name']
        mae = row['mae']
        rmse = row['rmse']
        mape = row['mape'] if pd.notna(row['mape']) else 'N/A'
        training_time = row['training_time'] if pd.notna(row['training_time']) else 'N/A'
        
        print(f"{i:2d}. {model_name}")
        print(f"    MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape}% | Time: {training_time}s")
        print()


def generate_summary_report(
    metrics_df: pd.DataFrame,
    horizon: int,
    data_info: Dict,
    workflow_info: Dict,
    results_dir: Path
) -> str:
    """Generate a comprehensive summary report."""
    
    # Get dynamic directories for saving the report
    _, final_dir, _ = get_horizon_directories()
    
    # Filter valid results
    valid_results = metrics_df.dropna(subset=['mae']).copy()
    
    # Report content
    report_lines = [
        "🚀 BITCOIN FORECASTING PIPELINE - SUMMARY REPORT",
        "=" * 60,
        "",
        f"📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"🎯 Forecast Horizon: {horizon} days",
        f"📊 Dataset: {data_info.get('total_samples', 'N/A')} samples",
        f"🏋️  Training Set: {data_info.get('train_samples', 'N/A')} samples ({data_info.get('train_start', 'N/A')} to {data_info.get('train_end', 'N/A')})",
        f"🧪 Test Set: {data_info.get('test_samples', 'N/A')} samples ({data_info.get('test_start', 'N/A')} to {data_info.get('test_end', 'N/A')})",
        "",
        "🤖 MODEL INVENTORY",
        "-" * 40,
        f"📈 Statistical Models: {workflow_info.get('stat_models_count', 0)}",
        f"🤖 ML Models: {workflow_info.get('ml_models_count', 0)}",
        f"🧠 Neural Models: {workflow_info.get('neural_models_count', 0)}",
        f"🎯 Total Models Evaluated: {workflow_info.get('total_models', 0)}",
        "",
    ]
    
    if not valid_results.empty:
        # Sort by MAE
        ranked_models = valid_results.sort_values('mae')
        
        report_lines.extend([
            "🏆 MODEL PERFORMANCE RANKING (by MAE)",
            "-" * 40,
        ])
        
        for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
            model_name = row['model_name']
            mae = row['mae']
            rmse = row['rmse'] 
            mape = row['mape'] if pd.notna(row['mape']) else 'N/A'
            training_time = row['training_time'] if pd.notna(row['training_time']) else 'N/A'
            
            report_lines.extend([
                f"{i:2d}. {model_name}",
                f"    MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape}% | Time: {training_time}s",
                ""
            ])
        
        # Overall statistics
        successful_models = len(valid_results)
        best_mae = valid_results['mae'].min()
        worst_mae = valid_results['mae'].max()
        avg_mae = valid_results['mae'].mean()
        
        report_lines.extend([
            "",
            "📊 PERFORMANCE SUMMARY",
            "-" * 40,
            f"✅ Successful Models: {successful_models}/{len(metrics_df)}",
            f"🥇 Best MAE: {best_mae:.4f}",
            f"🥉 Worst MAE: {worst_mae:.4f}",
            f"📊 Average MAE: {avg_mae:.4f}",
            "",
        ])
    else:
        report_lines.extend([
            "⚠️  No valid model results available for analysis.",
            "",
        ])
    
    # Join all lines
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = final_dir / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"📋 Summary report saved to: {report_path}")
    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)
    
    return report_text 