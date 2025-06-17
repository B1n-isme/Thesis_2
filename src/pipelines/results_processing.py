"""
Results processing module for the Bitcoin forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tabulate import tabulate

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


def _get_top_n_by_metric_lines(df: pd.DataFrame, metric: str, n: int = 3) -> List[str]:
    """Generates formatted lines for top N models by a given metric."""
    lines = []
    
    metric_df = df.dropna(subset=[metric])
    if metric_df.empty:
        lines.append(f"  No models with valid '{metric}' values to rank.")
        return lines

    top_n_df = metric_df.nsmallest(n, metric)
    
    for i, (_, row) in enumerate(top_n_df.iterrows(), 1):
        model_name = row['model_name']
        mae = row['mae']
        rmse = row['rmse'] 
        mape_str = f"{row['mape']:.4f}%" if pd.notna(row['mape']) else 'N/A'
        time_str = f"{row['training_time']:.2f}s" if pd.notna(row['training_time']) else 'N/A'
        
        lines.extend([
            f"{i:2d}. {model_name}",
            f"    MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape_str} | Time: {time_str}",
            ""
        ])
    return lines


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
        report_lines.extend([
            "📊 FULL MODEL PERFORMANCE METRICS",
            "-" * 50,
        ])
        
        table_df = valid_results[['model_name', 'mae', 'rmse', 'mape', 'training_time']].copy()
        table_df.columns = ['Model', 'MAE', 'RMSE', 'MAPE (%)', 'Time (s)']
        table_df['MAE'] = table_df['MAE'].map('{:.4f}'.format)
        table_df['RMSE'] = table_df['RMSE'].map('{:.4f}'.format)
        table_df['MAPE (%)'] = table_df['MAPE (%)'].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        table_df['Time (s)'] = table_df['Time (s)'].map(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
        
        report_lines.append(tabulate(table_df, headers='keys', tablefmt='psql', showindex=False))
        report_lines.append("\n")

        report_lines.extend([
            "🏆 TOP 3 MODELS BY MAE (lower is better)",
            "-" * 50,
        ])
        report_lines.extend(_get_top_n_by_metric_lines(valid_results, 'mae', 3))

        report_lines.extend([
            "🏆 TOP 3 MODELS BY RMSE (lower is better)",
            "-" * 50,
        ])
        report_lines.extend(_get_top_n_by_metric_lines(valid_results, 'rmse', 3))

        report_lines.extend([
            "🏆 TOP 3 MODELS BY MAPE (lower is better)",
            "-" * 50,
        ])
        report_lines.extend(_get_top_n_by_metric_lines(valid_results, 'mape', 3))
        
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