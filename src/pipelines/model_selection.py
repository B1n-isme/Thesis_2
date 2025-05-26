# src/pipelines/model_selection.py
"""
Comprehensive model selection and comparison framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path

from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mape, smape

class ModelComparison:
    """Comprehensive model comparison across different frameworks."""
    
    def __init__(self, results_dir: str = "model_comparison_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        
    def evaluate_neural_models(self, models, train_df, horizon: int) -> List[Dict]:
        """Evaluate neural models with cross-validation, fallback to forecast if CV fails."""
        results = []
        
        for model in models:
            print(f"Evaluating {model.__class__.__name__}...")
            start_time = time.time()
            
            try:
                # Create NeuralForecast instance
                nf = NeuralForecast(models=[model], freq='D')
                
                # Try cross-validation first (uses train_df internally)
                try:
                    cv_results = nf.cross_validation(
                        df=train_df,
                        n_windows=3,  # Reduced for faster evaluation
                        step_size=horizon
                    )
                    
                    # Calculate metrics from CV results
                    metrics = self._calculate_metrics(cv_results, [model.__class__.__name__])
                    evaluation_method = "cross_validation"
                    
                except Exception as cv_error:
                    print(f"  Cross-validation failed for {model.__class__.__name__}: {str(cv_error)}")
                    print(f"  Falling back to train/forecast method...")
                    
                    # Fallback: Manual split for train/forecast
                    val_size = min(len(train_df) // 4, horizon * 3)
                    df_train = train_df.iloc[:-val_size].copy()
                    df_test = train_df.iloc[-val_size:].copy()
                    
                    # Train on df_train and forecast on df_test
                    nf.fit(df_train)
                    forecasts = nf.predict(futr_df=df_test[['unique_id', 'ds']])
                    
                    # Merge with actual values for evaluation
                    eval_df = df_test.merge(forecasts, on=['unique_id', 'ds'], how='inner')
                    
                    # Calculate metrics from forecast results
                    metrics = self._calculate_metrics(eval_df, [model.__class__.__name__])
                    evaluation_method = "train_forecast"
                
                # Training time
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model.__class__.__name__,
                    'framework': 'neuralforecast',
                    'training_time': training_time,
                    'evaluation_method': evaluation_method,
                    'is_auto': 'Auto' in model.__class__.__name__,
                    **metrics[model.__class__.__name__]
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {model.__class__.__name__}: {str(e)}")
                results.append({
                    'model_name': model.__class__.__name__,
                    'framework': 'neuralforecast',
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def evaluate_statistical_models(self, models, train_df, horizon: int) -> List[Dict]:
        """Evaluate statistical models."""
        results = []
        
        # Prepare data for StatsForecast
        df_stats = train_df[['unique_id', 'ds', 'y']].copy()
        
        for model in models:
            print(f"Evaluating {model.__class__.__name__}...")
            start_time = time.time()
            
            try:
                # Create StatsForecast instance
                sf = StatsForecast(models=[model], freq='D', verbose=True)
                
                # Cross-validation
                cv_results = sf.cross_validation(
                    df=df_stats,
                    h=horizon,
                    n_windows=3,
                    step_size=horizon
                )
                
                # Calculate metrics
                metrics = self._calculate_metrics(cv_results, [model.__class__.__name__])
                
                # Training time
                training_time = time.time() - start_time
                
                result = {
                    'model_name': model.__class__.__name__,
                    'framework': 'statsforecast',
                    'training_time': training_time,
                    'is_auto': 'Auto' in model.__class__.__name__,
                    **metrics[model.__class__.__name__]
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {model.__class__.__name__}: {str(e)}")
                results.append({
                    'model_name': model.__class__.__name__,
                    'framework': 'statsforecast',
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def _calculate_metrics(self, cv_results: pd.DataFrame, model_names: List[str]) -> Dict:
        """Calculate comprehensive metrics for cross-validation results."""
        metrics = {}
        
        for model_name in model_names:
            if model_name not in cv_results.columns:
                continue
                
            # Basic metrics
            model_metrics = evaluate(
                cv_results,
                metrics=[mae, mse, rmse, mape, smape],
                models=[model_name]
            )
            
            # Additional custom metrics
            y_true = cv_results['y'].values
            y_pred = cv_results[model_name].values
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                # Directional accuracy
                directional_accuracy = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
                
                # Prediction intervals coverage (if available)
                coverage = self._calculate_coverage(cv_results, model_name)
                
                metrics[model_name] = {
                    'mae': model_metrics[model_metrics['metric'] == 'mae'][model_name].iloc[0],
                    'mse': model_metrics[model_metrics['metric'] == 'mse'][model_name].iloc[0],
                    'rmse': model_metrics[model_metrics['metric'] == 'rmse'][model_name].iloc[0],
                    'mape': model_metrics[model_metrics['metric'] == 'mape'][model_name].iloc[0],
                    'smape': model_metrics[model_metrics['metric'] == 'smape'][model_name].iloc[0],
                    'directional_accuracy': directional_accuracy,
                    'coverage_80': coverage.get('80', None),
                    'coverage_90': coverage.get('90', None),
                    'n_predictions': len(y_true_clean)
                }
            else:
                metrics[model_name] = {'error': 'No valid predictions'}
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (correct prediction of up/down movement)."""
        if len(y_true) < 2:
            return np.nan
            
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction)
    
    def _calculate_coverage(self, cv_results: pd.DataFrame, model_name: str) -> Dict:
        """Calculate prediction interval coverage."""
        coverage = {}
        
        # Check for prediction intervals
        for level in [80, 90]:
            lo_col = f"{model_name}-lo-{level}"
            hi_col = f"{model_name}-hi-{level}"
            
            if lo_col in cv_results.columns and hi_col in cv_results.columns:
                y_true = cv_results['y']
                y_lo = cv_results[lo_col]
                y_hi = cv_results[hi_col]
                
                # Coverage = proportion of true values within prediction intervals
                within_interval = (y_true >= y_lo) & (y_true <= y_hi)
                coverage[str(level)] = within_interval.mean()
        
        return coverage
    
    def compare_all_models(self, neural_models, stat_models, train_df, horizon: int) -> pd.DataFrame:
        """Compare all models and return ranked results."""
        
        print("=== Starting Model Comparison ===")
        all_results = []
        
        # Evaluate neural models
        if neural_models:
            print("\n--- Evaluating Neural Models ---")
            neural_results = self.evaluate_neural_models(neural_models, train_df, horizon)
            all_results.extend(neural_results)
        
        # Evaluate statistical models
        if stat_models:
            print("\n--- Evaluating Statistical Models ---")
            stat_results = self.evaluate_statistical_models(stat_models, train_df, horizon)
            all_results.extend(stat_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Filter out failed models - fix indexing error
        if 'error' in results_df.columns:
            successful_results = results_df[results_df['error'].isna()].copy()
        else:
            successful_results = results_df.copy()
        
        if len(successful_results) > 0:
            # Rank models by MAE (lower is better)
            successful_results['mae_rank'] = successful_results['mae'].rank()
            successful_results['rmse_rank'] = successful_results['rmse'].rank()
            successful_results['mape_rank'] = successful_results['mape'].rank()
            
            # Combined rank (simple average)
            successful_results['combined_rank'] = (
                successful_results['mae_rank'] + 
                successful_results['rmse_rank'] + 
                successful_results['mape_rank']
            ) / 3
            
            # Sort by combined rank
            successful_results = successful_results.sort_values('combined_rank')
        
        # Save results
        self.save_results(results_df, successful_results)
        
        return successful_results
    
    def save_results(self, all_results: pd.DataFrame, successful_results: pd.DataFrame):
        """Save comparison results."""
        timestamp = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        all_results.to_csv(self.results_dir / f"all_results_{timestamp}.csv", index=False)
        
        # Save successful results
        successful_results.to_csv(self.results_dir / f"ranked_results_{timestamp}.csv", index=False)
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_models': len(all_results),
            'successful_models': len(successful_results),
            'failed_models': len(all_results) - len(successful_results),
            'best_model': successful_results.iloc[0]['model_name'] if len(successful_results) > 0 else None,
            'best_mae': successful_results.iloc[0]['mae'] if len(successful_results) > 0 else None
        }
        
        with open(self.results_dir / f"summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {self.results_dir}")
        print(f"Summary: {summary}")
    
    def get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Get top N model names for ensemble or further analysis."""
        if len(results_df) == 0:
            return []
        
        return results_df.head(top_n)['model_name'].tolist()

# Usage example
def run_comprehensive_model_selection(train_df, horizon: int = 7, season_length: int = 7):
    """Run comprehensive model selection pipeline."""
    
    # Import model functions
    from src.models.model_registry import ModelRegistry
    
    # Get models
    neural_models = ModelRegistry.get_auto_models(horizon, num_samples_per_model=5)  # Reduced for faster testing
    deterministic_models = ModelRegistry.get_neural_models(horizon)
    stat_models = ModelRegistry.get_statistical_models(season_length)
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Compare models
    results = comparison.compare_all_models(
        neural_models=neural_models + deterministic_models,
        stat_models=stat_models,
        train_df=train_df,
        horizon=horizon
    )
    
    # Get top models
    top_models = comparison.get_top_models(results, top_n=5)
    
    print("\n=== Top 5 Models ===")
    for i, model in enumerate(top_models, 1):
        row = results[results['model_name'] == model].iloc[0]
        print(f"{i}. {model} (MAE: {row['mae']:.4f}, Framework: {row['framework']})")
    
    return results, top_models