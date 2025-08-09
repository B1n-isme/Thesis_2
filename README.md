# Bitcoin Forecasting Pipeline - Clean Version

A streamlined, production-ready pipeline for Bitcoin price forecasting using the Nixtla ecosystem (statsforecast, mlforecast, neuralforecast).

## 🎯 Pipeline Overview

This pipeline implements a comprehensive workflow for Bitcoin daily close price forecasting:

1. **Data Collection & Preprocessing** - Load and preprocess Bitcoin price data with exogenous features.
2. **Feature Selection** - Perform stability selection to derived new set of features for each horizon.
2. **Cross Validation** - Perform expanding window CV with Hyperparameter Tuning
3. **Model Comparison** - Compare models from all three frameworks using best parameters
4. **Final Evaluation** - Generate comprehensive reports and recommendations

## 🚀 Quick Start

### Prerequisites
```bash
pip install statsforecast mlforecast neuralforecast utilsforecast
pip install pandas numpy matplotlib plotly
```

### Validation Test
```bash
python test_pipeline.py
```

### Quick Test Run
```bash
python main_clean.py --fast-mode --skip-hpo
```

### Full Pipeline
```bash
python main_clean.py
```

## 📊 Usage Options

### Command Line Arguments

- `--fast-mode`: Use fewer models and samples for quick testing
- `--skip-hpo`: Skip hyperparameter optimization step

### Examples

```bash
# Quick testing (3 models, 3 samples per model, skip HPO)
python main_clean.py --fast-mode --skip-hpo

# Fast with HPO (fewer models but with optimization)
python main_clean.py --fast-mode

# Full pipeline (all models, full HPO)
python main_clean.py

# Help
python main_clean.py --help
```

## 🏗️ Architecture

### Cleaned Structure

```
├── main_clean.py                    # Single entry point
├── test_pipeline.py                 # Validation tests
├── config/
│   └── base.py                      # Centralized configuration
├── src/
│   ├── models/
│   │   └── model_registry.py        # Unified model definitions
│   ├── dataset/
│   │   └── data_preparation.py      # Data loading and preprocessing
│   ├── hpo/
│   │   └── hyperparameter_tuning.py # HPO with auto models
│   ├── pipelines/
│   │   └── model_selection.py       # Model comparison framework
│   └── utils/
│       ├── other.py                 # Utilities
│       ├── plot.py                  # Visualization
│       └── hyperparam_loader.py     # Configuration loading
└── results/                         # Output directory
    ├── hpo/                         # HPO results
    ├── cv/                          # Cross-validation results
    ├── final/                       # Final model results
    └── models/                      # Saved models
```

### Key Improvements

1. **Single Entry Point**: Consolidated `main.py` and `main2.py` into `main_clean.py`
2. **Unified Model Registry**: All models from three frameworks in one place
3. **Consistent Configuration**: Centralized config with proper imports
4. **Clear Pipeline Steps**: Well-defined workflow with progress tracking
5. **Fast Mode Support**: Quick testing with reduced models/samples
6. **Error Handling**: Robust error handling with meaningful messages
7. **Comprehensive Logging**: Detailed progress and result reporting

## 📈 Model Registry

The `ModelRegistry` class provides unified access to all models:

### Statistical Models (statsforecast)
- **Primary**: AutoARIMA, AutoETS (multiple configurations), AutoTheta
- **Secondary**: AutoCES, RandomWalkWithDrift, WindowAverage, SeasonalNaive
- **Fast Mode**: Top 3 essential models only

### Neural Models (neuralforecast)
- **Full Set**: NHITS, NBEATS, LSTM, TFT, GRU
- **Fast Mode**: NHITS, NBEATS, LSTM only
- **Optimized**: Bitcoin-specific hyperparameters

### Auto Models (for HPO)
- **Full Set**: AutoNHITS, AutoNBEATS, AutoLSTM, AutoTFT
- **Fast Mode**: AutoNHITS, AutoNBEATS, AutoLSTM only
- **Adaptive Samples**: 3 samples in fast mode, 10+ in full mode

## 🔧 Configuration

Key settings in `config/base.py`:

```python
# Forecasting
HORIZON = 7                    # 7-day forecast
TEST_LENGTH_MULTIPLIER = 5     # 35 days test set
FREQUENCY = 'D'                # Daily frequency

# Cross-validation
CV_N_WINDOWS = 5               # 5 CV folds
CV_STEP_SIZE = HORIZON         # Non-overlapping windows

# HPO
NUM_SAMPLES_PER_MODEL = 10     # Hyperparameter samples
```

## 📊 Output

The pipeline generates:

1. **HPO Results**: Best hyperparameters for each auto model
2. **CV Results**: Cross-validation performance metrics
3. **Model Comparison**: Ranked comparison of all models
4. **Final Report**: Summary with recommendations
5. **Visualizations**: Forecast plots and performance charts

### Result Files

```
results/
├── hpo/
│   └── best_hyperparameters.csv
├── cv/
│   └── model_comparison_YYYYMMDD_HHMMSS.csv
└── final/
    ├── final_results_YYYYMMDD_HHMMSS.json
    └── summary_report_YYYYMMDD_HHMMSS.txt
```

## 🎯 Workflow Steps

### Step 1: Data Preparation
- Load Bitcoin price data from `data/final/dataset.parquet`
- Extract exogenous features (SMA, volume, etc.)
- Split into train/test sets (respecting temporal order)
- Format for Nixtla frameworks

### Step 2: Hyperparameter Optimization (Optional)
- Use Auto Models with cross-validation
- Search optimal hyperparameters for each model type
- Save best configurations for later use

### Step 3: Model Comparison
- Train models using best hyperparameters (or defaults)
- Compare performance across all frameworks
- Calculate comprehensive metrics (MAE, RMSE, directional accuracy)

### Step 4: Final Evaluation
- Rank models by performance
- Generate ensemble recommendations
- Create summary report with insights

## 🚀 Performance Tips

### Fast Mode Benefits
- **3x faster** execution time
- Suitable for development and testing
- Covers essential model types

### Full Mode Benefits
- Comprehensive model coverage
- Thorough hyperparameter search
- Production-ready results

### Hardware Recommendations
- **CPU**: 8+ cores for parallel processing
- **RAM**: 16GB+ for large datasets
- **GPU**: Optional but accelerates neural models

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python test_pipeline.py` first
2. **Data Not Found**: Check `DATA_PATH` in config
3. **Memory Issues**: Use `--fast-mode` flag
4. **Ray Errors**: Reduce `RAY_NUM_CPUS` in config

### Debug Mode
```bash
python -u main_clean.py --fast-mode --skip-hpo > debug.log 2>&1
```

## 📝 Extending the Pipeline

### Adding New Models
1. Update `ModelRegistry` class in `src/models/model_registry.py`
2. Add model configurations for HPO
3. Test with `python test_pipeline.py`

### Custom Loss Functions
1. Add to `get_loss_functions()` in model registry
2. Update loss mapping in config

### New Evaluation Metrics
1. Extend `model_selection.py`
2. Add custom metric calculations

## 🎖️ Best Practices

1. **Always test first**: Use `--fast-mode --skip-hpo` for initial validation
2. **Version results**: Pipeline timestamps all outputs
3. **Monitor performance**: Check memory usage with large datasets
4. **Regular updates**: Retrain models with new data periodically
5. **Ensemble approaches**: Combine top models for better performance

## 📞 Support

For issues or questions:
1. Run validation tests: `python test_pipeline.py`
2. Check configuration: `config/base.py`
3. Review logs in results directory
4. Use fast mode for debugging

---

**Note**: This pipeline is optimized for Bitcoin forecasting but can be adapted for other time series by modifying the data preparation and model configurations.
