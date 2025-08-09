## Bitcoin Price Forecasting — Horizon-Aware Pipeline (Nixtla Ecosystem)

Robust, reproducible Bitcoin daily close-price forecasting across multiple horizons (7, 14, 30, 60, 90 days) using StatsForecast and NeuralForecast, with a rigorous feature-selection pipeline and expanding-window cross-validation.

### Badges

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Nixtla](https://img.shields.io/badge/Nixtla-StatsForecast%20%7C%20NeuralForecast-ff69b4)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458)
![Reproducibility](https://img.shields.io/badge/Reproducible-Seeds%20%7C%20Config-success)

### Why this repo

- Horizon-aware evaluation: each horizon is treated as a first-class scenario.
- Strong methodology: stability feature selection, multicollinearity control, log-return modeling with price back-transformation.
- Fully timestamped results and plots checked into `results/` for immediate inspection.

### High-level workflow

```mermaid
graph TD
  A[Load data + features] --> B[Feature selection (stability + multicollinearity)]
  B --> C[Cross-validation (expanding windows)]
  C --> D[Pick best models per horizon]
  D --> E[Final fit on full train + holdout evaluation]
  E --> F[Plots, metrics, insights]
```

### Key capabilities

- Feature selection
  - Stability selection across tree models (XGBoost, LightGBM, RandomForest)
  - Consensus-based selection + VIF/correlation pruning
- Modeling
  - Statistical: `AutoETS` (plus others available)
  - Neural: iTransformer, BiTCN, TFT (via NeuralForecast)
  - Optional MLForecast models wired but not primary focus
- Evaluation & metrics
  - Expanding-window CV, horizon = `HORIZON`, step = `CV_STEP_SIZE`
  - Metrics: MAE, RMSE, MASE, Directional Accuracy (DA), training time
  - Back-transform from log-returns to prices for fair comparison

---

## Thesis project details

### Overview

This repository supports the thesis: "Forecasting of Bitcoin Price: An analysis from time horizon perspective." The study examines how forecast performance degrades with horizon length and shows that the optimal model class depends on the horizon. A U-shaped relationship between model complexity and performance emerges across short (7/14 days), medium (30/60 days), and long (90 days) horizons.

### Data and preprocessing

- Period: 2017–2024, daily
- Inputs: BTC OHLCV, on-chain metrics, sentiment indicators, and cross-market features (e.g., equities, commodities, FX)
- Target: Price modeled as log-returns for stationarity; forecasts back-transformed to price
- Features: Category-aware transformations (retain oscillators/spreads raw; difference/log-return price-like features) and robust scaling
- Horizon-specific feature sets via stability selection + multicollinearity reduction

### Evaluation protocol

- Stage 1: Expanding-window cross-validation per horizon; select model/hyperparameters by lowest MASE
- Stage 2: Retrain on full training set; evaluate on a strictly unseen holdout period
- Concept drift observed between CV and holdout; results reported for both

### Key findings

- Short-term (7/14 days): TCN best captures high-frequency patterns and adapts well
- Mid-term (30/60 days): ETS is most robust and reliable for trend-dominated horizons
- Long-term (90 days): TFT wins by modeling long-range dependencies
- Concept drift impacts all models from CV to holdout; deep models adapt well short-term; ETS remains stable mid-term

### Implications and future work

- Practical: Build a portfolio of horizon-specialist models, not a single all-purpose model
- Methodological: Horizon-aware selection and two-stage evaluation are critical
- Future work: Add GARCH/state-space models, dynamic feature selection, and stronger HPO strategies

## Quick start

1) Install
```bash
pip install -r requirements.txt
```

2) Configure horizon and paths in `config/base.py` (e.g., `HORIZON`, `TEST_LENGTH_MULTIPLIER`).

3) Run feature selection (saves selected features parquet for the chosen horizon)
```bash
python src/pipelines/feature_selection.py \
  --tree_methods xgboost lightgbm random_forest \
  --min_consensus_level 2 \
  --handle_multicollinearity \
  --n_bootstrap 50 \
  --selection_threshold 0.6
```

4) Cross-validation (stats + neural)
```bash
python src/pipelines/model_evaluation.py
```

5) Final retrain + holdout evaluation + artifacts
```bash
python src/pipelines/model_forecasting.py
```

6) Plots
```bash
python src/pipelines/visualization.py
```

Optional insights summary across horizons
```bash
python insights.py
```

---

## Configuration

Set in `config/base.py`:

```python
HORIZON = 7                 # Try: 7, 14, 30, 60, 90
TEST_LENGTH_MULTIPLIER = 1  # Holdout length = HORIZON * multiplier
LEVELS = [95]               # PI levels (optional)

CV_N_WINDOWS = 30           # Number of expanding windows
CV_STEP_SIZE = HORIZON      # Non-overlapping windows

RAW_DATA_PATH = 'data/final/raw_dataset.parquet'
DATA_PATH = f'data/final/feature_selection_{HORIZON}_mc.parquet'

ENABLE_ROLLING_FORECAST = True
ROLLING_REFIT_FREQUENCY = 0

NUM_SAMPLES_PER_MODEL = 10  # For auto neural search
```

Data is prepared in `src/dataset/data_preparation.py`.
Feature transforms differenced/log-returns as appropriate and keeps oscillators in raw form. Target is modeled in log-returns and back-transformed for evaluation.

---

## Project structure (summarized)

```
.
├── config/
│   └── base.py
├── data/
│   ├── raw/               # source datasets and scraped files
│   ├── processed/         # curated CSVs (sentiment, CBBI, combined)
│   └── final/             # modeling parquet/CSV incl. raw_dataset.parquet
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_combine.ipynb
│   └── 03_processing.ipynb
├── src/
│   ├── dataset/
│   │   └── data_preparation.py
│   ├── pipelines/
│   │   ├── feature_selection.py
│   │   ├── model_evaluation.py
│   │   ├── model_forecasting.py
│   │   ├── visualization.py
│   │   ├── results_processing.py
│   │   └── feature_results/   # saved feature lists per horizon
│   ├── models/
│   │   ├── statsforecast/models.py
│   │   ├── neuralforecast/models.py
│   │   └── neuralforecast/auto_cfg.py
│   ├── utils/
│   │   └── utils.py
│   └── visualization/
│       ├── final_visualization.py
│       └── merge_cv_visualization.py
├── results/
│   ├── insights/           # metric vs horizon plots, degradation analyses
│   └── results_{7|14|30|60|90}d/
│       ├── cv/             # cv_df.csv, cv_metrics.csv, best_configurations*.yaml, plots/
│       └── final/          # final_plot_results*.json, metrics_results*.csv, plots/
├── main.py                 # analysis/reporting pipeline entry
├── insights.py             # cross-horizon insights
├── parquet2csv.py
├── README.md
└── requirements.txt
```

---

## What’s already computed in this repo

- Cross-validation outputs for 7/14/30/60/90-day horizons: see `results/results_*d/cv/`
- Final holdout results and unified plots per horizon: see `results/results_*d/final/`
- Insights (metric vs horizon trends, degradation analysis): `results/insights/`

Thesis winners per horizon (primary conclusions):
- 7 & 14 days: TCN
- 30 & 60 days: ETS
- 90 days: TFT

Notes:
- iTransformer was competitive at 60 days in this repo’s runs and appears in saved best configs.
- TCN/TFT forecasts are included in the final plots where available.

Note: best neural configs used for final runs are saved (per horizon) at `results/results_*d/cv/best_configurations_comparison_nf.yaml` and then loaded by `get_normal_neural_models` for final evaluation.

---

## Metrics & methodology

- Standard: MAE, RMSE, MASE (with train_df for scaling)
- Financial: Directional Accuracy (DA)
- Time: training_time gathered and summarized
- Back-transformation: predictions produced on log-returns are converted back to price using the last known train price and cumulative sum of returns for apples-to-apples comparison.

---

## Reproducing thesis results

For each horizon in `{7, 14, 30, 60, 90}`:
- Set `HORIZON` in `config/base.py`
- Run feature selection, CV, final retrain (see Quick start)
- Artifacts will be saved under `results/results_{HORIZON}d/`

Key files to inspect:
- `cv/cv_df.csv`, `cv/cv_metrics.csv`, `cv/best_configurations_comparison_nf.yaml`
- `final/final_plot_results*.json`, `final/metrics_results*.csv`, `final/plots/`

## Reproducibility

- Global seeds and deterministic configs in `config/base.py`
- All outputs timestamped; intermediate and final artifacts stored under `results/`

---

## Data

- Input: `data/final/raw_dataset.parquet`
- Feature-selected parquet exported to: `data/final/feature_selection_{HORIZON}_mc.parquet`

If needed, `parquet2csv.py` shows how to convert parquet to CSV.

---

## Acknowledgments

- Built on the excellent Nixtla ecosystem: `statsforecast`, `neuralforecast`, `utilsforecast`.
- Additional tooling: `pandas`, `numpy`, `matplotlib`, `optuna`, `ray[tune]` (for auto neural).

---

## Citation

If you use this repository or its methodology, please reference the Nixtla libraries and this repository.
