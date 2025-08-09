This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been compressed (code blocks are separated by ⋮---- delimiter).

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: **/*.csv, **/*.json, **/*.ipynb, **/__pycache__/, **/lightning_logs/**, **/.gitignore, **/*.txt, **/*.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.cursor/
  rules/
    ai-engineer.mdc
    data-scientist.mdc
config/
  base.py
results/
  results_14d/
    cv/
      best_configurations_comparison_nf.yaml
  results_30d/
    cv/
      best_configurations_comparison_nf.yaml
  results_60d/
    cv/
      best_configurations_comparison_nf.yaml
  results_7d/
    cv/
      best_configurations_comparison_nf.yaml
  results_90d/
    cv/
      best_configurations_comparison_nf.yaml
src/
  dataset/
    data_preparation.py
  metrics_calculation/
    add_training_times.py
    evaluate_final_forecast.py
    merge_cv_df_metrics.py
    metric.py
  models/
    mlforecast/
      auto_cfg.py
      models.py
    neuralforecast/
      auto_cfg.py
      models.py
    statsforecast/
      models.py
  pipelines/
    feature_selection_pipeline/
      __init__.py
      base.py
      feature_selection_diagram.mmd
      robust_selection.py
    feature_selection.py
    model_evaluation.py
    model_forecasting.py
    results_processing.py
    visualization.py
  scraping/
    engineer.py
    fngindex.py
    scraper.py
  utils/
    utils.py
  visualization/
    final_visualization.py
    merge_cv_visualization.py
.repomixignore
insights.py
main.py
parquet2csv.py
pipeline.py
```

# Files

## File: .cursor/rules/ai-engineer.mdc
```
---
description: 
globs: 
alwaysApply: false
---
---
description: Comprehensive guidelines for generating high-quality, robust, and maintainable Python code for Machine Learning (ML) and Deep Learning (DL) tasks within the PyTorch ecosystem, adhering to best practices, PEP 8 (enforced by Ruff), strict type hinting, and Google-style docstrings, as guided by a Senior AI/ML Engineer and Python Mentor.
globs: *.py
---

## Role Definition & Core Objective

You are a Senior AI/ML Engineer and an expert Python Mentor, operating within the Cursor IDE.

You embody Python mastery, with deep knowledge of best practices, design patterns, and efficient, maintainable code.

Your primary objective is to guide a junior engineer by generating high-quality Python code specifically for Machine Learning (ML) and Deep Learning (DL) tasks.

You explain complex concepts clearly and provide rationale for design decisions, fostering the junior's growth.

All outputs must be robust, efficient, maintainable, and adhere to the highest industry standards, leveraging user's existing codebase context when available.

## Core Technology Stack & Standards

* **Python Version:** Python 3.9+
* **Code Formatting & Linting:** Ruff (enforcing PEP 8, replacing black, isort, flake8).
* **Type Hinting:** Strict adherence using the `typing` module. All functions, methods, and variables where appropriate must have type annotations.
* **Documentation:** Google Python Style Guide for all docstrings.
* **Testing Framework:** `pytest` for examples and unit tests.
* **Core ML/DL Libraries:**
    * **PyTorch:** Primary framework for model definition and tensor operations.
    * **PyTorch Lightning:** For structured and streamlined training loops and best practices.
    * **NumPy & Pandas:** For efficient data manipulation and vectorized operations.
    * **Scikit-learn:** For classical ML tasks and utility functions.
* **Hyperparameter Optimization:** Optuna.
* **Experiment Tracking (Conceptual):** Familiarity with concepts like MLflow or TensorBoard to guide logging practices, even if not directly generating full configs.
* **Configuration:** YAML or Python dataclasses/Pydantic for managing experiment or model configurations.

## Coding Guidelines

### 1. Pythonic Excellence & Readability

* **PEP 8 Compliance:** Strictly enforced by Ruff.
* **Clarity & Simplicity:** Prioritize clear, explicit, and readable code. Avoid overly complex or obscure constructs.
* **Naming Conventions:** `snake_case` for functions, methods, and variables; `CamelCase` for classes. Use descriptive names.
* **Modularity:** Break down complex logic into smaller, single-responsibility functions or methods.

### 2. Robustness & Reliability

* **Comprehensive Type Annotations:** All function/method signatures and critical variables must be type-hinted.
* **Detailed Google-Style Docstrings:** Mandatory for every function, class, and method.
    * Must include `Args:`, `Returns:`, and `Raises:` sections with types and clear descriptions.
    * Include a concise summary of the object's purpose.
* **Specific Exception Handling:** Use specific exception types (e.g., `ValueError`, `TypeError`, `FileNotFoundError`). Avoid broad `except Exception:`. Provide informative error messages.
* **Input Validation:** Implement checks for data types, values, and shapes where critical.
* **Resource Management:** Use context managers (`with ...`) for files and other managed resources.

### 3. ML/DL Best Practices & Efficiency

* **Design for Reusability:** Create modular components (functions, classes) that can be reused.
* **Vectorization:** Leverage NumPy/Pandas vectorized operations over Python loops for performance-critical data tasks.
* **State Management:** Minimize global state; pass state explicitly as parameters.
* **Configuration Management:** Suggest YAML or dataclasses for managing hyperparameters and model configurations.
* **Reproducibility:** Emphasize practices that lead to reproducible results (e.g., setting random seeds, clear data processing steps).

### 4. Mentoring & Explanation

* **Rationale for Decisions:** Briefly explain significant design choices, library usage, or trade-offs (e.g., performance vs. readability, why a specific algorithm or pattern is chosen).
* **Clarity for Juniors:** Tailor explanations to be understandable by a junior engineer.
* **Constructive Guidance:** Frame suggestions positively, focusing on learning and improvement.

## Code Output Requirements

* **Complete & Executable Examples:** When generating non-trivial code, provide runnable snippets demonstrating usage.
* **pytest-style Unit Tests:** For key functions/methods, include simple example tests illustrating core functionality and edge cases. Frame these as demonstrations of testing best practices.
* **Ruff-Compliant Formatting:** All generated Python code must be formatted according to Ruff's defaults (which align with Black and PEP 8).
* **Contextual Integration:** Where possible, and if context from the Cursor environment is available, make examples and suggestions relevant to the user's existing code.
* **Comments for Rationale:** Use comments to explain the *why* behind complex or non-obvious logic, not just what the code does.

## Interaction Style

* **Clarification:** If a request is ambiguous or lacks detail, ask clarifying questions before generating code.
* **Iterative Refinement:** Be prepared to refine solutions based on feedback.
* **Focus on Task:** Prioritize the ML/DL task at hand, bringing in other software engineering principles as they support that core goal.
```

## File: .cursor/rules/data-scientist.mdc
```
---
description: 
globs: 
alwaysApply: false
---
You are a Senior Data Analyst mentoring a Junior Analyst in Cursor. Guide users through practical data analysis workflows, focusing on data understanding, sound analysis, correct interpretation, and clear communication of actionable insights.

**Rules:**

1.  **Step-by-Step Guidance & Rationale:**
    * Break tasks into logical steps. Explain the "why" behind each analytical choice.
    * Use clear language; explain technical terms if used.

2.  **Standard Tools & Readable Code:**
    * Python: Primarily Pandas (manipulation/cleaning) & Matplotlib/Seaborn (visualization).
    * SQL: Well-formatted queries; explain join/aggregation/filter logic.
    * Comment code for analytical goals.

3.  **Data Preparation:**
    * Guide data loading, handling missing values (explain strategies), type correction, and duplicate management.
    * Suggest simple, relevant feature engineering if applicable.

4.  **Exploratory Data Analysis (EDA):**
    * Demonstrate obtaining/interpreting descriptive statistics.
    * Show how to examine distributions and explore variable relationships.

5.  **Visualization & Interpretation:**
    * Recommend and code appropriate, well-labeled charts (titles, labels, legends), explaining choice.
    * **Always clearly interpret visualizations:** explain what the chart shows and key insights.

6.  **Translate Findings to Insights:**
    * Convert technical outputs (stats, visuals) into plain language insights.
    * Relate findings to the analytical goal; suggest next steps if relevant.

7.  **Foundational Techniques:**
    * Focus on standard, widely-used techniques suitable for junior analysts. Avoid undue complexity unless justified.
    * Directly address the user's specific task.

8.  **Cursor Context Integration:**
    * Leverage user's existing codebase/file context for relevant examples and analyses.
```

## File: src/metrics_calculation/add_training_times.py
```python
# CV_DIR, FINAL_DIR, PLOT_DIR = get_horizon_directories()
⋮----
# # Load the CSV files
# metrics_df = pd.read_csv(f'{FINAL_DIR}/metrics_results.csv')
# cv_df = pd.read_csv(f'{CV_DIR}/cv_metrics.csv')
⋮----
# # Check if both dataframes have the same number of rows
# if len(metrics_df) != len(cv_df):
#     print(f"Warning: Different number of rows - metrics_df: {len(metrics_df)}, cv_df: {len(cv_df)}")
# else:
#     print(f"Both files have {len(metrics_df)} rows")
⋮----
# # Add the training_time column row-wise
# metrics_df['training_time'] = metrics_df['training_time'] + cv_df['training_time']
⋮----
# # Save the updated dataframe
# output_file = f'{FINAL_DIR}/metrics_results.csv'
# metrics_df.to_csv(output_file, index=False)
⋮----
# print(f"Updated CSV saved to: {output_file}")
# print("\nFirst few rows of updated data:")
# print(metrics_df[['model_name', 'training_time']].head())
⋮----
metrics_df = pd.read_csv('results/results_7d/cv/cv_metrics.csv')
```

## File: src/metrics_calculation/evaluate_final_forecast.py
```python
def load_and_transform_final_results(json_path: Path) -> Tuple[pd.DataFrame, float, List[str]]
⋮----
"""
    Loads final forecast results from JSON and transforms them into a DataFrame.

    Args:
        json_path (Path): Path to the JSON file.

    Returns:
        Tuple[pd.DataFrame, float, List[str]]: A tuple containing the forecast DataFrame,
        the last value of the training set, and a list of model names.
    """
⋮----
data = json.load(f)
⋮----
df = pd.DataFrame({
⋮----
model_names = []
⋮----
# The Naive forecast for the first step is the last value of the training set.
y_last_train = data['models']['Naive']['predictions']['mean'][0]
⋮----
"""
    Calculates evaluation metrics for the final forecast using utilsforecast.

    Args:
        forecast_df (pd.DataFrame): DataFrame with forecasts.
        y_last_train (float): The last actual value from the training set.
        model_names (List[str]): A list of model names to evaluate.
        train_df (pd.DataFrame): The training dataframe for MASE calculation.

    Returns:
        pd.DataFrame: A DataFrame with the calculated metrics for each model.
    """
# Ensure 'unique_id' column exists for the evaluate function
⋮----
uid = train_df['unique_id'].unique()[0] if 'unique_id' in train_df.columns and train_df['unique_id'].nunique() > 0 else 'ts_1'
⋮----
# Use utilsforecast for standard metrics
evaluation_df = evaluate(
⋮----
# Transform results to wide format
metrics_df = evaluation_df.drop(columns=['unique_id']).set_index('metric').T
metrics_df = metrics_df.reset_index().rename(columns={'index': 'model_name'})
⋮----
# Calculate RMSE/MAE ratio
⋮----
# Prepare for and calculate financial metrics (Directional Accuracy)
⋮----
da_scores = {}
y_true = forecast_df['y'].values
y_lag1_vals = forecast_df['y_lag1'].values
⋮----
y_pred = forecast_df[model].values
financial_metrics = _calculate_financial_metrics(y_true, y_pred, y_lag1_vals)
⋮----
da_df = pd.DataFrame.from_dict(da_scores, orient='index', columns=['da'])
da_df = da_df.reset_index().rename(columns={'index': 'model_name'})
⋮----
# Merge standard and financial metrics
final_results_df = pd.merge(metrics_df, da_df, on='model_name')
⋮----
def main()
⋮----
"""Main function to execute the evaluation pipeline."""
# Define paths
results_dir = Path('results') / f'results_{HORIZON}d' / 'final'
json_path = results_dir / 'final_plot_results.json'
output_path = results_dir / 'metrics_results.csv'
⋮----
# Load forecast results
⋮----
# Load training data for MASE calculation
# We use `apply_transformations=False` to get the original values for the naive error calculation
⋮----
# Calculate metrics
metrics_df = calculate_final_metrics(forecast_df, y_last_train, model_names, train_df)
⋮----
# Reorder columns to be similar to CV results
metric_cols = ['mae', 'rmse', 'rmse_mae_ratio', 'mase', 'da']
ordered_cols = ['model_name'] + [col for col in metric_cols if col in metrics_df.columns]
other_cols = [col for col in metrics_df.columns if col not in ordered_cols]
final_df = metrics_df[ordered_cols + other_cols]
⋮----
# # Save results
# output_path.parent.mkdir(parents=True, exist_ok=True)
# final_df.to_csv(output_path, index=False)
```

## File: src/metrics_calculation/merge_cv_df_metrics.py
```python
# cv_dir, _, _ = get_horizon_directories()
cv_dir = 'results/results_90d/cv'
⋮----
# Read and merge all DataFrames
base_df = None
⋮----
df = pd.read_csv(f)
⋮----
base_df = df
⋮----
base_df = pd.merge(base_df, df.drop(columns=['y']),
⋮----
cols = [c for c in base_df.columns if not c.startswith('Auto') and not c.endswith('-95')] + \
# print(cols)
base_df = base_df[cols]
⋮----
# cv_df_nf = pd.read_csv(f"{cv_dir}/cv_df_0_20250620_202043.csv")
# cv_df_stats = pd.read_csv(f"{cv_dir}/cv_df_0_20250622_025002.csv")
# cv_metrics_nf = pd.read_csv(f"{cv_dir}/cv_metrics_nf.csv")
# cv_metrics_stats = pd.read_csv(f"{cv_dir}/cv_metrics_stats.csv")
⋮----
# Merge cv_df_nf and cv_df_stats on common columns, handling duplicate 'y' column
# cv_df = pd.merge(
#     cv_df_stats,
#     cv_df_nf.drop(columns=['y']),
#     on=["unique_id", "ds", "cutoff"],
#     how="outer"
# )
⋮----
# Concatenate cv_metrics_nf and cv_metrics_stats
# cv_metrics = pd.concat([cv_metrics_nf, cv_metrics_stats], ignore_index=True)
⋮----
# save the results
# cv_df.to_csv(f"{cv_dir}/cv_df.csv", index=False)
# cv_metrics.to_csv(f"{cv_dir}/cv_metrics.csv", index=False)
```

## File: src/metrics_calculation/metric.py
```python
def main()
⋮----
"""
    Reads cross-validation data, calculates metrics, and saves them to a CSV file.
    """
⋮----
# Define file paths
input_cv_path = f'{cv_dir}/cv_df.csv'
input_metrics_path = f'{cv_dir}/cv_metrics.csv'
output_metrics_path = f'{cv_dir}/cv_metrics_2.csv'
⋮----
# Read the cross-validation dataframe
⋮----
cv_df = pd.read_csv(input_cv_path)
⋮----
# Process the CV results to get metrics
metrics_df = process_cv_results(cv_df, train_df)
⋮----
# Merge training time from existing metrics
⋮----
existing_metrics = pd.read_csv(input_metrics_path)
metrics_df = metrics_df.merge(
⋮----
# Save the metrics to a new CSV file
```

## File: src/scraping/engineer.py
```python
def calculate_technical_indicators(df)
⋮----
"""Calculate technical indicators from Bitcoin price data."""
# Create a copy to avoid modifying original dataframe
data = df.copy()
⋮----
# Calculate multiple period SMAs
⋮----
# Calculate MA differences
⋮----
# Calculate MA ratios
⋮----
# Calculate MA slopes (5-period change)
⋮----
# Calculate price distance from MAs
⋮----
# Calculate RSI
⋮----
# Calculate MACD
macd = MACD(close=data['btc_close'])
⋮----
# Calculate Bollinger Bands
bb = BollingerBands(close=data['btc_close'], window=20, window_dev=2)
⋮----
def main()
⋮----
df = pd.read_csv('data/raw/dataset.csv')
df = calculate_technical_indicators(df)
```

## File: src/scraping/fngindex.py
```python
BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
START_DATE = '2022-03-01'
END_DATE = '2025-06-01'
ua = UserAgent()
⋮----
headers = {
⋮----
r = requests.get(BASE_URL + START_DATE, headers = headers)
data = r.json()
⋮----
# # store data as json
# with open('Fear_and_Greed_Index/fngindex.json', 'w') as f:
# 	json.dump(data, f)
⋮----
fng_data = pd.DataFrame({'Date': pd.date_range(start=START_DATE, end=END_DATE, freq='D')})
⋮----
x = int(data_point['x'])
x = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d')
y = int(data_point['y'])
rating = data_point.get('rating', 'unknown')  # Default to 'unknown' if 'rating' is missing
⋮----
#currently any days that do not have data points from cnn are filled with zeros, uncomment the following line to backfill
#fng_data['Fear Greed'].replace(to_replace=0, method='bfill')
```

## File: src/scraping/scraper.py
```python
# Constants
BASE_URL = 'https://api.blockchain.info/charts/'
BLOCKCHAIN_ENDPOINTS = {
FINAL_OUTPUT = "data/hello/dataset.csv"
⋮----
# # check if the path is reachable
⋮----
# Helper Functions
⋮----
def generate_intervals(start_date, days=30)
⋮----
"""Generate date intervals of a fixed number of days."""
end_date = datetime.now().date()
⋮----
interval_end = min(start_date + timedelta(days=days - 1), end_date)
⋮----
start_date = interval_end + timedelta(days=1)
⋮----
# Helper function for ARIMA-based imputation
def arima_impute(series)
⋮----
if series.isnull().sum() > 0 and series.notnull().sum() > 5:  # Ensure enough valid points
model = ARIMA(series, order=(1, 1, 1))  # Adjust (p, d, q) as needed
fitted_model = model.fit()
series = series.fillna(fitted_model.fittedvalues)
⋮----
def fetch_blockchain_data(start_date)
⋮----
"""Fetch blockchain metrics."""
all_data = {metric: [] for metric in BLOCKCHAIN_ENDPOINTS.keys()}
⋮----
params = {'start': start.isoformat(), 'end': end.isoformat(), 'format': 'json'}
url = f"{BASE_URL}{endpoint}"
response = requests.get(url, params=params)
⋮----
combined_df = pd.DataFrame()
⋮----
metric_df = pd.DataFrame(data)
⋮----
combined_df = pd.concat([combined_df, metric_df], ignore_index=True)
⋮----
# Convert 'x' from Unix timestamp to datetime in UTC
⋮----
# Calculate days since the first record
start_date = combined_df['Date'].min()
⋮----
# Aggregate duplicates by taking the mean of 'y' and keeping the first 'date'
combined_df = combined_df.groupby(['days_since_start', 'metric'], as_index=False).agg({'y': 'mean', 'Date': 'first'})
⋮----
# Pivot the dataset to spread metrics into separate columns
pivot_df = combined_df.pivot(index='days_since_start', columns='metric', values='y').reset_index()
⋮----
# Generate a complete range of days and merge with the pivoted DataFrame
all_days = pd.DataFrame({'days_since_start': range(combined_df['days_since_start'].max() + 1)})
pivot_df = all_days.merge(pivot_df, on='days_since_start', how='left')
⋮----
# Add the original datetime back to the pivoted DataFrame
⋮----
# Drop the 'days_since_start' column and reorder columns to make 'date' first
pivot_df = pivot_df.drop(columns=['days_since_start']).set_index('Date').reset_index()
⋮----
# Reorder columns for consistency
pivot_df = pivot_df[['Date', 'active_addresses', 'hash_rate', 'miner_revenue', 'difficulty', 'estimated_transaction_volume_usd']]
⋮----
# Optional: Rename columns for clarity
⋮----
def fetch_google_trends_data(keyword, start_date)
⋮----
"""Fetch Google Trends data."""
pytrends = TrendReq(hl='en-US', tz=360)
intervals = generate_intervals(start_date)
all_data = pd.DataFrame()
⋮----
data = pytrends.interest_over_time()
⋮----
data = data.drop(columns=['isPartial'], errors='ignore')
all_data = pd.concat([all_data, data])
⋮----
all_data = all_data.rename(columns={'bitcoin': 'google_trends_bitcoin'})
⋮----
def fetch_yfinance_data(tickers, start_date)
⋮----
"""Fetch data from Yahoo Finance."""
combined_data = pd.DataFrame()
⋮----
data = yf.download(ticker, start=start_date, end=datetime.now().date().isoformat())
⋮----
data = data[[f'{description}']]
combined_data = pd.concat([combined_data, data], axis=1, join='outer')
⋮----
# Step 1: Create a full date range for the combined data
full_date_range = pd.date_range(start=combined_data.index.min(), end=combined_data.index.max())
combined_data = combined_data.reindex(full_date_range)
⋮----
# # Step 2: Forward Fill for Edge Cases
# combined_data = combined_data.ffill()
⋮----
# # Step 3: Linear Interpolation for Small Gaps
# combined_data = combined_data.interpolate(method="linear")
⋮----
# # Step 4: ARIMA-Based Imputation for Large Gaps
# for column in combined_data.columns:
#     combined_data[column] = arima_impute(combined_data[column])
⋮----
# # Step 5: Final Fallback for Remaining Missing Values
# combined_data = combined_data.fillna(combined_data.mean())  # Fallback strategy
⋮----
def calculate_technical_indicators(start_date)
⋮----
"""Fetch historical BTC data and calculate technical indicators."""
btc_data = yf.download("BTC-USD", start=start_date, end=datetime.now().date().isoformat())
btc_data = btc_data.reset_index()
⋮----
data = btc_data[['Close']].copy()
⋮----
# Calculate multiple period SMAs and EMAs
⋮----
# Calculate MA differences and ratios
⋮----
# Calculate MA slopes
⋮----
# Calculate price distance from MAs
⋮----
# Calculate RSI
⋮----
# Calculate MACD
macd = MACD(close=data['btc_close'])
⋮----
# Calculate Bollinger Bands
bb = BollingerBands(close=data['btc_close'], window=20, window_dev=2)
⋮----
# Calculate ATR and other metrics
⋮----
# merge all data
def fetch_new_data(start_date)
⋮----
start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
⋮----
"""Merge data from all sources."""
# google_trends = fetch_google_trends_data("bitcoin", start_date)
# blockchain = fetch_blockchain_data(start_date)
tickers = {
yfinance_data = fetch_yfinance_data(tickers, start_date)
# technical_indicators = calculate_technical_indicators(start_date)
⋮----
# Merge all datasets on Date
all_data = [yfinance_data]
⋮----
# Ensure all 'Date' columns are timezone-aware in UTC
⋮----
# If already timezone-aware, convert to UTC
⋮----
# If timezone-naive, localize to UTC
⋮----
merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), all_data)
# btc_close = merged_df.pop('btc_close')
# merged_df['btc_close'] = btc_close
# # Remove the first 35 rows
# merged_df = merged_df.iloc[35:]
# # Impute missing values using forward fill method
# merged_df.ffill(inplace=True)
# # Impute remaining missing values using backward fill method
# merged_df.bfill(inplace=True)
⋮----
# Main Execution
⋮----
# start_date_str = input("Enter the start date (YYYY-MM-DD): ")
start_date = "2016-11-01"
⋮----
# Collect and merge data
final_data = fetch_new_data(start_date)
⋮----
# Save to CSV
```

## File: src/visualization/final_visualization.py
```python
"""
Script to generate final forecast visualizations from saved results.
"""
⋮----
def main()
⋮----
"""
    Main function to load forecast results and generate plots.
    """
⋮----
# Get directories based on current HORIZON configuration
⋮----
# Find the latest final_plot_results.json file
search_pattern = str(final_dir / "final_plot_results*.json")
list_of_files = glob.glob(search_pattern)
⋮----
latest_file = max(list_of_files, key=Path)
⋮----
# Load the forecast results data from the JSON file
⋮----
all_forecasts_data = load_json_to_dict(Path(latest_file))
⋮----
# Load train and test dataframes for plotting context.
# We use apply_transformations=False to ensure 'y' is in the original price scale for both dataframes.
⋮----
# Create the unified forecast plot
```

## File: src/visualization/merge_cv_visualization.py
```python
def create_grid_plot()
⋮----
horizons = [7, 14, 30, 60]
image_paths = []
⋮----
# Add ETS images
⋮----
# # Add TFT 90d image
# image_paths.append('results/results_90d/cv/plot/TFT_cv_90d.png')
⋮----
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, figure=fig) # 3 rows, 2 columns
⋮----
# Define the axes based on the GridSpec
axes = [
⋮----
fig.add_subplot(gs[0, 0]), # Row 0, Col 0
fig.add_subplot(gs[0, 1]), # Row 0, Col 1
fig.add_subplot(gs[1, 0]), # Row 1, Col 0
fig.add_subplot(gs[1, 1]), # Row 1, Col 1
# fig.add_subplot(gs[2, :])  # Row 2, spans both columns (centered)
⋮----
img = Image.open(img_path)
⋮----
# Extract information from the filename to create a more readable title
filename = img_path.split('/')[-1] # e.g., ETS_cv_7d.png
parts = filename.replace('.png', '').split('_') # e.g., ['ETS', 'cv', '7d']
model = parts[0].upper()
horizon = parts[-1].replace('d', '') # Remove 'd' for cleaner display
⋮----
title = f'{model} on {horizon}-days horizon'
⋮----
axes[i].set_title(title) # Set title to filename
axes[i].axis('off') # Hide axes
⋮----
# Adjust layout for better spacing
# plt.tight_layout()
```

## File: .repomixignore
```
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
```

## File: parquet2csv.py
```python
def parquet_to_csv(parquet_path, csv_path)
⋮----
"""
    Convert a .parquet file to a .csv file.
    
    Args:
        parquet_path (str): Path to the input .parquet file.
        csv_path (str): Path to save the output .csv file.
    """
df = pd.read_parquet(parquet_path)
⋮----
# Example usage
⋮----
parquet_path = "data/final/raw_dataset.parquet"
csv_path = "data/final/raw_dataset.csv"
```

## File: pipeline.py
```python
import pandas as pd# Load the training target dataset from the provided URL
⋮----
Y_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet')
⋮----
# Rename columns to match the Nixtlaverse's expectations
# The 'item_id' becomes 'unique_id' representing the unique identifier of the time series
# The 'timestamp' becomes 'ds' representing the time stamp of the data points
# The 'demand' becomes 'y' representing the target variable we want to forecast
Y_df = Y_df.rename(
⋮----
# Convert the 'ds' column to datetime format to ensure proper handling of date-related operations in subsequent steps
⋮----
Y_df = Y_df.query('unique_id.str.startswith("FOODS_3")').reset_index(drop=True)
⋮----
# Feature: plot random series for EDA
⋮----
# Import necessary models from the statsforecast library
⋮----
# SeasonalNaive: A model that uses the previous season's data as the forecast
⋮----
# Naive: A simple model that uses the last observed value as the forecast
⋮----
# HistoricAverage: This model uses the average of all historical data as the forecast
⋮----
# CrostonOptimized: A model specifically designed for intermittent demand forecasting
⋮----
# ADIDA: Adaptive combination of Intermittent Demand Approaches, a model designed for intermittent demand
⋮----
# IMAPA: Intermittent Multiplicative AutoRegressive Average, a model for intermittent series that incorporates autocorrelation
⋮----
# AutoETS: Automated Exponential Smoothing model that automatically selects the best Exponential Smoothing model based on AIC
⋮----
horizon = 28
models = [
⋮----
# Instantiate the StatsForecast class
sf = StatsForecast(
⋮----
models=models,  # A list of models to be used for forecasting
freq='D',  # The frequency of the time series data (in this case, 'D' stands for daily frequency)
n_jobs=-1,  # The number of CPU cores to use for parallel execution (-1 means use all available cores)
verbose=True,  # Show progress
⋮----
# Get the current time before forecasting starts, this will be used to measure the execution time
init = time()
⋮----
# Call the forecast method of the StatsForecast instance to predict the next 28 days (h=28)
fcst_df = sf.forecast(df=Y_df, h=28)
⋮----
# Get the current time after the forecasting ends
end = time()
⋮----
# Calculate and print the total time taken for the forecasting in minutes
⋮----
# Import the necessary models from various libraries
⋮----
# LGBMRegressor: A gradient boosting framework that uses tree-based learning algorithms from the LightGBM library
⋮----
# XGBRegressor: A gradient boosting regressor model from the XGBoost library
⋮----
# LinearRegression: A simple linear regression model from the scikit-learn library
⋮----
# Instantiate the MLForecast object
mlf = MLForecast(
⋮----
models=[LGBMRegressor(verbosity=-1), XGBRegressor(), LinearRegression()],  # List of models for forecasting: LightGBM, XGBoost and Linear Regression
freq='D',  # Frequency of the data - 'D' for daily frequency
lags=list(range(1, 7)),  # Specific lags to use as regressors: 1 to 6 days
⋮----
1: [ExpandingMean()],  # Apply expanding mean transformation to the lag of 1 day
⋮----
date_features=['year', 'month', 'day', 'dayofweek', 'quarter', 'week'],  # Date features to use as regressors
⋮----
# Start the timer to calculate the time taken for fitting the models
⋮----
# Fit the MLForecast models to the data
⋮----
# Calculate the end time after fitting the models
⋮----
# Print the time taken to fit the MLForecast models, in minutes
⋮----
fcst_mlf_df = mlf.predict(28)
⋮----
nf = NeuralForecast(
⋮----
fcst_nf_df = nf.predict()
⋮----
# Merge the forecasts from StatsForecast and NeuralForecast
fcst_df = fcst_df.merge(fcst_nf_df, how='left', on=['unique_id', 'ds'])
⋮----
# Merge the forecasts from MLForecast into the combined forecast dataframe
fcst_df = fcst_df.merge(fcst_mlf_df, how='left', on=['unique_id', 'ds'])
⋮----
# Cross-validation
# StatsForecast
⋮----
cv_df = sf.cross_validation(df=Y_df, h=horizon, n_windows=3, step_size=horizon)
⋮----
# MLForecast
⋮----
cv_mlf_df = mlf.cross_validation(
⋮----
# NeuralForecast
⋮----
cv_nf_df = nf.cross_validation(
⋮----
# Merge cross validation forecasts
cv_df = cv_df.merge(cv_nf_df.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
cv_df = cv_df.merge(cv_mlf_df.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
⋮----
agg_cv_df = cv_df.loc[:,~cv_df.columns.str.contains('hi|lo')].groupby(['ds', 'cutoff']).sum(numeric_only=True).reset_index()
⋮----
agg_Y_df = Y_df.groupby(['ds']).sum(numeric_only=True).reset_index()
⋮----
# Evaluation Metrics
⋮----
evaluation_df = evaluate(cv_df.drop(columns='cutoff'), metrics=[mse, mae, smape])
⋮----
# group value by metric
by_metric = evaluation_df.groupby('metric').mean(numeric_only=True)
⋮----
# Best models by metric
⋮----
# Choose best model
# Choose the best model for each time series, metric, and cross validation window
⋮----
# count how many times a model wins per metric and cross validation window
count_best_model = evaluation_df.groupby(['metric', 'best_model']).size().rename('n').to_frame().reset_index()
# plot results
```

## File: results/results_60d/cv/best_configurations_comparison_nf.yaml
```yaml
# AutoTSMixerx:
#   input_size: 120
#   learning_rate: 0.0001
#   scaler_type: robust
#   max_steps: 150
#   batch_size: 16
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 3
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_50
#   - btc_sma_14_50_ratio
#   - active_addresses_blockchain
#   - PiCycle_cbbi
#   - Fear Greed
#   - volume_sentiment
#   - Oil_Crude_Price
#   - Oil_Brent_Price
#   - Oil_Volatility
#   - CBOE_Volatility
#   - EURUSD
#   - MVRV_cbbi
#   - problem_malicious_sentiment
#   - Gold_Volatility
#   - Trolololo_cbbi
#   - ReserveRisk_cbbi
#   - bullish_sentiment
#   - marketcap_sentiment
#   - btc_ema_5
#   - btc_bb_low
#   - btc_bb_width
#   - btc_atr_14
#   - btc_volatility_index
#   - btc_trading_volume
#   - miner_revenue_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - RUPL_cbbi
#   - RHODL_cbbi
#   - DJI
#   - NYFANG
#   - EM_ETF
#   - DXY
#   - btc_sma_5
#   n_block: 1
#   ff_dim: 64
#   dropout: 0.13994847335768676
#   n_series: 1
AutoiTransformer:
  input_size: 360
  learning_rate: 0.0005
  scaler_type: robust
  max_steps: 50
  batch_size: 32
  windows_batch_size: 512
  val_check_steps: 50
  random_seed: 3
  accelerator: gpu
  logger: false
  hidden_size: 128
  n_heads: 8
  n_series: 1
# AutoBiTCN:
#   input_size: 180
#   learning_rate: 0.0001
#   scaler_type: robust
#   max_steps: 200
#   batch_size: 16
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 3
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_50
#   - btc_sma_14_50_ratio
#   - active_addresses_blockchain
#   - PiCycle_cbbi
#   - Fear Greed
#   - volume_sentiment
#   - Oil_Crude_Price
#   - Oil_Brent_Price
#   - Oil_Volatility
#   - CBOE_Volatility
#   - EURUSD
#   - MVRV_cbbi
#   - problem_malicious_sentiment
#   - Gold_Volatility
#   - Trolololo_cbbi
#   - ReserveRisk_cbbi
#   - bullish_sentiment
#   - marketcap_sentiment
#   - btc_ema_5
#   - btc_bb_low
#   - btc_bb_width
#   - btc_atr_14
#   - btc_volatility_index
#   - btc_trading_volume
#   - miner_revenue_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - RUPL_cbbi
#   - RHODL_cbbi
#   - DJI
#   - NYFANG
#   - EM_ETF
#   - DXY
#   - btc_sma_5
#   hidden_size: 32
#   dropout: 0.4273528774505862
# AutoTFT:
#   input_size: 180
#   learning_rate: 1.9589815480832375e-05
#   scaler_type: robust
#   max_steps: 200
#   batch_size: 32
#   windows_batch_size: 1024
#   val_check_steps: 50
#   random_seed: 19
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_50
#   - btc_sma_14_50_ratio
#   - active_addresses_blockchain
#   - PiCycle_cbbi
#   - Fear Greed
#   - volume_sentiment
#   - Oil_Crude_Price
#   - Oil_Brent_Price
#   - Oil_Volatility
#   - CBOE_Volatility
#   - EURUSD
#   - MVRV_cbbi
#   - problem_malicious_sentiment
#   - Gold_Volatility
#   - Trolololo_cbbi
#   - ReserveRisk_cbbi
#   - bullish_sentiment
#   - marketcap_sentiment
#   - btc_ema_5
#   - btc_bb_low
#   - btc_bb_width
#   - btc_atr_14
#   - btc_volatility_index
#   - btc_trading_volume
#   - miner_revenue_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - RUPL_cbbi
#   - RHODL_cbbi
#   - DJI
#   - NYFANG
#   - EM_ETF
#   - DXY
#   - btc_sma_5
#   hidden_size: 64
#   n_head: 8
```

## File: results/results_90d/cv/best_configurations_comparison_nf.yaml
```yaml
# AutoTSMixerx:
#   input_size: 180
#   learning_rate: 0.001
#   scaler_type: robust
#   max_steps: 200
#   batch_size: 64
#   windows_batch_size: 256
#   val_check_steps: 50
#   random_seed: 17
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_50_slope
#   - btc_atr_14
#   - ReserveRisk_cbbi
#   - Fear Greed
#   - bearish_sentiment
#   - market_narrative_sentiment
#   - Gold_Share
#   - Gold_Volatility
#   - Oil_Volatility
#   - DJI
#   - NYFANG
#   - CBOE_Volatility
#   - EM_ETF
#   - DXY
#   - EURUSD
#   - btc_volatility_index
#   - Confidence_cbbi
#   - regulations_sentiment
#   - Gold_Price
#   - Oil_Crude_Price
#   - Oil_Brent_Price
#   - btc_bb_low
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - miner_revenue_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - RHODL_cbbi
#   - MVRV_cbbi
#   - volume_sentiment
#   - btc_bb_width
#   - marketcap_sentiment
#   n_block: 2
#   ff_dim: 128
#   dropout: 0.5973598251776474
#   n_series: 1
# AutoiTransformer:
#   input_size: 270
#   learning_rate: 0.0005
#   scaler_type: robust
#   max_steps: 150
#   batch_size: 16
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 19
#   accelerator: gpu
#   logger: false
#   hidden_size: 256
#   n_heads: 8
#   n_series: 1
AutoBiTCN:
  input_size: 270
  learning_rate: 0.0005
  scaler_type: robust
  max_steps: 200
  batch_size: 16
  windows_batch_size: 1024
  val_check_steps: 50
  random_seed: 5
  accelerator: gpu
  logger: false
  hist_exog_list:
  - btc_sma_50_slope
  - btc_atr_14
  - ReserveRisk_cbbi
  - Fear Greed
  - bearish_sentiment
  - market_narrative_sentiment
  - Gold_Share
  - Gold_Volatility
  - Oil_Volatility
  - DJI
  - NYFANG
  - CBOE_Volatility
  - EM_ETF
  - DXY
  - EURUSD
  - btc_volatility_index
  - Confidence_cbbi
  - regulations_sentiment
  - Gold_Price
  - Oil_Crude_Price
  - Oil_Brent_Price
  - btc_bb_low
  - btc_trading_volume
  - active_addresses_blockchain
  - miner_revenue_blockchain
  - estimated_transaction_volume_usd_blockchain
  - RHODL_cbbi
  - MVRV_cbbi
  - volume_sentiment
  - btc_bb_width
  - marketcap_sentiment
  hidden_size: 16
  dropout: 0.21518228453047375
# AutoTFT:
#   input_size: 180
#   learning_rate: 0.0022488316385749995
#   scaler_type: robust
#   max_steps: 100
#   batch_size: 4
#   windows_batch_size: 128
#   val_check_steps: 50
#   random_seed: 2
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_50_slope
#   - btc_atr_14
#   - ReserveRisk_cbbi
#   - Fear Greed
#   - bearish_sentiment
#   - market_narrative_sentiment
#   - Gold_Share
#   - Gold_Volatility
#   - Oil_Volatility
#   - DJI
#   - NYFANG
#   - CBOE_Volatility
#   - EM_ETF
#   - DXY
#   - EURUSD
#   - btc_volatility_index
#   - Confidence_cbbi
#   - regulations_sentiment
#   - Gold_Price
#   - Oil_Crude_Price
#   - Oil_Brent_Price
#   - btc_bb_low
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - miner_revenue_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - RHODL_cbbi
#   - MVRV_cbbi
#   - volume_sentiment
#   - btc_bb_width
#   - marketcap_sentiment
#   hidden_size: 64
#   n_head: 4
```

## File: src/models/mlforecast/auto_cfg.py
```python
def lgb_auto_cfg(trial: optuna.Trial) -> dict
⋮----
"""
    Refined LightGBM hyperparameter configuration for volatile time series.
    """
⋮----
"verbosity": -1,  # Silent mode
⋮----
def xgb_auto_cfg(trial: optuna.Trial) -> dict
⋮----
"""
    Refined XGBoost hyperparameter configuration for volatile time series.
    """
⋮----
def cat_auto_cfg(trial: optuna.Trial) -> dict
⋮----
"""
    Refined CatBoost hyperparameter configuration for volatile time series.
    """
```

## File: src/pipelines/feature_selection_pipeline/__init__.py
```python
"""Feature Selection Pipeline Package."""
⋮----
__all__ = [
```

## File: src/pipelines/feature_selection_pipeline/feature_selection_diagram.mmd
```
graph TD
    subgraph "Stage 1: Stability Selection (on Stationary Target)"
        A["Start: Input Time Series DataFrame"] --> B["Prepare Data: Create Stationary Target (log-returns)"];
        B --> C{"For each tree model (XGB, LGBM, RF):<br>1. Run Stability Selection on bootstrap samples<br>2. Identify features that are consistently important"}
        C --> D["Output: Three separate lists of 'stable' features"];
    end

    D --> E{"Stage 2: Consensus Building"};
    subgraph "Consensus Stage"
        E --> F["Aggregate results from all models"];
        F --> G["Select features that pass a minimum consensus level (e.g., selected by >= 2 models)"];
        G --> H["Output: A single list of high-confidence consensus features"];
    end

    H --> I{"Stage 3: Multicollinearity Reduction"};
    subgraph "Filtering Stage"
        I --> J["<br>1. Cluster highly correlated features.
Remove less stable feature from each pair."];
        J --> K["<br>2. Iteratively remove features with high Variance Inflation Factor (VIF)"];
        K --> L["Output: A filtered, non-redundant set of features"];
    end

    %% L --> M{"Stage 4: Final Validation (on Original Target)"};
    %% subgraph "Validation Stage"
    %%     M --> N["Train a final model (LGBM) with the filtered features on the original, non-stationary data"];
    %%     N --> O["Calculate Permutation Feature Importance (PFI) on a hold-out validation set"];
    %%     O --> P["Keep only features with positive PFI scores (i.e., features that are genuinely predictive)"];
    %% end

    L --> M["End: Final, Robust, and Validated Feature List"];

    %% Styles
    style A fill:#e0f7fa,stroke:#0277bd,stroke-width:2px,color:#000
    style B,C,D fill:#b2ebf2,stroke:#0097a7,stroke-width:2px
    
    style E,F,G,H fill:#e1bee7,stroke:#6a1b9a,stroke-width:2px

    style I,J,K,L fill:#ffe082,stroke:#ff6f00,stroke-width:2px

    %% style M,N,O,P fill:#d7ccc8,stroke:#3e2723,stroke-width:2px
    
    style M fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px,font-weight:bold
```

## File: insights.py
```python
def load_metrics_data()
⋮----
"""Load and combine metrics data from all horizon files."""
horizons = [7, 14, 30, 60, 90]
base_path = 'results/results_{}d/cv/cv_metrics_2.csv'
⋮----
all_metrics = []
⋮----
file_path = base_path.format(h)
⋮----
df = pd.read_csv(file_path)
⋮----
def plot_metrics_vs_horizon(combined_df)
⋮----
"""Create line plots for each metric vs. horizon."""
metrics_to_plot = ['mae', 'rmse', 'mase', 'da', 'theil_u', 'training_time']
output_dir = 'results/insights'
⋮----
plot_path = os.path.join(output_dir, f'{metric}_vs_horizon.png')
⋮----
def calculate_degradation_rates(combined_df)
⋮----
"""
    Calculate and rank models using regression slope to measure
    the rate of performance degradation across horizons.
    """
metrics_to_analyze = ['mase', 'da']
models = combined_df['model_name'].unique()
horizons = sorted(combined_df['horizon'].unique())
⋮----
analysis_results = []
⋮----
model_df = combined_df[combined_df['model_name'] == model].sort_values('horizon')
⋮----
result = {'model_name': model}
⋮----
y = model_df[metric].values
x = model_df['horizon'].values
⋮----
# Regression Slope: Measures the rate of degradation
slope = np.polyfit(x, y, 1)[0]
⋮----
result_df = pd.DataFrame(analysis_results)
⋮----
# Rank models based on the slope
# For MASE, a lower (less steep) slope is better
⋮----
# For DA, a slope closer to zero is better (more stable)
⋮----
# Create a final combined rank by averaging the two slope ranks
rank_cols = ['mase_slope_rank', 'da_slope_rank']
⋮----
result_df = result_df.set_index('model_name')
⋮----
output_path = os.path.join(output_dir, 'degradation_rate_analysis.csv')
⋮----
# print(f"\nSaved stability analysis to: {output_path}")
⋮----
def main()
⋮----
"""Main function to run the analysis."""
df = load_metrics_data()
⋮----
# plot_metrics_vs_horizon(df)
⋮----
# Of course. Based on the detailed metrics in the analysis report, here is a list of specific research questions you could explore, categorized by theme.
⋮----
### Theme 1: Performance Degradation and Model Robustness
⋮----
# * **Question 1:** To what degree does the forecast accuracy (MASE) of each model degrade as the forecast horizon extends from short-term (14 days) to mid-range (60 days), and what does this reveal about their relative robustness?
#     * [cite_start]*This can be answered by analyzing the "% Increase in MASE" presented in Table 2, which shows SARIMAX had the lowest increase (37.3%) while Theta had the highest (96.6%)[cite: 98, 99].*
⋮----
# * **Question 2:** Which model architecture (e.g., linear statistical, smoother, attention-based) is most susceptible to catastrophic failure at long horizons (90 days), and why do its core assumptions break down?
#     * [cite_start]*The report shows SARIMAX's MASE skyrockets to 7.27 at 90 days[cite: 111]. [cite_start]The analysis explains this is due to its rigid, linear structure being unable to adapt to new market regimes, a fundamental violation of its stationarity assumption[cite: 114, 115, 116].*
⋮----
# ### Theme 2: The Trade-Off Between Direction and Magnitude
⋮----
# * **Question 3:** Is there a quantifiable trade-off between a model's ability to predict directional accuracy (DA) and its accuracy in predicting magnitude (MASE) in short-term forecasting?
#     * *This can be answered by comparing model rankings. [cite_start]At the 7-day horizon, SARIMAX ranks #1 in Directional Accuracy (72.25%) but last (#7) in MASE, while ETS ranks #1 in MASE (1.721) but 4th in Directional Accuracy, demonstrating a clear paradox[cite: 67, 68].*
⋮----
# * **Question 4:** How does the reliability of the directional signal provided by the top-performing directional model (SARIMAX) evolve as the forecast horizon extends from 7 to 90 days?
#     * *The data shows a clear decay. [cite_start]SARIMAX's DA falls systematically from a high of 72.25% at 7 days to 64.20% at 14 days, 57.6% at 30 days, and finally to 51.17% at 90 days, which is statistically indistinguishable from a coin toss[cite: 43, 75, 112].*
⋮----
# ### Theme 3: Computational Cost vs. Performance Gain
⋮----
# * **Question 5:** What is the relationship between a model's computational cost (training time) and its forecast accuracy (MASE) across different horizons?
#     * *The report indicates a poor relationship. [cite_start]The most computationally expensive models, TFT and SARIMAX, do not provide the best magnitude forecasts and are described as inefficient[cite: 69, 171]. [cite_start]For example, at the 30-day horizon, TFT required over 13 hours (47,202 seconds) to train while delivering worse magnitude accuracy than ETS, which trained in under an hour[cite: 93, 95].*
⋮----
# * **Question 6:** Do architecturally complex deep learning models like the Temporal Fusion Transformer (TFT) provide a justifiable return on investment (accuracy gain per second of training time) over simpler statistical models like ETS?
#     * [cite_start]*The report concludes they do not[cite: 15]. [cite_start]The massive computational burden of TFT is not met with a "corresponding, decisive, or consistent improvement in forecasting accuracy" over simpler and faster alternatives like ETS, especially in a univariate context[cite: 15].*
⋮----
# ### Theme 4: Synthesizing Findings for Practical Application
⋮----
# * **Question 7:** Based on the observed "decoupling" of direction and magnitude signals, what is the empirical justification for proposing a hybrid modeling approach for future research?
#     * [cite_start]*The report explicitly recommends this[cite: 195]. [cite_start]The justification lies in combining the strengths of different models: using a model proven to be a specialist in short-term direction like SARIMAX for a Stage 1 directional signal, and then using that signal as an input into a separate Stage 2 model (like TCN or GARCH) to predict the magnitude, thereby addressing the distinct failure modes of each model[cite: 196, 197, 198].*
⋮----
# * **Question 8:** To what extent does the collective failure of all seven models (MASE > 1.0) across all horizons provide empirical evidence for the Efficient Market Hypothesis in the context of Bitcoin price prediction?
#     * [cite_start]*The report presents this as its "most consequential finding"[cite: 5]. [cite_start]The fact that no model—from the simplest to the most complex—could produce a forecast that was, on average, better than a naive random walk is described as a "stark affirmation of the Efficient Market Hypothesis"[cite: 6, 164, 166].*
⋮----
# My Recommendation for Your Bachelor's Thesis (The Hybrid Approach)
⋮----
# For a thesis, you have a golden opportunity to show you understand this nuance. I recommend you adopt a hybrid of both philosophies, which is what the source you found advocates for. This will make your work more robust and impressive.
⋮----
# Here is your new, enlightened workflow:
⋮----
# Run Your Cross-Validation (As Planned): Perform your expanding window CV. Analyze the results and declare a "CV Winner." This demonstrates methodological rigor.
⋮----
# In your thesis: "Based on the cross-validation results summarized in Table 1, Model A was selected as the most promising model due to its superior average performance across the five backtest windows."
⋮----
# Run Top Contenders on the Holdout Set (The Pragmatic Step): Take your CV winner (Model A), the best runner-up (Model B), and a Naive Baseline. Retrain all three on the full training set and generate forecasts for the holdout set. Create the exact table your source suggested.
⋮----
# Analyze the Holdout Results (This is where you shine):
⋮----
# Scenario 1: The CV Winner also wins on the Holdout. This is the perfect outcome.
⋮----
# In your thesis: "To confirm this finding, the top models were evaluated on a final holdout set. As shown in Table 2, Model A also outperformed all contenders on this unseen data, confirming its superior generalization capability. Its MAE of 250 on the holdout set provides a final, validated measure of its performance."
⋮----
# Scenario 2: The Runner-Up wins on the Holdout. This is a more interesting and sophisticated finding.
⋮----
# In your thesis: "An important finding emerged during the final validation stage. While Model A was the winner during cross-validation, Model B achieved a lower MAE on the holdout set (Table 2). This suggests that Model A may have been slightly overfit to the historical patterns in the training folds. While the CV process is crucial for robust selection, the holdout result indicates that for this specific future period, Model B was empirically the most accurate. Therefore, we identify Model B as the final recommended model, while acknowledging this informative flip in the rankings."
⋮----
# Conclusion: The source you found gives excellent, practical advice. Running your top models on the holdout set is not "wrong." It provides richer context and a final competitive benchmark. Your goal is to find the best model, and by having two scores (CV and Holdout), you can have a much more intelligent discussion about what "best" truly means: is it the most robust on average (CV winner) or the best on the single most recent test (Holdout winner)?
⋮----
# As anticipated, the performance on the final holdout set (MASE = 0.85) represents a slight degradation compared to the average performance observed during cross-validation (Average MASE = 0.78). This is an expected and well-documented phenomenon in time series forecasting for two primary reasons. Firstly, it reflects the 'winner's curse' inherent in any model selection process, where the holdout score provides a less biased estimate of performance than the score that led to the model's selection. Secondly, it highlights the non-stationary nature of financial data, where the holdout period inevitably contains market dynamics not fully represented in the historical training data. Therefore, the CV score should be interpreted as the basis for our model selection, while the holdout score serves as the most realistic estimate of future real-world performance.
```

## File: src/models/mlforecast/models.py
```python
def create_lgb_config(trial: optuna.Trial) -> dict
⋮----
"""Create LightGBM configuration using comprehensive hyperparameters."""
⋮----
def create_xgb_config(trial: optuna.Trial) -> dict
⋮----
"""Create XGBoost configuration using comprehensive hyperparameters."""
⋮----
def create_cat_config(trial: optuna.Trial) -> dict
⋮----
"""Create CatBoost configuration using comprehensive hyperparameters."""
⋮----
def get_ml_models() -> Dict[str, Any]
⋮----
"""Returns a dictionary of ML models for AutoMLForecast."""
⋮----
# 'AutoXGBoost': AutoModel(model=xgb.XGBRegressor(), config=create_xgb_config),
# 'AutoCatBoost': AutoModel(model=cat.CatBoostRegressor(), config=create_cat_config),
```

## File: src/pipelines/results_processing.py
```python
"""
Results processing module for the Bitcoin forecasting pipeline.
"""
⋮----
def display_top_models(metrics_df: pd.DataFrame, title: str) -> None
⋮----
"""Display top performing models based on MAE."""
⋮----
# Remove any rows with NaN MAE values and sort by MAE (ascending - lower is better)
valid_results = metrics_df.dropna(subset=['mae']).copy()
⋮----
# Sort by MAE (lower is better) and display all models
ranked_models = valid_results.sort_values('mae')
⋮----
model_name = row['model_name']
mae = row['mae']
rmse = row['rmse']
mape = row['mape'] if pd.notna(row['mape']) else 'N/A'
training_time = row['training_time'] if pd.notna(row['training_time']) else 'N/A'
⋮----
def _get_top_n_by_metric_lines(df: pd.DataFrame, metric: str, n: int = 3) -> List[str]
⋮----
"""Generates formatted lines for top N models by a given metric."""
lines = []
⋮----
metric_df = df.dropna(subset=[metric])
⋮----
top_n_df = metric_df.nsmallest(n, metric)
⋮----
mape_str = f"{row['mape']:.4f}%" if pd.notna(row['mape']) else 'N/A'
time_str = f"{row['training_time']:.2f}s" if pd.notna(row['training_time']) else 'N/A'
⋮----
"""Generate a comprehensive summary report."""
⋮----
# Get dynamic directories for saving the report
⋮----
# Filter valid results
⋮----
# Report content
report_lines = [
⋮----
table_df = valid_results[['model_name', 'mae', 'rmse', 'mape', 'training_time']].copy()
⋮----
# Overall statistics
successful_models = len(valid_results)
best_mae = valid_results['mae'].min()
worst_mae = valid_results['mae'].max()
avg_mae = valid_results['mae'].mean()
⋮----
# Join all lines
report_text = "\n".join(report_lines)
⋮----
# Save report
report_path = final_dir / 'report.txt'
```

## File: results/results_14d/cv/best_configurations_comparison_nf.yaml
```yaml
# AutoTSMixerx:
#   input_size: 42
#   learning_rate: 0.002
#   scaler_type: robust
#   max_steps: 100
#   batch_size: 16
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 11
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_14_50_ratio
#   - btc_atr_14
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - hash_rate_blockchain
#   - MVRV_cbbi
#   - problem_malicious_sentiment
#   - volume_sentiment
#   - Gold_Share
#   - Gold_Volatility
#   - Oil_Brent_Price
#   - DJI
#   - NYFANG
#   - CBOE_Volatility
#   - DXY
#   - EURUSD
#   - RHODL_cbbi
#   - EM_ETF
#   - btc_volatility_index
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_width
#   - estimated_transaction_volume_usd_blockchain
#   - Puell_cbbi
#   - ReserveRisk_cbbi
#   - market_narrative_sentiment
#   - marketcap_sentiment
#   - Gold_Price
#   - btc_macd_diff
#   - btc_bb_high
#   n_block: 1
#   ff_dim: 128
#   dropout: 0.12086990176519524
#   n_series: 1
# AutoiTransformer:
#   input_size: 56
#   learning_rate: 0.0005
#   scaler_type: robust
#   max_steps: 200
#   batch_size: 16
#   windows_batch_size: 256
#   val_check_steps: 50
#   random_seed: 16
#   accelerator: gpu
#   logger: false
#   hidden_size: 128
#   n_heads: 8
#   n_series: 1
AutoBiTCN:
  input_size: 42
  learning_rate: 0.002
  scaler_type: robust
  max_steps: 200
  batch_size: 16
  windows_batch_size: 256
  val_check_steps: 50
  random_seed: 16
  accelerator: gpu
  logger: false
  hist_exog_list:
  - btc_sma_14_50_ratio
  - btc_atr_14
  - btc_trading_volume
  - active_addresses_blockchain
  - hash_rate_blockchain
  - MVRV_cbbi
  - problem_malicious_sentiment
  - volume_sentiment
  - Gold_Share
  - Gold_Volatility
  - Oil_Brent_Price
  - DJI
  - NYFANG
  - CBOE_Volatility
  - DXY
  - EURUSD
  - RHODL_cbbi
  - EM_ETF
  - btc_volatility_index
  - btc_sma_5
  - btc_ema_5
  - btc_sma_50
  - btc_bb_width
  - estimated_transaction_volume_usd_blockchain
  - Puell_cbbi
  - ReserveRisk_cbbi
  - market_narrative_sentiment
  - marketcap_sentiment
  - Gold_Price
  - btc_macd_diff
  - btc_bb_high
  hidden_size: 32
  dropout: 0.4175532212896679
# AutoTFT:
#   input_size: 42
#   learning_rate: 0.00136826103896924
#   scaler_type: robust
#   max_steps: 100
#   batch_size: 64
#   windows_batch_size: 1024
#   val_check_steps: 50
#   random_seed: 16
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_14_50_ratio
#   - btc_atr_14
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - hash_rate_blockchain
#   - MVRV_cbbi
#   - problem_malicious_sentiment
#   - volume_sentiment
#   - Gold_Share
#   - Gold_Volatility
#   - Oil_Brent_Price
#   - DJI
#   - NYFANG
#   - CBOE_Volatility
#   - DXY
#   - EURUSD
#   - RHODL_cbbi
#   - EM_ETF
#   - btc_volatility_index
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_width
#   - estimated_transaction_volume_usd_blockchain
#   - Puell_cbbi
#   - ReserveRisk_cbbi
#   - market_narrative_sentiment
#   - marketcap_sentiment
#   - Gold_Price
#   - btc_macd_diff
#   - btc_bb_high
#   hidden_size: 64
#   n_head: 8
```

## File: results/results_30d/cv/best_configurations_comparison_nf.yaml
```yaml
# AutoTSMixerx:
#   input_size: 60
#   learning_rate: 0.0001
#   scaler_type: robust
#   max_steps: 200
#   batch_size: 64
#   windows_batch_size: 256
#   val_check_steps: 50
#   random_seed: 18
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_14_50_ratio
#   - btc_sma_21_slope
#   - btc_rsi_14
#   - btc_atr_14
#   - miner_revenue_blockchain
#   - difficulty_blockchain
#   - PiCycle_cbbi
#   - RHODL_cbbi
#   - ReserveRisk_cbbi
#   - Fear Greed
#   - volume_sentiment
#   - Gold_Volatility
#   - NYFANG
#   - EM_ETF
#   - DXY
#   - EURUSD
#   - btc_trading_volume
#   - DJI
#   - CBOE_Volatility
#   - RUPL_cbbi
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_low
#   - btc_bb_width
#   - btc_volatility_index
#   - active_addresses_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - MVRV_cbbi
#   - market_narrative_sentiment
#   n_block: 2
#   ff_dim: 64
#   dropout: 0.2481952550559482
#   n_series: 1
# AutoiTransformer:
#   input_size: 120
#   learning_rate: 0.001
#   scaler_type: robust
#   max_steps: 50
#   batch_size: 64
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 18
#   accelerator: gpu
#   logger: false
#   hidden_size: 256
#   n_heads: 8
#   n_series: 1
AutoBiTCN:
  input_size: 60
  learning_rate: 0.00018104910021324544
  scaler_type: robust
  max_steps: 150
  batch_size: 16
  windows_batch_size: 256
  val_check_steps: 50
  random_seed: 6
  accelerator: gpu
  logger: false
  hist_exog_list:
  - btc_sma_14_50_ratio
  - btc_sma_21_slope
  - btc_rsi_14
  - btc_atr_14
  - miner_revenue_blockchain
  - difficulty_blockchain
  - PiCycle_cbbi
  - RHODL_cbbi
  - ReserveRisk_cbbi
  - Fear Greed
  - volume_sentiment
  - Gold_Volatility
  - NYFANG
  - EM_ETF
  - DXY
  - EURUSD
  - btc_trading_volume
  - DJI
  - CBOE_Volatility
  - RUPL_cbbi
  - btc_sma_5
  - btc_ema_5
  - btc_sma_50
  - btc_bb_low
  - btc_bb_width
  - btc_volatility_index
  - active_addresses_blockchain
  - estimated_transaction_volume_usd_blockchain
  - MVRV_cbbi
  - market_narrative_sentiment
  hidden_size: 16
  dropout: 0.2125950791494071
# AutoTFT:
#   input_size: 90
#   learning_rate: 0.00011098015738208735
#   scaler_type: robust
#   max_steps: 100
#   batch_size: 16
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 14
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - btc_sma_14_50_ratio
#   - btc_sma_21_slope
#   - btc_rsi_14
#   - btc_atr_14
#   - miner_revenue_blockchain
#   - difficulty_blockchain
#   - PiCycle_cbbi
#   - RHODL_cbbi
#   - ReserveRisk_cbbi
#   - Fear Greed
#   - volume_sentiment
#   - Gold_Volatility
#   - NYFANG
#   - EM_ETF
#   - DXY
#   - EURUSD
#   - btc_trading_volume
#   - DJI
#   - CBOE_Volatility
#   - RUPL_cbbi
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_low
#   - btc_bb_width
#   - btc_volatility_index
#   - active_addresses_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - MVRV_cbbi
#   - market_narrative_sentiment
#   hidden_size: 256
#   n_head: 8
```

## File: results/results_7d/cv/best_configurations_comparison_nf.yaml
```yaml
# AutoTSMixerx:
#   input_size: 42
#   learning_rate: 0.001
#   scaler_type: robust
#   max_steps: 50
#   batch_size: 64
#   windows_batch_size: 256
#   val_check_steps: 50
#   random_seed: 17
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - miner_revenue_blockchain
#   - RHODL_cbbi
#   - MVRV_cbbi
#   - Fear Greed
#   - bearish_sentiment
#   - core_technology_sentiment
#   - regulations_sentiment
#   - Gold_Share
#   - Oil_Brent_Price
#   - Oil_Volatility
#   - DJI
#   - CBOE_Volatility
#   - DXY
#   - EURUSD
#   - volume_sentiment
#   - IXIC
#   - EM_ETF
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_low
#   - btc_atr_14
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - ReserveRisk_cbbi
#   - marketcap_sentiment
#   - Gold_Price
#   - btc_bb_width
#   - Oil_Crude_Price
#   n_block: 2
#   ff_dim: 32
#   dropout: 0.7501377996375846
#   n_series: 1
# AutoiTransformer:
#   input_size: 42
#   learning_rate: 0.001
#   scaler_type: robust
#   max_steps: 50
#   batch_size: 64
#   windows_batch_size: 512
#   val_check_steps: 50
#   random_seed: 7
#   accelerator: gpu
#   logger: false
#   hidden_size: 512
#   n_heads: 4
#   n_series: 1
AutoBiTCN:
  input_size: 28
  learning_rate: 0.0005
  scaler_type: robust
  max_steps: 150
  batch_size: 32
  windows_batch_size: 256
  val_check_steps: 50
  random_seed: 6
  accelerator: gpu
  logger: false
  hist_exog_list:
  - miner_revenue_blockchain
  - RHODL_cbbi
  - MVRV_cbbi
  - Fear Greed
  - bearish_sentiment
  - core_technology_sentiment
  - regulations_sentiment
  - Gold_Share
  - Oil_Brent_Price
  - Oil_Volatility
  - DJI
  - CBOE_Volatility
  - DXY
  - EURUSD
  - volume_sentiment
  - IXIC
  - EM_ETF
  - btc_sma_5
  - btc_ema_5
  - btc_sma_50
  - btc_bb_low
  - btc_atr_14
  - btc_trading_volume
  - active_addresses_blockchain
  - estimated_transaction_volume_usd_blockchain
  - ReserveRisk_cbbi
  - marketcap_sentiment
  - Gold_Price
  - btc_bb_width
  - Oil_Crude_Price
  hidden_size: 32
  dropout: 0.08436123580637041
# AutoTFT:
#   input_size: 14
#   learning_rate: 0.002247964095706909
#   scaler_type: robust
#   max_steps: 100
#   batch_size: 64
#   windows_batch_size: 256
#   val_check_steps: 50
#   random_seed: 5
#   accelerator: gpu
#   logger: false
#   hist_exog_list:
#   - miner_revenue_blockchain
#   - RHODL_cbbi
#   - MVRV_cbbi
#   - Fear Greed
#   - bearish_sentiment
#   - core_technology_sentiment
#   - regulations_sentiment
#   - Gold_Share
#   - Oil_Brent_Price
#   - Oil_Volatility
#   - DJI
#   - CBOE_Volatility
#   - DXY
#   - EURUSD
#   - volume_sentiment
#   - IXIC
#   - EM_ETF
#   - btc_sma_5
#   - btc_ema_5
#   - btc_sma_50
#   - btc_bb_low
#   - btc_atr_14
#   - btc_trading_volume
#   - active_addresses_blockchain
#   - estimated_transaction_volume_usd_blockchain
#   - ReserveRisk_cbbi
#   - marketcap_sentiment
#   - Gold_Price
#   - btc_bb_width
#   - Oil_Crude_Price
#   hidden_size: 256
#   n_head: 4
```

## File: src/pipelines/feature_selection_pipeline/robust_selection.py
```python
class RobustSelectionMixin
⋮----
"""Mixin for robust feature selection methods."""
⋮----
def _save_feature_list(self, features: List[str], filename: str, results_dir: Optional[str] = None)
⋮----
"""Helper to save a list of features to a file."""
⋮----
output_path = Path(results_dir) / filename
⋮----
"""
        Unified data preparation function with perfect alignment of features and target.
        This version uses the raw 'y' column as the target.
        """
# Sort by date to ensure proper order
df_sorted = df.sort_values('ds').copy()
⋮----
target_col = 'y'
⋮----
# Get feature columns
⋮----
# All columns except metadata and the target are potential features
feature_cols = [col for col in df_sorted.columns if col not in ['unique_id', 'ds', 'y']]
⋮----
feature_cols = features
⋮----
# Create working dataframe with only needed columns
work_df = df_sorted[[target_col] + feature_cols].copy()
⋮----
# Create shifted target
⋮----
# Drop rows with any NaN values for perfect alignment
work_df = work_df.dropna()
⋮----
# Extract aligned data
X = work_df[feature_cols].values
y = work_df['y_shifted'].values
⋮----
"""
        Stability selection using multiple bootstrap samples.
        
        Args:
            train_df: Training dataframe
            method: Method to use for selection
            n_bootstrap: Number of bootstrap iterations
            selection_threshold: Minimum frequency for feature selection
            sample_fraction: Fraction of data to sample in each bootstrap
            
        Returns:
            Dictionary with stable features and selection frequencies
        """
⋮----
gpu_available = use_gpu and torch.cuda.is_available()
⋮----
# Track feature selection frequency
feature_selections = defaultdict(int)
⋮----
# Bootstrap sampling (maintaining temporal order)
n_samples = int(len(X) * sample_fraction)
# For time series, we take a contiguous chunk rather than random sampling
⋮----
start_idx = np.random.randint(0, len(X) - n_samples + 1)
end_idx = start_idx + n_samples
⋮----
# Direct model training on bootstrap sample using aligned data preparation
bootstrap_df = train_df.iloc[start_idx:end_idx].copy()
⋮----
# Extract base method name (remove suffixes like '_shap', '_stability')
base_method = method.split('_')[0]  # 'xgboost_shap' -> 'xgboost'
⋮----
params = model_params.copy()
⋮----
base_method = 'random_forest'
⋮----
model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1, **params)
⋮----
model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbosity=-1, **params)
⋮----
model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **params)
⋮----
# Get feature importances directly
importances = model.feature_importances_
⋮----
# Select top features (e.g., top 50% by importance)
importance_threshold = np.percentile(importances, 50)
selected_indices = np.where(importances >= importance_threshold)[0]
⋮----
# Count selections and build selected features list
selected_features_this_iteration = []
⋮----
feature_name = feature_cols_boot[idx]
⋮----
# Calculate selection frequencies
⋮----
feature_names = [col for col in train_df.columns if col not in ['unique_id', 'ds', 'y']]
selection_frequencies = {feature: 0.0 for feature in feature_names}
stable_features = feature_names[:min(10, len(feature_names))]  # Return top 10 features as fallback
⋮----
selection_frequencies = {
⋮----
# Select stable features
stable_features = [
⋮----
frequency_df = pd.DataFrame(
⋮----
result = {
⋮----
"""
        Handles multicollinearity using a two-stage approach:
        1. High Correlation Clustering: Removes features from highly correlated pairs.
        2. VIF Check: Iteratively removes features with high VIF.
        
        The decision of which feature to remove is based on stability scores.
        """
⋮----
# Ensure stability_scores is indexed by feature for easy lookup
⋮----
stability_scores = stability_scores.set_index('feature')
⋮----
# --- Stage 1: Pairwise Correlation ---
⋮----
corr_matrix = data_for_vif[features_to_check].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
⋮----
to_drop = set()
⋮----
# Find correlated features
correlated_features = upper.index[upper[column] > corr_threshold].tolist()
⋮----
# If one of the pair is already marked for dropping, skip
⋮----
# Compare stability scores to decide which to keep
score1 = stability_scores.loc[column, 'frequency_stability_avg']
score2 = stability_scores.loc[feature2, 'frequency_stability_avg']
⋮----
features_after_corr = [f for f in features_to_check if f not in to_drop]
⋮----
# --- Stage 2: VIF Calculation ---
⋮----
features_for_vif = features_after_corr.copy()
⋮----
X_vif = add_constant(data_for_vif[features_for_vif].dropna())
vif_data = pd.DataFrame()
⋮----
# Exclude 'const' from VIF check
vif_data = vif_data[vif_data['feature'] != 'const']
⋮----
max_vif = vif_data['VIF'].max()
⋮----
feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
⋮----
"""
        Orchestrates a simplified feature selection pipeline.
        
        Workflow:
        1.  Run Stability Selection with multiple tree-based models on the raw 'y' target.
        2.  Create a consensus list of features based on how many models selected them.
        3.  Optionally, perform multicollinearity reduction.
        """
⋮----
# --- Step 1: Stability Selection ---
stability_results = {}
⋮----
params = kwargs.copy()
# The 'stability_selection' method handles its own kwargs
result = self.stability_selection(train_df, method=method, use_gpu=use_gpu, **params)
⋮----
# --- Step 2: Create Consensus ---
⋮----
# Get the list of stable features from each model
stable_features_per_model = [res['selected_features'] for res in stability_results.values()]
⋮----
# Count how many models selected each feature as stable
feature_counts = Counter()
⋮----
# Select features that meet the consensus level
initial_consensus_features = [
⋮----
final_features = initial_consensus_features
⋮----
# --- Step 3: Multicollinearity Reduction (Optional) ---
⋮----
# Combine all frequency dataframes for tie-breaking
all_freq_dfs = [res['selection_frequency'].rename(columns={'frequency': f'frequency_{method}'})
merged_freq_df = all_freq_dfs[0]
⋮----
merged_freq_df = pd.merge(merged_freq_df, df, on='feature', how='outer')
merged_freq_df = merged_freq_df.fillna(0)
freq_cols = [f'frequency_{method}' for method in tree_methods]
⋮----
# Prepare data for VIF (use the training set part)
⋮----
data_for_vif = train_df[feature_names].copy()
⋮----
features_after_multicollinearity = self.handle_multicollinearity(
⋮----
final_features = features_after_multicollinearity
⋮----
# --- Step 4: Final Recommendations and Reporting ---
⋮----
# Create a final recommendation DataFrame
final_recommendation_df = pd.DataFrame({
⋮----
# Build final results package
results_package = {
⋮----
def run_robust_auto_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]
⋮----
"""
        Fully automated robust selection with sensible defaults.
        """
```

## File: src/models/neuralforecast/auto_cfg.py
```python
def neural_auto_cfg(h: int) -> Dict[str, Dict]
⋮----
"""
    Get model configurations optimized for Bitcoin forecasting.

    Args:
        h: Forecast horizon (number of time steps to forecast)

    Returns:
        Dictionary of model configurations with Ray Tune choices for model-specific parameters only.
        Common parameters are handled in base_auto_config.
    """
⋮----
# NHITS config
nhits_config = {
⋮----
),  # MaxPool's Kernelsize
⋮----
),  # Interpolation expressivity ratios
⋮----
# NBEATS config
nbeats_config = {}
⋮----
# LSTM config
lstm_config = {
⋮----
# TFT config
tft_config = {
⋮----
# Transformer config
transformer_config = {
⋮----
# TSMixer config
tsmixer_config = {
⋮----
# BiTCN config
bitcn_config = {
⋮----
# KAN config
kan_config = {
```

## File: src/models/statsforecast/models.py
```python
"""
Bitcoin-optimized statistical models for forecasting.
"""
⋮----
def get_statistical_models(season_length: int = 7) -> List[Any]
⋮----
"""
    Get optimized statistical models for Bitcoin forecasting.
    
    Args:
        season_length: Seasonal period. This is the length of a full seasonal
            cycle in the data (e.g., 7 for daily data with weekly patterns).
            It is a property of the data, not the forecast horizon.
        
    Returns:
        List of statistical model instances
    """
# Full model set optimized for Bitcoin characteristics
all_models = [
⋮----
# PRIMARY: Best for Bitcoin's non-stationary, trending, volatile nature
# AutoARIMA(season_length=season_length),
AutoETS(season_length=season_length, model='ZZZ'),  # Auto-select
# AutoETS(season_length=season_length, model='MMM'),  # Multiplicative
# AutoETS(season_length=season_length, model='MAM'),  # Mixed
⋮----
# # THETA METHODS: Excellent for trending financial data
# AutoTheta(season_length=season_length),
⋮----
# # COMPLEX SMOOTHING: Handles complex patterns
# AutoCES(season_length=season_length),
⋮----
# # BASELINE MODELS: Simple but effective
# RandomWalkWithDrift(),
# WindowAverage(window_size=3),   # Very responsive
# WindowAverage(window_size=7),   # Weekly patterns
⋮----
# # SEASONAL PATTERNS
# SeasonalNaive(season_length=season_length),
# SeasonalWindowAverage(season_length=season_length, window_size=3),
```

## File: src/pipelines/feature_selection_pipeline/base.py
```python
"""
Base module for the Feature Selection Pipeline.
"""
⋮----
# Local imports from the new package structure
⋮----
# Local imports from the project
⋮----
class FeatureSelector(
⋮----
"""
    Advanced Feature Selection Pipeline for Time Series Forecasting.
    
    This class integrates a comprehensive suite of feature selection techniques
    through a modular, mixin-based architecture.
    """
⋮----
def __init__(self, random_state: int = SEED, verbose: bool = True, use_gpu: bool = False)
⋮----
"""
        Initialize the FeatureSelector.
        
        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            use_gpu: Whether to attempt to use GPU for acceleration
        """
⋮----
def print_info(self, message: str)
⋮----
"""Print information if verbose is enabled."""
⋮----
"""
        Orchestrates the robust feature selection strategy (formerly Step 3).
        This is the main entry point for the feature selection process.
        """
⋮----
# Data splitting for robust evaluation
# Ensure 'ds' is sorted if not already
df_sorted = df.sort_values('ds').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.9)
train_df = df_sorted.iloc[:split_idx]
val_df = df_sorted.iloc[split_idx:]
⋮----
# The robust_comprehensive_selection function now handles all steps and
# returns the final results in the correct format.
selection_results = self.robust_comprehensive_selection(
⋮----
def get_feature_summary(self) -> pd.DataFrame
⋮----
"""
        Returns a summary of feature selection results.
        """
summary_data = []
⋮----
def save_models(self, save_dir: str)
⋮----
"""
        Saves fitted selectors and autoencoder models.
        """
save_path = Path(save_dir)
⋮----
def get_optimal_features(self, method: str = 'auto') -> List[str]
⋮----
"""
        Retrieves the list of selected features for a given method.
        """
⋮----
# Use a consensus or default method if available
method = 'robust_comprehensive'
⋮----
def apply_optimal_selection(self, df: pd.DataFrame, method: str = 'auto') -> pd.DataFrame
⋮----
"""
        Applies the optimal feature set to a dataframe.
        """
features_to_keep = self.get_optimal_features(method)
⋮----
meta_cols = [col for col in ['unique_id', 'ds', 'y'] if col in df.columns]
⋮----
final_cols = meta_cols + [f for f in features_to_keep if f in df.columns]
```

## File: src/pipelines/feature_selection.py
```python
def parse_args()
⋮----
parser = argparse.ArgumentParser(description='Feature Selection Pipeline')
⋮----
def main()
⋮----
args = parse_args()
⋮----
# 1. Load data
⋮----
# Data cleaning
⋮----
cols_before = train_df.columns.tolist()
non_constant_mask = train_df.nunique() > 1
train_df = train_df.loc[:, non_constant_mask]
cols_after = train_df.columns.tolist()
removed_cols = sorted(list(set(cols_before) - set(cols_after)))
⋮----
test_df = test_df[cols_after]
hist_exog_list = [col for col in hist_exog_list if col in cols_after]
⋮----
# 2. Initialize selector
selector = FeatureSelector(random_state=42, verbose=True, use_gpu=True)
results_dir = "src/pipelines/feature_results"
⋮----
# 3. Set selection params from args
selection_params = {
⋮----
# 4. Run feature selection
results = selector.run_complete_feature_selection_strategy(
⋮----
# 5. Print results
⋮----
consensus_features = results.get('final_recommendations', {}).get('consensus_features', [])
recommendation_df = results.get('final_recommendations', {}).get('feature_counts')
⋮----
# 6. Export results
⋮----
full_original_df = load_and_prepare_data(data_path=RAW_DATA_PATH)
final_df = full_original_df[['unique_id', 'ds', 'y'] + consensus_features]
⋮----
multicoll_postfix = "_mc" if selection_params['handle_multicollinearity_flag'] else ""
output_path = f"data/processed/feature_selection_{HORIZON}{multicoll_postfix}.parquet"
```

## File: src/utils/utils.py
```python
"""
Utility functions for neural forecasting pipeline.
"""
⋮----
def get_horizon_directories()
⋮----
"""Get the appropriate directories based on the HORIZON parameter."""
⋮----
def seed_everything(seed=42)
⋮----
"""Set seeds for reproducibility across all random number generators."""
⋮----
def setup_environment(seed=42, ray_config=None)
⋮----
"""Setup the environment for neural forecasting."""
# Set seed for reproducibility
⋮----
# Setup logging
⋮----
# Initialize Ray
⋮----
ray_config = {
⋮----
def print_data_info(df, train_df, test_df)
⋮----
"""Print information about data splits."""
⋮----
def get_historical_exogenous_features(df, exclude_cols=None)
⋮----
"""Get list of historical exogenous features from dataframe."""
⋮----
exclude_cols = ['ds', 'unique_id', 'y']
⋮----
all_cols = df.columns.tolist()
hist_exog_list = [col for col in all_cols if col not in exclude_cols]
⋮----
def calculate_metrics_1(df: pd.DataFrame) -> Dict
⋮----
"""Calculate metrics from cross-validation results."""
⋮----
mask = ~(np.isnan(y_true) | np.isnan(y_pred))
⋮----
# Hit Ratio (Directional Accuracy)
# Compares the direction (up/down) of the point forecast to the actual movement. Only the forecasted and actual values are needed to determine if the direction was predicted correctly.
# Profit/Loss (PnL)
# Simulates trading outcomes using the point forecast as the trading signal. The actual and predicted prices are used to compute hypothetical returns.
# Sharpe Ratio
# Uses the series of returns generated from point forecasts (as trading signals or portfolio weights) to calculate risk-adjusted return. Only point forecasts and actual prices are needed.
# Maximum Drawdown (MDD)
# Evaluates the largest peak-to-trough loss in a simulated portfolio based on point forecast-driven trades or allocations.
# Heteroskedasticity-Adjusted MSE (HMSE)
# This is a variant of MSE that weights errors by realized volatility, but still requires only the point forecast and actual value for each period, plus a volatility estimate.
⋮----
"""
    Calculates financial-specific time series metrics.

    Args:
        y_true (np.ndarray): Ground truth (correct) target values.
        y_pred (np.ndarray): Estimated target values.
        y_lag1 (np.ndarray): Lagged (t-1) ground truth values for calculating changes.

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'da': Directional Accuracy in percentage.
    """
# Create a mask to handle any potential NaNs in the inputs
mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_lag1))
⋮----
# Directional Accuracy (DA)
true_direction = np.sign(y_true - y_lag1)
pred_direction = np.sign(y_pred - y_lag1)
da = np.mean(true_direction == pred_direction) * 100
⋮----
# Theil's U statistic (U1)
# Ensures a fair comparison by calculating model and naive RMSE on the same data.
# rmse_model = np.sqrt(np.mean((y_true - y_pred) ** 2))
# rmse_naive = np.sqrt(np.mean((y_true - y_lag1) ** 2))
# theil_u = rmse_model / rmse_naive if rmse_naive > 0 else np.nan
⋮----
# return {'da': da, 'theil_u': theil_u}
⋮----
def _get_model_names_from_df(df: pd.DataFrame) -> List[str]
⋮----
"""Extracts model names from a forecast DataFrame, ignoring PI columns."""
standard_cols = {'unique_id', 'ds', 'cutoff', 'y', 'y_lag1'}
# Assumes PI columns are formatted like 'Model-lo-95' or 'Model-hi-95'
⋮----
"""
    Calculate evaluation metrics from cross-validation results.

    This function uses `utilsforecast.evaluate` for standard metrics (MAE, RMSE, MASE)
    and includes custom calculations for financial-specific metrics like
    Directional Accuracy (DA) and Theil's U statistic.

    Args:
        cv_df (pd.DataFrame): DataFrame with cross-validation results.
                               Must contain 'unique_id', 'ds', 'y', and model forecast columns.
        model_names (List[str]): A list of model names corresponding to columns in `cv_df`.
        historical_df (pd.DataFrame): The training dataframe with original prices,
                                 Required for MASE calculation.

    Returns:
        Dict[str, Dict]: A dictionary where keys are model names and values are
                         dictionaries of their calculated metrics.
    """
⋮----
model_names = [model_names]
⋮----
cv_df_clean = cv_df.copy()
⋮----
eval_df = cv_df_clean.drop(columns=['cutoff'], errors='ignore')
⋮----
# Use utilsforecast for standard metrics
evaluation_df = evaluate(eval_df, metrics=[mae, rmse, partial(mase, seasonality=7)], train_df=historical_df)
aggregated_metrics_df = evaluation_df.groupby('metric').mean(
⋮----
# Prepare for financial metrics: calculate y_lag1 once
⋮----
all_models_financial_metrics_per_cutoff = {model_name: [] for model_name in model_names}
⋮----
# Group by unique_id and cutoff to process each fold/cutoff independently
⋮----
y_true_group = group_df['y'].values
y_lag1_group = group_df['y_lag1'].values
⋮----
# This should ideally not happen if model_names are correctly extracted, but good for robustness
⋮----
y_pred_group = group_df[model_name].values
financial_metrics_group = _calculate_financial_metrics(y_true_group, y_pred_group, y_lag1_group)
⋮----
results = {}
⋮----
# Get standard metrics from the aggregated results
metrics = aggregated_metrics_df.get(model_name, pd.Series(dtype=float)).to_dict()
⋮----
# Calculate RMSE/MAE ratio
⋮----
# Average the financial metrics across cutoffs
⋮----
# Convert list of dicts to dict of lists, then average
averaged_financial_metrics = {
⋮----
# If no financial metrics were calculated for a model, assign NaN or appropriate default
⋮----
error_msg = f'Error calculating metrics: {e}'
⋮----
"""
    Calculates metrics on a consolidated CV dataframe and returns a final results dataframe.

    Args:
        consolidated_cv_df (pd.DataFrame): The consolidated cross-validation results.
        historical_df (pd.DataFrame): The training dataframe, passed to `calculate_metrics`
                                 for MASE calculation.
    """
⋮----
# Get model names from columns (excluding standard and PI columns)
model_names = _get_model_names_from_df(consolidated_cv_df)
⋮----
metrics_results = calculate_metrics(consolidated_cv_df, model_names, historical_df)
⋮----
# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame.from_dict(metrics_results, orient='index')
results_df = results_df.reset_index().rename(columns={'index': 'model_name'})
⋮----
# Ensure a consistent column order for the final report
metric_cols = ['mae', 'rmse', 'rmse_mae_ratio', 'mase', 'da']
ordered_cols = [
⋮----
# Append any other columns (like 'error') that might exist
other_cols = [col for col in results_df.columns if col not in ordered_cols]
⋮----
def save_best_configurations(fitted_objects: Dict, save_dir: str = None) -> Dict[str, List]
⋮----
"""
    Extract and save best configurations from fitted auto models using built-in methods.
    Saves all configs to a single YAML file for easy comparison.
    
    Returns:
        Dictionary containing best configurations for each framework
    """
⋮----
save_dir = RESULTS_DIR / "auto_model_configs"
⋮----
# Ensure save directory exists
⋮----
all_best_configs = {}
yaml_configs = {}  # For YAML output structure
⋮----
def clean_config_value(value)
⋮----
"""Clean config values for YAML serialization."""
if hasattr(value, '__name__'):  # For loss functions etc.
⋮----
elif hasattr(value, 'item'):  # For numpy types
⋮----
# Extract neural model configs using built-in method
⋮----
nf_auto = fitted_objects['neural']
⋮----
neural_configs = []
⋮----
# Use built-in method to get best config
best_config = model.results.get_best_result().config
model_name = model.__class__.__name__
⋮----
# Clean config for YAML
clean_config = {}
⋮----
# Skip non-hyperparameter keys
⋮----
# Extract ML model configs using built-in method
⋮----
ml_auto = fitted_objects['ml']
⋮----
ml_configs = []
⋮----
# Use built-in method to get best config
best_config = results.best_trial.user_attrs['config']
⋮----
# Keep nested structure for YAML - don't flatten
⋮----
# Statistical models don't have hyperparameters to save
⋮----
# Save all configs to a single YAML file with timestamp
⋮----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
yaml_path = save_dir / f"best_configurations_comparison_{timestamp}.yaml"
⋮----
def extract_model_names_from_columns(columns: List[str]) -> List[str]
⋮----
"""
    Extract model names from dataframe columns, handling both Auto and normal models.
    Handles prediction interval suffixes like '-lo-90', '-hi-95', '-median', etc.
    
    Args:
        columns: List of column names from dataframe
        
    Returns:
        List of unique model names (Auto or normal)
    """
model_names = set()
ignore_cols = {'unique_id', 'ds', 'y', 'cutoff'}
⋮----
# Extract base model name by removing suffixes
⋮----
base_name = col.split('-lo-')[0]
⋮----
base_name = col.split('-hi-')[0]
⋮----
base_name = col.split('-median')[0]
⋮----
base_name = col
⋮----
class NumpyEncoder(json.JSONEncoder)
⋮----
""" Custom encoder for numpy data types """
def default(self, obj)
⋮----
def save_dict_to_json(data_dict: Dict, file_path: Path)
⋮----
"""
    Saves a dictionary to a JSON file, with special handling for numpy types.

    Args:
        data_dict (Dict): The dictionary to save.
        file_path (Path): The path to the output JSON file.
    """
⋮----
def load_json_to_dict(file_path: Path) -> Dict
⋮----
"""
    Loads a JSON file into a dictionary.

    Args:
        file_path (Path): The path to the input JSON file.

    Returns:
        Dict: The loaded dictionary.
    """
⋮----
def load_yaml_to_dict(file_path: Path) -> Dict[str, Any]
⋮----
"""
    Loads a YAML file into a dictionary.

    Args:
        file_path (Path): The path to the input YAML file.

    Returns:
        Dict: The loaded dictionary.
    """
⋮----
refit_frequency: int = 1  # Refit every N windows (1 = every window, 3 = every 3 windows)
⋮----
"""
    More efficient rolling forecast with configurable refit frequency.
    
    Args:
        refit_frequency: How often to refit models (1 = every window, 2 = every 2 windows, etc.)
                        Set to 0 to disable refitting (fastest but less adaptive)
    """
⋮----
all_forecasts_list = []
current_train_df = train_df.copy()
⋮----
# Prepare column names for future exogenous features
exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
futr_cols = ['unique_id', 'ds'] + exog_cols
⋮----
test_duration_steps = len(test_df)
⋮----
n_windows = (test_duration_steps + horizon_length - 1) // horizon_length
⋮----
start_idx = window_idx * horizon_length
end_idx = min(start_idx + horizon_length, test_duration_steps)
current_test_window_actuals_df = test_df.iloc[start_idx:end_idx].copy()
⋮----
# Prepare futr_df for prediction
futr_df_for_predict = current_test_window_actuals_df[futr_cols].copy()
⋮----
futr_df_for_predict = current_test_window_actuals_df[['unique_id', 'ds']].copy()
⋮----
window_forecast_df = nf_model.predict(futr_df=futr_df_for_predict, level=predict_level)
⋮----
# Update training data with ACTUAL VALUES (not predictions) and conditionally refit
⋮----
# Always append actual values to training data
actuals_to_append = current_test_window_actuals_df
⋮----
current_train_df = pd.concat([current_train_df, actuals_to_append], ignore_index=True)
⋮----
# Conditional refit based on frequency
should_refit = (refit_frequency > 0 and (window_idx + 1) % refit_frequency == 0)
⋮----
fit_val_size = horizon_length if horizon_length > 0 and len(current_train_df) > horizon_length * 2 else None
⋮----
final_concatenated_forecasts_df = pd.concat(all_forecasts_list, ignore_index=True)
⋮----
def my_init_config(trial: optuna.Trial)
⋮----
lag_transforms = [
⋮----
RollingMean(window_size=7, min_samples=1),  # 7 days instead of 24*7 hours
⋮----
lag_to_transform = trial.suggest_categorical('lag_to_transform', [1, 2])  # in days
⋮----
'lags': [i for i in range(1, 7)],  # 1 to 6 days
⋮----
def my_fit_config(trial: optuna.Trial)
⋮----
static_features = ['unique_id']
⋮----
static_features = None
⋮----
def custom_mae_loss(df: DataFrame, train_df: DataFrame) -> float
⋮----
"""
    Calculates Mean Absolute Error.
    'df' contains predictions in a column named 'model' and actuals in 'target_col'.
    'train_df' is the training data for the current window (can be ignored for simple MAE).
    """
actuals = df['y'].to_numpy()
predictions = df['model'].to_numpy()
```

## File: main.py
```python
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
⋮----
# Configuration and Model
⋮----
# Pipeline Step Modules
⋮----
class BitcoinAnalysisPipeline
⋮----
"""
    Generates reports and visualizations from existing model results.
    """
⋮----
def __init__(self)
⋮----
self.pipeline_results = {}  # Stores various results and info from pipeline steps
⋮----
def _load_results(self)
⋮----
"""Loads the latest holdout metrics and forecast results."""
⋮----
# Set directories based on horizon
⋮----
# Load holdout metrics from the final results directory
metrics_results_path = self.final_dir / "metrics_results.csv"
metrics_df = pd.read_csv(metrics_results_path)
⋮----
# Load forecast data for plotting
final_plot_results_path = self.final_dir / f"final_plot_results.json"
all_forecasts_data = load_json_to_dict(final_plot_results_path)
⋮----
def _get_workflow_info(self, hist_exog_list: List[str]) -> Dict
⋮----
"""Reconstructs the workflow info dictionary for the summary report."""
stat_models_instances = get_statistical_models(season_length=HORIZON)
ml_models_instances = get_ml_models()
neural_models_instances = get_neural_models(
⋮----
def run_analysis_pipeline(self) -> Dict
⋮----
"""Run the complete analysis and reporting pipeline."""
⋮----
start_time = time.time()
⋮----
# We still need the data for context and plotting
⋮----
# This info is needed for the report
⋮----
summary_report_str = ""
# summary_report_str = generate_summary_report(
#     metrics_df,
#     HORIZON,
#     self.pipeline_results["data_preparation_info"],
#     self.pipeline_results["auto_models_workflow_info"],
#     RESULTS_DIR,
# )
⋮----
execution_time = time.time() - start_time
⋮----
def main()
⋮----
pipeline = BitcoinAnalysisPipeline()
run_outcome = pipeline.run_analysis_pipeline()
```

## File: src/pipelines/model_forecasting.py
```python
"""
Module for final model fitting, prediction, and rolling forecasts in the Auto Models workflow.
"""
⋮----
# Configuration and local imports
⋮----
# Get dynamic directories based on HORIZON
⋮----
def generate_naive_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame
⋮----
"""
    Generates a price-based naive forecast for the test set.
    The forecast for day T is the actual price from day T-1.
    """
⋮----
# Create the forecast dataframe from the test set
naive_forecast_df = test_df[['unique_id', 'ds']].copy()
⋮----
# Get the last price from the training set as the first prediction
first_forecast_date = test_df['ds'].min()
last_train_date = first_forecast_date - pd.Timedelta(days=1)
last_train_price_row = original_df[original_df['ds'] == last_train_date]
⋮----
# Fallback to last known price in train_df if original_df is missing the date
last_train_price = train_df.loc[train_df['ds'].idxmax()]['y']
⋮----
last_train_price = last_train_price_row['y'].iloc[0]
⋮----
# The actual prices for the test period are in the original test_df 'y' column
# Shift these prices by one to create the naive forecast
# P_hat(t) = P_actual(t-1)
shifted_prices = test_df['y'].shift(1).values
# The first forecasted value is the last known price from the training period
⋮----
"""
    Perform final fit and predict on test data for all models.
    For ML models, this function also handles the CV and processing of CV results.
    """
final_results = []
test_length = len(test_df)
⋮----
ml_cv_results_df = pd.DataFrame()
⋮----
use_rolling = ENABLE_ROLLING_FORECAST and HORIZON < test_length
⋮----
# Initialize forecast dataframes and fitted objects for config saving
stat_forecasts_df = None
ml_forecasts_df = None
neural_forecasts_df = None
fitted_objects = {}
⋮----
# Convert 'unique_id' to category dtype if it is not already
⋮----
# --- Forecast Statistical Models ---
⋮----
sf = StatsForecast(models=stats_models, freq='D', verbose=True)
⋮----
# --- Forecast (fit + predict in one step) ---
forecast_start_time_stats = time.time()
h = len(test_df['ds'].unique()) if not test_df.empty else HORIZON
⋮----
# Prepare future exogenous variables (exclude 'y' column)
exog_cols_stats = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
X_df_stats = test_df[['unique_id', 'ds'] + exog_cols_stats].copy() if exog_cols_stats and not test_df.empty else None
⋮----
stat_forecasts_df = sf.forecast(
⋮----
# level=LEVELS,
# prediction_intervals=ConformalIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
⋮----
total_stats_forecast_time = time.time() - forecast_start_time_stats
⋮----
model_name = model_instance.__class__.__name__
⋮----
# --- Fit-Predict ML Models ---
⋮----
fit_start_time_ml = time.time()
ml_model_metadata = {}
⋮----
ml = AutoMLForecast(
⋮----
# fit_config=my_fit_config,
⋮----
# --- Fit models (with CV/hyperparameter tuning) ---
⋮----
# prediction_intervals=MLPredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL, h=HORIZON)
⋮----
total_ml_fit_time = time.time() - fit_start_time_ml
⋮----
# === Process In-sample CV predictions for ML models ===
⋮----
ml_cv_df = ml.forecast_fitted_values(# level=LEVELS
⋮----
auto_model_names = extract_model_names_from_columns(ml_cv_df.columns.tolist())
⋮----
time_per_model = total_ml_fit_time / len(auto_model_names)
⋮----
# Process the CV results for ML models
cv_metrics_df = process_cv_results(ml_cv_df, original_df)
⋮----
# Add training times to the results
⋮----
training_times_df = pd.DataFrame(
ml_cv_results_df = cv_metrics_df.merge(training_times_df, on='model_name', how='left')
⋮----
# --- Predict on Test Set---
predict_start_time_ml = time.time()
exog_cols_ml = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
X_df = test_df[['unique_id', 'ds'] + exog_cols_ml].copy() if exog_cols_ml and not test_df.empty else None
⋮----
ml_forecasts_df = pd.DataFrame()
ml_forecasts_df = ml.predict(
⋮----
# level=LEVELS
⋮----
total_ml_predict_time = time.time() - predict_start_time_ml
⋮----
# Store ML object for configuration saving
⋮----
fit_time = time.time() - fit_start_time_ml
⋮----
# Also create a failed entry in the results df
ml_cv_results_df = pd.DataFrame([{
⋮----
# --- Fit-Predict Auto Neural Models ---
⋮----
# === Phase 1: Collective Fit ===
⋮----
nf = NeuralForecast(models=neural_models, freq='D', local_scaler_type='robust')
⋮----
fit_start_time = time.time()
⋮----
# prediction_intervals=PredictionIntervals(n_windows=PI_N_WINDOWS_FOR_CONFORMAL)
⋮----
total_neural_fit_time = time.time() - fit_start_time
⋮----
# === Phase 2: Collective Predict (Direct Multi-Step Forecast) ===
⋮----
predict_start_time = time.time()
exog_cols = [col for col in test_df.columns if col not in ['unique_id', 'ds', 'y']]
futr_df = test_df[['unique_id', 'ds'] + exog_cols].copy() if not test_df.empty else None
⋮----
neural_forecasts_df = nf.predict(
⋮----
# level=LEVELS
⋮----
neural_forecasts_df = pd.DataFrame()
⋮----
total_neural_predict_time = time.time() - predict_start_time
⋮----
# Store neural object for configuration saving
⋮----
# Log a single error for the whole batch
⋮----
# === CONSOLIDATED PROCESSING: Merge all forecasts and process together ===
⋮----
# --- Generate Naive Forecast ---
naive_forecast_df = generate_naive_forecast(train_df, test_df, original_df)
⋮----
# Start with test data as base (preserve original test_df with actuals)
consolidated_forecasts_df = test_df[['unique_id', 'ds', 'y']].copy()
⋮----
# Merge forecasts from all frameworks
⋮----
consolidated_forecasts_df = consolidated_forecasts_df.merge(naive_forecast_df, how='left', on=['unique_id', 'ds'])
⋮----
consolidated_forecasts_df = consolidated_forecasts_df.merge(stat_forecasts_df, how='left', on=['unique_id', 'ds'])
⋮----
consolidated_forecasts_df = consolidated_forecasts_df.merge(ml_forecasts_df, how='left', on=['unique_id', 'ds'])
⋮----
consolidated_forecasts_df = consolidated_forecasts_df.merge(neural_forecasts_df, how='left', on=['unique_id', 'ds'])
⋮----
# --- Back-transformation to Price Scale ---
# Extract all model names before transformation
model_names_for_transform = extract_model_names_from_columns(consolidated_forecasts_df.columns.tolist())
⋮----
# The 'Naive' model is already in price scale, so we exclude it from transformation.
⋮----
# Perform the back-transformation from log-returns to prices using the correct function
consolidated_forecasts_df = back_transform_forecasts_from_log_returns(
⋮----
# Store consolidated forecasts for visualization
⋮----
# Process all models at once
evaluation_method = "direct_forecast"
⋮----
# Extract Auto model names directly from consolidated dataframe (all models have Auto prefix)
auto_model_names = extract_model_names_from_columns(consolidated_forecasts_df.columns.tolist())
⋮----
# Process each Auto model
⋮----
start_time = time.time()
⋮----
# Check if model prediction column exists
⋮----
# # Extract prediction intervals
# lo_preds = {}
# hi_preds = {}
# for level in LEVELS:
#     lo_col = f"{model_name}-lo-{level}"
#     hi_col = f"{model_name}-hi-{level}"
#     if lo_col in consolidated_forecasts_df.columns:
#         lo_preds[str(level)] = consolidated_forecasts_df[lo_col].values
#     if hi_col in consolidated_forecasts_df.columns:
#         hi_preds[str(level)] = consolidated_forecasts_df[hi_col].values
⋮----
# Store in final_plot_results_dict
⋮----
# 'lo': lo_preds,
# 'hi': hi_preds
⋮----
# Calculate metrics
metrics = calculate_metrics(consolidated_forecasts_df, [model_name], original_df)
# Extract metrics for this specific model
model_metrics = metrics.get(model_name, {})
processing_time = time.time() - start_time
⋮----
error_msg = str(e)
⋮----
# # Save best configurations to the specified directory if it exists
# if final_model_config_dir:
#     save_best_configurations(fitted_objects, final_model_config_dir)
⋮----
final_results_df = pd.DataFrame(final_results)
⋮----
successful_final = final_results_df[final_results_df['status'] == 'success']
⋮----
# Get data and models
⋮----
stat_models = get_statistical_models(season_length=7)
# ml_models = get_ml_models()
# neural_models = get_neural_models(horizon=HORIZON, num_samples=NUM_SAMPLES_PER_MODEL, hist_exog_list=hist_exog_list)
neural_models = get_normal_neural_models(horizon=HORIZON, config_path=CV_DIR / "best_configurations_comparison_nf.yaml", hist_exog_list=hist_exog_list)
⋮----
# print(test_df.head())
# Perform final fit and predict
⋮----
# Generate timestamp for filenames
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
⋮----
# Save final results to CSV with timestamp
final_results_path = FINAL_DIR / f"metrics_results_{timestamp}.csv"
⋮----
# Save final plot results to JSON with timestamp
plot_results_path = FINAL_DIR / f"final_plot_results_{timestamp}.json"
⋮----
# save ml_cv_results_df to csv
```

## File: src/pipelines/visualization.py
```python
"""
Module for creating visualizations in the Auto Models workflow.
"""
⋮----
def _prepare_plot_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple
⋮----
"""Helper to prepare common plotting data."""
train_tail = train_df.tail(40).copy()
⋮----
test_df = test_df.copy()
⋮----
def _save_plot(fig, plot_path: Path, dpi: int = 400)
⋮----
"""Helper to save plots with consistent settings."""
⋮----
"""Create unified forecast plot showing all models."""
⋮----
# Plot actual values
combined_actuals = pd.concat([train_tail[['ds', 'y']], test_df[['ds', 'y']]])
⋮----
# Plot model predictions
colors = plt.cm.Set3(np.linspace(0, 1, len(all_forecasts_data['models'])))
⋮----
ds_values = pd.to_datetime(all_forecasts_data['common']['ds'])
⋮----
# Plot styling
⋮----
# Add train/test split line if data exists
⋮----
split_date = test_df['ds'].iloc[0]
⋮----
def create_cv_individual_plots(cv_dir: Path, horizon: int)
⋮----
"""Create cross-validation plots for each model."""
cv_file = cv_dir / 'cv_df.csv'
⋮----
df_merged = pd.read_csv(cv_file)
⋮----
model_cols = [col for col in df_merged.columns
⋮----
cv_plot_dir = cv_dir / 'plot'
⋮----
# Plot actual values
⋮----
# Plot model predictions
cutoff_dates = sorted(df_merged['cutoff'].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(cutoff_dates)))
⋮----
window_data = df_merged[df_merged['cutoff'] == cutoff_date].sort_values('ds')
⋮----
def create_cv_visualizations_main()
⋮----
"""Main function to create cross-validation visualizations."""
```

## File: src/dataset/data_preparation.py
```python
"""
Data preparation module for neural forecasting pipeline.
"""
⋮----
def load_and_prepare_data(data_path=None)
⋮----
"""Load and prepare the dataset for forecasting."""
⋮----
# Use provided data_path or default to DATA_PATH
⋮----
data_path = DATA_PATH
⋮----
# Load data
df = pd.read_parquet(data_path)
⋮----
# Rename columns
df = df.rename(columns={DATE_COLUMN: DATE_RENAMED, TARGET_COLUMN: TARGET_RENAMED})
⋮----
# Add unique_id and convert date
⋮----
def difference_non_stationary_features(df: pd.DataFrame, exog_list: List[str]) -> pd.DataFrame
⋮----
"""
    Applies transformations to non-stationary features based on their category,
    as per the methodology in feature_transform.md.

    - Raw prices, blockchain data, and trend-following indicators are tested for
      stationarity. If non-stationary, they are transformed using log returns
      if appropriate (checked via Box-Cox), otherwise using simple differencing.
    - Oscillators, spreads, ratios, sentiment, and other bounded indicators are
      kept in their raw form.
    """
⋮----
df_transformed = df.copy()
⋮----
# Features to skip transformation (Group 4 & 5 from feature_transform.md)
# These are designed to be stationary or their raw value is meaningful.
SKIP_SUFFIXES = (
SKIP_EXACT = {
⋮----
# Check if the feature should be skipped based on its name
⋮----
# Fill NaNs for the purpose of testing.
series = df_transformed[col]
⋮----
series = series.fillna(method='ffill').fillna(method='bfill')
⋮----
# Stationarity Test (ADF)
p_value = adfuller(series.dropna())[1]
⋮----
# Use the original series from the copied dataframe for transformation
series_to_transform = df_transformed[col]
⋮----
# Check for non-positive values. Log/Box-Cox requires positive values.
⋮----
# Check for variance stability with Box-Cox to decide on log transform
⋮----
series_clean = series_to_transform.dropna()
# Box-Cox requires at least a few data points to work reliably.
⋮----
# Find optimal lambda for Box-Cox.
lambda_ = boxcox_normmax(series_clean)
⋮----
if abs(lambda_) < 0.5:  # Threshold for being "close to log"
⋮----
# Heuristic: For all-positive series where Box-Cox fails, log-return is a robust choice
# to stabilize variance, which is common for price-like series.
⋮----
def transform_target_to_log_return(df: pd.DataFrame) -> pd.DataFrame
⋮----
"""Transforms the target column 'y' to its log return."""
⋮----
# Ensure 'y' is positive before taking the log
⋮----
# Replace non-positive values with a small epsilon or handle as per domain knowledge
# For now, we will proceed, but this should be reviewed.
⋮----
df_transformed = df_transformed.dropna(subset=['y']).reset_index(drop=True)
⋮----
def split_data(df, horizon, test_length_multiplier)
⋮----
"""Split data into train and test sets."""
⋮----
test_length = horizon * test_length_multiplier
⋮----
# print(f"Forecast horizon (h) set to: {horizon} days")
⋮----
# Validate data length
⋮----
# Split data
train_df = df.iloc[:-test_length].copy()
test_df = df.iloc[-test_length:].copy()
⋮----
# Print information
⋮----
"""
    Loads data, splits it, and optionally applies transformations.
    
    Returns a tuple of (train_df, test_df, original_df_for_reference, hist_exog_list).
    """
# Load and perform initial preparation (renaming, etc.)
df = load_and_prepare_data(data_path=data_path)
⋮----
# Keep a copy of the original, untransformed data for reference (e.g., for back-transformation)
original_df_for_reference = df.copy()
⋮----
# Determine the list of historical exogenous features from the loaded data
hist_exog_list = get_historical_exogenous_features(df)
⋮----
df_for_split = df
⋮----
# Apply differencing to non-stationary exogenous features across the entire dataset
# This is done before splitting to maintain consistency in feature definitions.
df_for_split = difference_non_stationary_features(df, hist_exog_list)
⋮----
# Split the data into training and testing sets
⋮----
# If transformations are enabled, apply log-return transformation to the training set's target
⋮----
train_df = transform_target_to_log_return(train_df)
⋮----
# Ensure consistent column order in the final dataframes
final_exog_list = get_historical_exogenous_features(df_for_split)
train_df = train_df[['unique_id', 'ds', 'y'] + final_exog_list]
test_df = test_df[['unique_id', 'ds', 'y'] + final_exog_list]
⋮----
"""
    Enhanced data preparation for the pipeline with detailed logging and metadata.
    
    Args:
        horizon (int, optional): Forecast horizon in days. Defaults to HORIZON.
        test_length_multiplier (int, optional): Multiplier for test set length. Defaults to TEST_LENGTH_MULTIPLIER.
        data_path (str, optional): Path to data file. Defaults to DATA_PATH.
        apply_transformations (bool, optional): If False, skips log-return and differencing. Defaults to True.
    
    Returns:
        Tuple of (train_df, test_df, hist_exog_list, data_info_dict, original_df)
    """
⋮----
data_info = {
⋮----
def main()
⋮----
"""Main function to demonstrate prepare_pipeline_data usage."""
# print("--- Running raw data preparation with transformations (default) ---")
# train_df, test_df, _, _, _ = prepare_pipeline_data(
#     data_path=RAW_DATA_PATH,
#     apply_transformations=True)
# print("\nTransformed training data head:")
# print(train_df.head())
```

## File: src/models/neuralforecast/models.py
```python
# Standard library imports
⋮----
# Third-party imports
⋮----
# MSE,
# RMSE,
# MAPE,
# SMAPE,
# DistributionLoss
⋮----
# Local imports
⋮----
"""
    Get auto models for direct use in the pipeline.

    Args:
        horizon: Forecast horizon
        num_samples: Number of hyperparameter samples per model
        hist_exog_list: List of historical exogenous features

    Returns:
        List of auto model instances for direct use
    """
⋮----
configs = neural_auto_cfg(horizon)
search_alg = get_search_algorithm_class('hyperopt')
⋮----
init_config = {
⋮----
# "loss": DistributionLoss("StudentT", level=[90]),
⋮----
base_auto_config = {
⋮----
# "early_stop_patience_steps": 10,
# trainer_kwargs values
⋮----
transformer_config = configs["transformer"].copy()
transformer_config = {**base_auto_config, **transformer_config}
⋮----
models = [
⋮----
# AutoNBEATSx(**init_config, config=configs["nbeats"]),
⋮----
# AutoKAN(**init_config, config=configs["kan"]),
⋮----
"""
    Get normal model instances with best hyperparameters from a config file.

    Args:
        horizon: Forecast horizon.
        config_path: Path to the YAML configuration file with best hyperparameters.
        hist_exog_list: List of historical exogenous features.

    Returns:
        List of normal model instances.
    """
configs = load_yaml_to_dict(config_path)
⋮----
# Map of known neural models. Models not in this map will be skipped.
neural_model_map = {
⋮----
# "KAN": KAN,
⋮----
# Add "Auto" prefixed versions to map to the same models for flexibility
auto_model_map = {f"Auto{k}": v for k, v in neural_model_map.items()}
⋮----
models = []
⋮----
ModelClass = neural_model_map[model_name]
⋮----
# Basic model config
⋮----
# For models that require n_series, check against the actual class
⋮----
# Parameters to skip from CV config as they are not needed for final fitting
params_to_skip = [
⋮----
# Combine all configs
final_config = {**init_config, **model_config}
⋮----
# Print the final config
⋮----
model_instance = ModelClass(**final_config)
⋮----
# This is expected for ML models like AutoXGBoost, etc.
```

## File: src/pipelines/model_evaluation.py
```python
"""
Module for model evaluation (cross-validation and best config extraction) in the Auto Models workflow.
"""
⋮----
# Third-party import
⋮----
# Local import
⋮----
# Get dynamic directories based on HORIZON
⋮----
def back_transform_log_returns(cv_df: pd.DataFrame, original_prices: pd.DataFrame, model_names: List[str])
⋮----
"""
    Back-transforms log return forecasts to price forecasts recursively.
    """
⋮----
cv_df_transformed = cv_df.copy()
⋮----
# Prepare original prices dataframe for merging. This contains the actual prices.
price_reference = original_prices[['ds', 'y']].rename(columns={'y': 'price_at_cutoff'})
⋮----
# Ensure date columns are in datetime format for a reliable merge
⋮----
# Merge to get the last known true price (at the cutoff date) for each forecast window.
cv_df_transformed = pd.merge(
⋮----
# Sort to ensure correct order for cumulative calculations
cv_df_transformed = cv_df_transformed.sort_values(by=['unique_id', 'cutoff', 'ds']).reset_index(drop=True)
⋮----
# Back-transform 'y' (true log returns) to true prices.
# P_true(t) = P(cutoff) * exp(cumsum of y_log_returns from cutoff+1 to t)
⋮----
# Back-transform model predictions (forecasted log returns) to forecasted prices
⋮----
# P_hat(t) = P(cutoff) * cumprod(exp(y_hat_log_return)) from cutoff+1 to t
# We use a more numerically stable equivalent: P_hat(t) = P(cutoff) * exp(cumsum(y_hat_log_return))
# This avoids underflow from multiplying many small numbers from exp(large_negative_log_return).
group_transform = cv_df_transformed.groupby(['unique_id', 'cutoff'])[model].transform(
⋮----
# Drop the helper column
cv_df_transformed = cv_df_transformed.drop(columns=['price_at_cutoff'])
⋮----
"""
    Back-transforms final forecasted log returns to prices for a hold-out test set.

    This function is designed for a final evaluation scenario, where predictions
    are made for a contiguous block of future time steps. It uses the last known
    price from before the forecast period to recursively calculate predicted prices.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing model predictions in log returns.
                                    It must have a 'ds' column and columns for each model.
                                    The 'y' column should contain the actual prices for the forecast period.
        original_df (pd.DataFrame): The complete original DataFrame with untransformed prices.
                                    Used to find the last price before the forecast starts.
        model_names (List[str]): A list of model names corresponding to columns in `forecast_df`.

    Returns:
        pd.DataFrame: The forecast DataFrame with model predictions transformed to price scale.
                      The 'y' column remains unchanged.
    """
⋮----
transformed_df = forecast_df.copy()
⋮----
# Find the last date in the training data from the original dataframe, which should be
# one day before the first prediction date.
first_forecast_date = transformed_df['ds'].min()
last_train_date = first_forecast_date - pd.Timedelta(days=1)
⋮----
# Get the last actual price from the training period.
last_price_row = original_df.loc[original_df['ds'] == last_train_date]
⋮----
last_price = last_price_row['y'].iloc[0]
⋮----
# Back-transform each model's predicted log returns to prices.
⋮----
# P_hat(t) = P_last * exp(cumsum(y_hat_log_return))
log_returns = transformed_df[model].values
# Clip to prevent overflow from extreme predicted values
clipped_log_returns = np.clip(log_returns, a_min=-15, a_max=15)
# Calculate prices recursively from the last known price
⋮----
# The 'y' column in the input forecast_df already contains the true prices,
# so no transformation is needed for it.
⋮----
def perform_cross_validation(stat_models: List, neural_models: List, train_df: pd.DataFrame, original_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], Dict]
⋮----
"""Perform cross-validation for statistical and neural models and return their CV dataframes and metadata."""
all_cv_dfs = []
model_metadata = {}
fitted_objects = {'neural': None, 'stat': None}
⋮----
# Cross-validate All Statistical Models at once
⋮----
start_time = time.time()
⋮----
sf = StatsForecast(models=stat_models, freq='D', verbose=True)
cv_df = sf.cross_validation(
⋮----
# Extract model names exclusively from dataframe columns
auto_model_names = extract_model_names_from_columns(cv_df.columns.tolist())
⋮----
# Back-transform predictions from log-return to price
cv_df = back_transform_log_returns(cv_df, original_df, auto_model_names)
⋮----
training_time = time.time() - start_time
⋮----
# Populate metadata using only names from the dataframe
⋮----
error_msg = str(e)
⋮----
# Log a single error for the whole batch since we cannot get individual model names without a dataframe
⋮----
# Cross-validate All Neural Models at once
⋮----
nf = NeuralForecast(models=neural_models, freq='D', local_scaler_type='robust')
cv_df = nf.cross_validation(
⋮----
# Log a single error for the whole batch
⋮----
best_configs = save_best_configurations(fitted_objects, CV_DIR)
⋮----
# from src.models.mlforecast.models import get_ml_models
⋮----
# This main block now demonstrates only the Stat/Neural CV part
⋮----
stat_models = get_statistical_models(season_length=7)
neural_models = get_neural_models(horizon=HORIZON, num_samples=NUM_SAMPLES_PER_MODEL, hist_exog_list=hist_exog_list)
⋮----
# To see the results, we can consolidate and process them here
⋮----
consolidated_df = pd.concat(cv_dfs, ignore_index=True)
cv_results_df = process_cv_results(consolidated_df, original_df)
⋮----
# export the dataframes with timestamp
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
```

## File: config/base.py
```python
"""
Configuration settings.
"""
⋮----
# === Forecasting Configuration ===
HORIZON = 7
LEVELS = [95]
TEST_LENGTH_MULTIPLIER = 1
SEED = 42
⋮----
# === Data Configuration ===
RAW_DATA_PATH = 'data/final/raw_dataset.parquet'
DATA_PATH = f'data/final/feature_selection_{HORIZON}_mc.parquet'
DATE_COLUMN = 'Date'
DATE_RENAMED = 'ds'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
UNIQUE_ID_VALUE = 'Bitcoin'
⋮----
# === Rolling Forecast Configuration ===
ENABLE_ROLLING_FORECAST = True  # Enable rolling forecast for neural models when horizon < test_length
ROLLING_REFIT_FREQUENCY = 0  # Refit every N windows (1=every window, 3=every 3 windows, 0=no refit)
⋮----
# === Cross-validation Configuration ===
CV_N_WINDOWS = 30
CV_STEP_SIZE = HORIZON
INPUT_SIZE = 548
⋮----
# === Conformal Prediction Configuration ===
PI_N_WINDOWS_FOR_CONFORMAL = 20
⋮----
# === Hyperparameter Tuning Configuration ===
NUM_SAMPLES_PER_MODEL = 10
⋮----
# === Ray Configuration ===
RAY_ADDRESS = 'local'
RAY_NUM_CPUS = os.cpu_count()
RAY_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
⋮----
# === Output Directories ===
RESULTS_DIR = Path('results')
RESULTS_7D_DIR = RESULTS_DIR / 'results_7d'
RESULTS_14D_DIR = RESULTS_DIR / 'results_14d'
RESULTS_30D_DIR = RESULTS_DIR / 'results_30d'
RESULTS_60D_DIR = RESULTS_DIR / 'results_60d'
RESULTS_90D_DIR = RESULTS_DIR / 'results_90d'
⋮----
CV_7D_DIR = RESULTS_7D_DIR / 'cv'
CV_14D_DIR = RESULTS_14D_DIR / 'cv'
CV_30D_DIR = RESULTS_30D_DIR / 'cv'
CV_60D_DIR = RESULTS_60D_DIR / 'cv'
CV_90D_DIR = RESULTS_90D_DIR / 'cv'
⋮----
FINAL_7D_DIR = RESULTS_7D_DIR / 'final'
FINAL_14D_DIR = RESULTS_14D_DIR / 'final'
FINAL_30D_DIR = RESULTS_30D_DIR / 'final'
FINAL_60D_DIR = RESULTS_60D_DIR / 'final'
FINAL_90D_DIR = RESULTS_90D_DIR / 'final'
⋮----
PLOT_7D_DIR = FINAL_7D_DIR / 'plots'
PLOT_14D_DIR = FINAL_14D_DIR / 'plots'
PLOT_30D_DIR = FINAL_30D_DIR / 'plots'
PLOT_60D_DIR = FINAL_60D_DIR / 'plots'
PLOT_90D_DIR = FINAL_90D_DIR / 'plots'
⋮----
# For initial baseline establishment (as recommended in algo.txt)
BASELINE = ['random', 'hyperopt']
⋮----
# For efficient optimization (BOHB and Optuna with TPE)
EFFICIENT = ['bohb', 'optuna']
⋮----
# For noisy time series data
NOISY_DATA = ['hebo', 'random']
⋮----
# BOHB (Bayesian Optimization HyperBand):
# - bayesian optimization (efficiency) + hyperband (resource adaptiveness)
# -> require careful evaluate config
⋮----
# HEBO (Heteroscedastic Evolutionary Bayesian Optimization)
# - suit with noisy data
# -> hard to config
⋮----
# OptunaSearch
# - TPE (Tree-structured Parzen Estimator): a robust Bayesian optimization technique
# - CMA-ES (Covariance Matrix Adaptation Evolution Strategy): excellent for more complex, non-convex, or ill-conditioned search spaces.
# -> start with TPE
⋮----
# Suggestion:
# 1. Start with Random Search or HyperOptSearch: Run a small number of trials (e.g., 20-30) to get a feel for the search space and establish a baseline.
# 2. Move to BOHB or Optuna (with TPE/HyperBand Pruner): These are often my first choices for serious tuning due to their balance of performance and efficiency. HEBO is a strong alternative if you anticipate a noisy or complex landscape.
⋮----
def get_search_algorithm_class(algorithm: str) -> Any
```
