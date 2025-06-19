# Feature Selection Workflow Explained

This document outlines the complete, four-stage feature selection workflow for the Bitcoin price forecasting project. This strategy is designed to be rigorous, starting with a broad stability check and progressively filtering down to a final, validated set of features.

---

### Stage 1: Stability Selection (The Foundation)

This is the first and most important step, performed on a **stationary target** (`log(price).diff()`) to ensure the statistical validity of the feature importances.

-   **What it is:** The `stability_selection` function is called separately for three different tree-based models: XGBoost, LightGBM, and Random Forest.
-   **How it works:**
    1.  For each model, it runs a series of bootstrap iterations (defaulting to 50). In each iteration, it trains the model on a random, contiguous chunk of the training data.
    2.  After training, it records the features that the model deemed important (specifically, those with an importance score in the top 50th percentile for that iteration).
    3.  It counts how many times each feature was selected across all bootstrap iterations.
    4.  Finally, it creates a list of "stable features" for that model—features that were selected in at least 70% of the bootstrap runs (`selection_threshold=0.7`).
-   **Output:** The result of this stage is **three distinct lists** of features, one for each model, containing the features that each model found to be consistently important.

---

### Stage 2: Consensus Building

This stage synthesizes the results from the three models to create a single, high-confidence list.

-   **What it is:** The `robust_comprehensive_selection` function takes the three lists of stable features from Stage 1 and builds a consensus.
-   **How it works:**
    1.  It counts how many of the three models selected each feature as "stable."
    2.  It then creates a new "consensus list" containing only the features that were selected by at least `min_consensus_level` models (the default is 1, but this can be made more stringent).
-   **Output:** A single, unified list of features that have passed the stability check for one or more models. This is your initial high-quality feature set.

---

### Stage 3: Multicollinearity Reduction

This stage takes the consensus list and removes redundant features. It uses the stability scores calculated in Stage 1 to make intelligent decisions about which features to keep.

-   **What it is:** The `handle_multicollinearity` function applies a two-step filtering process.
-   **How it works:**
    1.  **Correlation Filtering:** It first calculates the correlation matrix for all features in the consensus list. If two features have a correlation above a high threshold (e.g., 0.9), it checks their average stability scores (from Stage 1) and **discards the feature with the lower score**. This is a much more intelligent way to break ties than random selection.
    2.  **VIF (Variance Inflation Factor) Filtering:** On the remaining features, it iteratively calculates the VIF. If any feature has a VIF score above a threshold (e.g., 10), the feature with the highest VIF is removed, and the process repeats until all features are below the threshold.
-   **Output:** A filtered, non-redundant set of features where multicollinearity has been significantly reduced.

---

### Stage 4: Final Validation with Permutation Importance

This is the final and most critical validation step. It checks if the refined features are genuinely predictive of the **original, non-stationary target variable**.

-   **What it is:** The `permutation_importance_validation` function performs a final check on a hold-out validation set.
-   **How it works:**
    1.  It trains a final LightGBM model using the filtered, non-redundant features from Stage 3. Crucially, it trains this model to predict the **original `y` value**, not the stationary returns.
    2.  It then calculates Permutation Feature Importance (PFI). For each feature, it randomly shuffles that feature's values in the validation set and measures how much the model's predictive accuracy (MAE) degrades.
    3.  Features that cause a significant drop in performance (i.e., have a positive importance score) are considered valuable.
-   **Output:** The final, validated list of features. This list contains only the features that have passed all four stages of this rigorous pipeline, ensuring they are stable, non-redundant, and truly predictive of Bitcoin price movements.
