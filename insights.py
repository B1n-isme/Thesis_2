import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_metrics_data():
    """Load and combine metrics data from all horizon files."""
    horizons = [7, 14, 30, 60, 90]
    base_path = 'results/results_{}d/cv/cv_metrics_2.csv'
    
    all_metrics = []
    for h in horizons:
        file_path = base_path.format(h)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['horizon'] = h
            all_metrics.append(df)
        else:
            print(f"Warning: File not found for horizon {h} at {file_path}")

    if not all_metrics:
        print("No metric files found. Exiting.")
        return None
    return pd.concat(all_metrics, ignore_index=True)

def plot_metrics_vs_horizon(combined_df):
    """Create line plots for each metric vs. horizon."""
    metrics_to_plot = ['mae', 'rmse', 'mase', 'da', 'theil_u', 'training_time']
    output_dir = 'results/horizon_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics_to_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        
        sns.lineplot(data=combined_df, x='horizon', y=metric, hue='model_name', marker='o', dashes=False)
        
        plt.title(f'{metric.replace("_", " ").title()} vs. Forecasting Horizon', fontsize=16)
        plt.xlabel('Forecasting Horizon (Days)', fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xticks([7, 14, 30, 60, 90])
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_vs_horizon.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

def calculate_degradation_rates(combined_df):
    """Calculate and rank models based on metric degradation across horizons."""
    metrics_to_analyze = ['mase', 'da']
    results = {}

    for metric in metrics_to_analyze:
        pivot_df = combined_df.pivot(index='model_name', columns='horizon', values=metric)
        pct_change_df = pivot_df.pct_change(axis='columns')
        results[f'avg_{metric}_degradation_pct'] = pct_change_df.mean(axis='columns') * 100

    result_df = pd.DataFrame(results)
    result_df['mase_rank'] = result_df['avg_mase_degradation_pct'].rank(ascending=True)
    result_df['da_rank'] = result_df['avg_da_degradation_pct'].rank(ascending=False)
    result_df['combined_rank'] = result_df[['mase_rank', 'da_rank']].mean(axis=1).rank()

    print("Model Performance Degradation Analysis")
    print("-" * 60)
    print(result_df.sort_values('combined_rank'))
    print("\nNotes:")
    print(" - Lower 'avg_mase_degradation_pct' is better.")
    print(" - Higher 'avg_da_degradation_pct' is better (less degradation).")
    print(" - 'combined_rank' provides an overall stability ranking.")

if __name__ == '__main__':
    df = load_metrics_data()
    if df is not None:
        # plot_metrics_vs_horizon(df)
        calculate_degradation_rates(df)

# Of course. Based on the detailed metrics in the analysis report, here is a list of specific research questions you could explore, categorized by theme.

### Theme 1: Performance Degradation and Model Robustness

# * **Question 1:** To what degree does the forecast accuracy (MASE) of each model degrade as the forecast horizon extends from short-term (14 days) to mid-range (60 days), and what does this reveal about their relative robustness?
#     * [cite_start]*This can be answered by analyzing the "% Increase in MASE" presented in Table 2, which shows SARIMAX had the lowest increase (37.3%) while Theta had the highest (96.6%)[cite: 98, 99].*

# * **Question 2:** Which model architecture (e.g., linear statistical, smoother, attention-based) is most susceptible to catastrophic failure at long horizons (90 days), and why do its core assumptions break down?
#     * [cite_start]*The report shows SARIMAX's MASE skyrockets to 7.27 at 90 days[cite: 111]. [cite_start]The analysis explains this is due to its rigid, linear structure being unable to adapt to new market regimes, a fundamental violation of its stationarity assumption[cite: 114, 115, 116].*

# ### Theme 2: The Trade-Off Between Direction and Magnitude

# * **Question 3:** Is there a quantifiable trade-off between a model's ability to predict directional accuracy (DA) and its accuracy in predicting magnitude (MASE) in short-term forecasting?
#     * *This can be answered by comparing model rankings. [cite_start]At the 7-day horizon, SARIMAX ranks #1 in Directional Accuracy (72.25%) but last (#7) in MASE, while ETS ranks #1 in MASE (1.721) but 4th in Directional Accuracy, demonstrating a clear paradox[cite: 67, 68].*

# * **Question 4:** How does the reliability of the directional signal provided by the top-performing directional model (SARIMAX) evolve as the forecast horizon extends from 7 to 90 days?
#     * *The data shows a clear decay. [cite_start]SARIMAX's DA falls systematically from a high of 72.25% at 7 days to 64.20% at 14 days, 57.6% at 30 days, and finally to 51.17% at 90 days, which is statistically indistinguishable from a coin toss[cite: 43, 75, 112].*

# ### Theme 3: Computational Cost vs. Performance Gain

# * **Question 5:** What is the relationship between a model's computational cost (training time) and its forecast accuracy (MASE) across different horizons?
#     * *The report indicates a poor relationship. [cite_start]The most computationally expensive models, TFT and SARIMAX, do not provide the best magnitude forecasts and are described as inefficient[cite: 69, 171]. [cite_start]For example, at the 30-day horizon, TFT required over 13 hours (47,202 seconds) to train while delivering worse magnitude accuracy than ETS, which trained in under an hour[cite: 93, 95].*

# * **Question 6:** Do architecturally complex deep learning models like the Temporal Fusion Transformer (TFT) provide a justifiable return on investment (accuracy gain per second of training time) over simpler statistical models like ETS?
#     * [cite_start]*The report concludes they do not[cite: 15]. [cite_start]The massive computational burden of TFT is not met with a "corresponding, decisive, or consistent improvement in forecasting accuracy" over simpler and faster alternatives like ETS, especially in a univariate context[cite: 15].*

# ### Theme 4: Synthesizing Findings for Practical Application

# * **Question 7:** Based on the observed "decoupling" of direction and magnitude signals, what is the empirical justification for proposing a hybrid modeling approach for future research?
#     * [cite_start]*The report explicitly recommends this[cite: 195]. [cite_start]The justification lies in combining the strengths of different models: using a model proven to be a specialist in short-term direction like SARIMAX for a Stage 1 directional signal, and then using that signal as an input into a separate Stage 2 model (like TCN or GARCH) to predict the magnitude, thereby addressing the distinct failure modes of each model[cite: 196, 197, 198].*

# * **Question 8:** To what extent does the collective failure of all seven models (MASE > 1.0) across all horizons provide empirical evidence for the Efficient Market Hypothesis in the context of Bitcoin price prediction?
#     * [cite_start]*The report presents this as its "most consequential finding"[cite: 5]. [cite_start]The fact that no model—from the simplest to the most complex—could produce a forecast that was, on average, better than a naive random walk is described as a "stark affirmation of the Efficient Market Hypothesis"[cite: 6, 164, 166].*

# My Recommendation for Your Bachelor's Thesis (The Hybrid Approach)

# For a thesis, you have a golden opportunity to show you understand this nuance. I recommend you adopt a hybrid of both philosophies, which is what the source you found advocates for. This will make your work more robust and impressive.

# Here is your new, enlightened workflow:

# Run Your Cross-Validation (As Planned): Perform your expanding window CV. Analyze the results and declare a "CV Winner." This demonstrates methodological rigor.

# In your thesis: "Based on the cross-validation results summarized in Table 1, Model A was selected as the most promising model due to its superior average performance across the five backtest windows."

# Run Top Contenders on the Holdout Set (The Pragmatic Step): Take your CV winner (Model A), the best runner-up (Model B), and a Naive Baseline. Retrain all three on the full training set and generate forecasts for the holdout set. Create the exact table your source suggested.

# Analyze the Holdout Results (This is where you shine):

# Scenario 1: The CV Winner also wins on the Holdout. This is the perfect outcome.

# In your thesis: "To confirm this finding, the top models were evaluated on a final holdout set. As shown in Table 2, Model A also outperformed all contenders on this unseen data, confirming its superior generalization capability. Its MAE of 250 on the holdout set provides a final, validated measure of its performance."

# Scenario 2: The Runner-Up wins on the Holdout. This is a more interesting and sophisticated finding.

# In your thesis: "An important finding emerged during the final validation stage. While Model A was the winner during cross-validation, Model B achieved a lower MAE on the holdout set (Table 2). This suggests that Model A may have been slightly overfit to the historical patterns in the training folds. While the CV process is crucial for robust selection, the holdout result indicates that for this specific future period, Model B was empirically the most accurate. Therefore, we identify Model B as the final recommended model, while acknowledging this informative flip in the rankings."

# Conclusion: The source you found gives excellent, practical advice. Running your top models on the holdout set is not "wrong." It provides richer context and a final competitive benchmark. Your goal is to find the best model, and by having two scores (CV and Holdout), you can have a much more intelligent discussion about what "best" truly means: is it the most robust on average (CV winner) or the best on the single most recent test (Holdout winner)?

# As anticipated, the performance on the final holdout set (MASE = 0.85) represents a slight degradation compared to the average performance observed during cross-validation (Average MASE = 0.78). This is an expected and well-documented phenomenon in time series forecasting for two primary reasons. Firstly, it reflects the 'winner's curse' inherent in any model selection process, where the holdout score provides a less biased estimate of performance than the score that led to the model's selection. Secondly, it highlights the non-stationary nature of financial data, where the holdout period inevitably contains market dynamics not fully represented in the historical training data. Therefore, the CV score should be interpreted as the basis for our model selection, while the holdout score serves as the most realistic estimate of future real-world performance.