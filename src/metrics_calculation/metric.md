MAE: average magnitude of error
RMSE (squared-error-penalized magnitude) over MSE: more interpretable as it's in the original unit
RMSE/MAE ratio: diagnose the variability & extremity of model's forecast errors
MASE over MAPE & sMAPE: 
- robust, stable, and theoretically sound scale-independent
- avoid asymmetry and division-by-zero issues of percentage error

Why Other Metrics Were Excluded from the Core List

To create an effective report, you must be decisive about what you leave out. Here is the justification for excluding the other metrics from your list, which you can use to defend your choices.

Removed MSE: Redundant with RMSE. RMSE is more interpretable as it's in the original unit (dollars), not dollars-squared. You lose no information by dropping MSE in favor of RMSE in a final report.

Removed MAPE and sMAPE: These are made redundant and are outright replaced by MASE. MASE is a more robust, stable, and theoretically sound scale-independent metric. It avoids the asymmetry and division-by-zero issues of percentage errors and provides a more meaningful benchmark comparison.

Removed RMAE, MSSE, RMSSE: These are all "scaled error" metrics, similar in spirit to MASE.

MASE is the most common and interpretable of the group.

RMSSE (Root Mean Squared Scaled Error) is an excellent metric, but it tells a similar story to MASE, just with a focus on large errors (the same way RMSE relates to MAE). For a refined core list, MASE is sufficient. If you want one "advanced" metric, you could add RMSSE, but be prepared to explain its nuanced difference from MASE.

Removed Theil's U: This is a classic metric, but it is conceptually replaced by modern scaled-error metrics. Theil's U compares the RMSE of your model to a naïve model's RMSE. MASE and RMSSE do the same thing (compare to a naïve model) but are generally considered more robust for modern forecasting tasks. Including both would be highly redundant.