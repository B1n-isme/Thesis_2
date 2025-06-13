- input_size: number of read past points

- step_size: slide step, if step_size >= input_size -> no overlap

- windows_batch_size: process input_size together in batch then average out mistake
    -> too small: noisy
    -> too large: takes longer learning

- max_steps: num of train iterations (num_epoch)
- val_check_steps: freq in making validate -> smaller is better since spot quick early stopping but can be computation overhead
    -> these 2 often go with early_stop_patience_steps

# Point Forecast -> Probabilistic Forecast
Include prediction interval to account for uncertainty

- refactor final_plot_results_dict in model_forecasting.py to visualize mean in 1 plot to compare and interval for independent plots


# Suppress Ray logs
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)

# Cross-validation
3. Using cross-validation
3.1 Using n_windows
3.2 Using a validation and test set
3.3 Cross-validation with refit
3.4 Overlapping windows in cross-validation