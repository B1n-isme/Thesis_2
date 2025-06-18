1. Primary Selection with Stability Selection: Use Stability Selection as your primary, and sole, initial feature selector. This will give you a core set of robust and reliable features.
    Configuration: Ensure you are using a time-aware bootstrapping method, such as the moving block bootstrap, to preserve the temporal structure of your data. 
2. Refinement with Multicollinearity Reduction: After obtaining the stable feature set, apply your planned two-stage multicollinearity reduction process:
    Stage 1: Use Spearman's rank correlation and hierarchical clustering to group highly correlated features.  From each cluster, select the feature with the highest stability score from the previous step.

    Stage 2: Apply an iterative Variance Inflation Factor (VIF) check to remove any remaining multi-way collinearity. 
3. Final Validation with Permutation Feature Importance (PFI): Use PFI on a held-out validation set as a final check. This will confirm that the selected features contribute positively to the model's generalization performance.  Pay close attention to and consider removing any features with negative PFI scores, as they may be indicative of overfitting. 