# shap_explainer.py
# -----------------
# Explains why the model made a decision.
# The model says "fraud" — SHAP shows which features pushed it toward that answer.

import joblib

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.config import BEST_MODEL_PATH, FIGURES_DIR


# ---------------------------------------------------------------------------
# SHAP text summary
# ---------------------------------------------------------------------------

def get_shap_summary(model, sample_df: pd.DataFrame, top_n: int = 5):
    """Return a short text summary of the most important SHAP features.

    Explains why the model made a decision by showing which features pushed
    the prediction toward fraud or toward safe, and by how much.

    Example output:
        "Hour (+0.9) pushes toward fraud, Amount_Log (+0.7) pushes toward fraud,
         Is_Night (-0.5) pushes toward safe"

    Args:
        model:     A fitted tree-based estimator compatible with shap.TreeExplainer.
        sample_df: A single-row or small DataFrame of scaled input features.
        top_n:     Number of top features to include in the summary. Defaults to 5.

    Returns:
        A tuple of:
            - summary:     Human-readable string of top feature contributions.
            - shap_values: Raw SHAP values array, or None on failure.
            - explainer:   The fitted shap.TreeExplainer instance, or None on failure.
    """
    try:
        # Build a tree-based explainer — faster and more accurate than KernelExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)

        # Different SHAP versions return slightly different shapes — handle both
        if isinstance(shap_values, list):
            # For binary classification, index 1 corresponds to the fraud class
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            values = shap_values[0]

        # Pair each feature name with its SHAP contribution value
        feature_values = list(zip(sample_df.columns, values))

        # Sort by absolute contribution so the most influential features come first
        feature_values = sorted(feature_values, key=lambda x: abs(x[1]), reverse=True)

        # Keep only the top N most influential features
        top_features = feature_values[:top_n]

        # Build a readable string showing direction and magnitude per feature
        summary_parts = []
        for feature, value in top_features:
            direction = "pushes toward fraud" if value > 0 else "pushes toward safe"
            summary_parts.append(f"{feature} ({value:+.3f}) {direction}")

        return ", ".join(summary_parts), shap_values, explainer

    except Exception as e:
        # Log the failure and return safe defaults so the app does not crash
        print(f"SHAP failed: {e}")
        return "No SHAP explanation available.", None, None


# ---------------------------------------------------------------------------
# SHAP summary plot
# ---------------------------------------------------------------------------

def save_shap_summary_plot(model, sample_df: pd.DataFrame, output_name: str = "shap_summary.png"):
    """Save a SHAP summary plot to the reports/figures directory.

    Produces a beeswarm plot showing how each feature influences model output
    across all samples in sample_df. Saved as a high-resolution PNG.

    Args:
        model:       A fitted tree-based estimator compatible with shap.TreeExplainer.
        sample_df:   DataFrame of scaled input features to explain.
        output_name: Filename for the saved plot. Defaults to 'shap_summary.png'.
    """
    # Ensure the figures output directory exists before saving
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Build the explainer and compute SHAP values for the full sample
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)

        plt.figure()

        # Use index 1 for the fraud class when SHAP returns a list of arrays
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], sample_df, show=False)
        else:
            shap.summary_plot(shap_values, sample_df, show=False)

        # Tighten layout before saving to avoid clipped labels
        plt.tight_layout()

        # Save at 200 DPI for crisp report-quality output
        plt.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")

        # Close the figure to free memory — important in long-running pipelines
        plt.close()

    except Exception as e:
        # Log the failure gracefully — a missing plot should not stop the pipeline
        print(f"Could not save SHAP plot: {e}")