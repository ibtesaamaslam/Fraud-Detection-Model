# evaluate_model.py
# ------------
# Checks how good the trained models are.
# We use better metrics than accuracy because fraud detection is imbalanced.

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import THRESHOLD, METRICS_PATH
from src.utils.helpers import ensure_project_directories
from src.visualization.plots import (
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
)


# ---------------------------------------------------------------------------
# Model evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every trained model and return a ranked comparison table.

    Checks how good the trained models are.
    We use better metrics than accuracy because fraud detection is imbalanced
    — a model that always predicts legitimate would score 99% accuracy but
    catch zero fraud cases.

    For each model the function:
        1) Predicts fraud probabilities on the held-out test set.
        2) Converts probabilities to binary labels using THRESHOLD.
        3) Computes accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
        4) Saves confusion matrix, ROC curve and PR curve plots to disk.
        5) Collects all metrics into a DataFrame sorted by ROC-AUC.

    Args:
        models:  Dict mapping model name to a fitted scikit-learn estimator.
        X_test:  Scaled test feature matrix.
        y_test:  Binary target series (0 = legitimate, 1 = fraud).

    Returns:
        A DataFrame with one row per model, sorted by ROC-AUC descending,
        and persisted to METRICS_PATH as a CSV file.
    """
    # Ensure all required output directories exist before saving plots
    ensure_project_directories()

    rows = []

    for name, model in models.items():
        print(f"\nEvaluating: {name}")

        # ---------------------------------------------------------------------------
        # Step 1 — Generate predictions
        # ---------------------------------------------------------------------------

        # Extract the probability of the positive (fraud) class
        y_proba = model.predict_proba(X_test)[:, 1]

        # Apply the configured threshold to convert probabilities to binary labels
        # A lower threshold catches more fraud but increases false positives
        y_pred = (y_proba >= THRESHOLD).astype(int)

        # ---------------------------------------------------------------------------
        # Step 2 — Compute metrics
        # ---------------------------------------------------------------------------

        # Standard classification metrics — precision and recall matter most for fraud
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        # Ranking metrics that evaluate probability scores, not just hard labels
        roc = roc_auc_score(y_test, y_proba)
        pr  = average_precision_score(y_test, y_proba)

        # Confusion matrix used for the plot — shows TP, FP, TN, FN breakdown
        cm = confusion_matrix(y_test, y_pred)

        print(classification_report(y_test, y_pred, digits=4))
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

        # ---------------------------------------------------------------------------
        # Step 3 — Save diagnostic plots
        # ---------------------------------------------------------------------------

        # Persist one set of plots per model for visual comparison in the report
        save_confusion_matrix_plot(cm, name)
        save_roc_curve(y_test, y_proba, name)
        save_pr_curve(y_test, y_proba, name)

        # ---------------------------------------------------------------------------
        # Step 4 — Collect metrics row
        # ---------------------------------------------------------------------------

        rows.append({
            "model":     name,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   roc,
            "pr_auc":    pr,
        })

    # ---------------------------------------------------------------------------
    # Step 5 — Build and save comparison table
    # ---------------------------------------------------------------------------

    # Sort by ROC-AUC descending so the best model appears at the top
    metrics_df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)

    # Persist the metrics table so results are reproducible without re-running
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nModel comparison table:")
    print(metrics_df)

    return metrics_df