# train_model.py
# ----------
# Trains multiple models and saves them to disk.
# We compare models so we can choose the best one.

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config import (
    MODELS_DIR,
    RANDOM_STATE,
    SMOTE_RATIO,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    LGBM_MODEL_PATH,
)
from src.utils.helpers import ensure_project_directories


# ---------------------------------------------------------------------------
# Model training pipeline
# ---------------------------------------------------------------------------

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """Train several classifiers after balancing the training data with SMOTE.

    We compare models so we can choose the best one.
    SMOTE makes synthetic extra fraud examples so the model does not ignore
    the minority class. Then we train a few different models and save them
    to disk for later evaluation and comparison.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Binary target series (0 = legitimate, 1 = fraud).

    Returns:
        A tuple of:
            - trained_models: Dict mapping model name to fitted estimator.
            - X_res:          Resampled feature matrix after SMOTE.
            - y_res:          Resampled target series after SMOTE.
    """
    # Ensure all required output directories exist before saving models
    ensure_project_directories()

    # ---------------------------------------------------------------------------
    # Step 1 — Balance the training data with SMOTE
    # ---------------------------------------------------------------------------

    # SMOTE generates synthetic fraud samples so the model sees enough examples
    # sampling_strategy=SMOTE_RATIO means fraud will be 20% of the majority class
    smote = SMOTE(
        sampling_strategy=SMOTE_RATIO,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # ---------------------------------------------------------------------------
    # Step 2 — Define candidate models
    # ---------------------------------------------------------------------------

    # Each model uses a different learning strategy — we compare them after training
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",        # Penalises fraud misclassification more
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,                      # Use all available CPU cores
            class_weight="balanced_subsample",
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,             # Small steps to avoid overfitting
            max_depth=5,
            subsample=0.8,                  # Use 80% of rows per tree
            colsample_bytree=0.8,           # Use 80% of features per tree
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",             # Faster histogram-based training
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,                      # Use all available CPU cores
        ),
    }

    # ---------------------------------------------------------------------------
    # Step 3 — Train and persist each model
    # ---------------------------------------------------------------------------

    trained_models = {}

    for name, model in models.items():
        print(f"Training model: {name}")

        # Fit the model on the SMOTE-resampled training data
        model.fit(X_res, y_res)
        trained_models[name] = model

        # Save each model to its configured path for later evaluation
        if name == "random_forest":
            joblib.dump(model, RF_MODEL_PATH)
        elif name == "xgboost":
            joblib.dump(model, XGB_MODEL_PATH)
        elif name == "lightgbm":
            joblib.dump(model, LGBM_MODEL_PATH)

    print("All models trained and saved.")
    return trained_models, X_res, y_res