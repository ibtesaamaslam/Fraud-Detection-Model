import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

PROCESSED_PATH = "data/processed"

def preprocess_data(df):
    """
    ELI10:
    This function:
    1. Cleans data
    2. Creates extra features
    3. Splits X and y
    4. Scales features
    5. Saves everything for later use
    """

    df = df.copy()

    # ---------------------------
    # TARGET
    # ---------------------------
    y = df["Class"]
    X = df.drop(columns=["Class"])

    # ---------------------------
    # FEATURE ENGINEERING (IMPORTANT)
    # ---------------------------

    # Example engineered features (these caused your mismatch)
    X["amount_log"] = np.log1p(X["Amount"])
    X["amount_scaled"] = X["Amount"] / (X["Amount"].max() + 1)

    X["time_hours"] = X["Time"] / 3600

    # ---------------------------
    # SAVE FEATURE NAMES (CRITICAL)
    # ---------------------------
    feature_names = X.columns.tolist()

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    with open(f"{PROCESSED_PATH}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ---------------------------
    # SCALE FEATURES
    # ---------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, f"{PROCESSED_PATH}/scaler.pkl")

    # ---------------------------
    # SAVE PROCESSED DATA
    # ---------------------------
    pd.DataFrame(X_scaled).to_csv(f"{PROCESSED_PATH}/X_scaled.csv", index=False)
    y.to_csv(f"{PROCESSED_PATH}/y.csv", index=False)

    return X_scaled, y, feature_names, scaler