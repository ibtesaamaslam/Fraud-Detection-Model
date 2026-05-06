import pandas as pd
import joblib
import json
import numpy as np

# Load artifacts once
model = joblib.load("models/lightgbm_model.pkl")
scaler = joblib.load("data/processed/scaler.pkl")

with open("data/processed/feature_names.json", "r") as f:
    feature_names = json.load(f)


def prepare_input(raw_row):
    """
    ELI10:
    Takes raw transaction and converts it into EXACT format model expects
    """

    df = pd.DataFrame([raw_row])

    # Remove target if exists
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # SAME feature engineering as training
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_scaled"] = df["Amount"] / (df["Amount"].max() + 1)
    df["time_hours"] = df["Time"] / 3600

    # Match feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled


def predict_transaction(raw_row):
    """
    Returns prediction + probability
    """

    X = prepare_input(raw_row)

    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    return pred, prob