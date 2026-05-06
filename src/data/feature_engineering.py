# feature_engineering.py
# -----------
# Feature engineering for raw transaction data.
# Feature engineering means making smarter columns from the raw columns.
# Raw columns are good, but smart features often help the model learn better.

import numpy as np
import pandas as pd

from src.utils.config import LARGE_AMOUNT_THRESHOLD


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engineered features from raw transaction columns.

    Feature engineering means making smarter columns from the raw columns.
    Raw columns are good, but smart features often help the model learn better.
    Transforms time and amount into model-friendly representations,
    and adds binary flags for behavioural patterns associated with fraud.

    Args:
        df: Raw transaction DataFrame. Must contain 'Time' and 'Amount' columns.

    Returns:
        A copy of the input DataFrame with the following additional columns:
            - Hour:            Hour of day (0-23) derived from transaction time in seconds.
            - Amount_Log:      Log-scaled transaction amount via log1p for skew reduction.
            - Is_Night:        1 if the transaction occurred between midnight and 05:59.
            - Is_Large_Amount: 1 if the transaction amount exceeds LARGE_AMOUNT_THRESHOLD.

    Raises:
        ValueError: If 'Time' or 'Amount' columns are absent from the input DataFrame.
    """
    # Work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()

    # Validate that all required source columns are present before proceeding
    required = {"Time", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    # ---------------------------------------------------------------------------
    # Time features
    # ---------------------------------------------------------------------------

    # Convert elapsed seconds into hour-of-day (0-23)
    # e.g. 3,600 seconds -> hour 1, 90,000 seconds -> hour 1 (next day)
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

    # ---------------------------------------------------------------------------
    # Amount features
    # ---------------------------------------------------------------------------

    # Apply log1p to compress large values and reduce right skew
    # Big numbers are compressed so the model sees differences more clearly
    # clip(lower=0) guards against any unexpected negative amounts
    df["Amount_Log"] = np.log1p(df["Amount"].clip(lower=0))

    # ---------------------------------------------------------------------------
    # Behavioural flags
    # ---------------------------------------------------------------------------

    # Flag transactions occurring in the early hours — a known fraud-risk window
    # Spending money at unusual hours may indicate suspicious activity
    df["Is_Night"] = df["Hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # Flag transactions exceeding the configured high-value threshold
    # If the amount is above the chosen line, we mark it as large
    df["Is_Large_Amount"] = (df["Amount"] >= LARGE_AMOUNT_THRESHOLD).astype(int)

    return df