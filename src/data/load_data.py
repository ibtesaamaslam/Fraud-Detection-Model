# load_data.py
# ---------
# Loads the raw Kaggle transactions dataset from disk.
# Raw data is untouched and unprocessed at this stage of the pipeline.

import pandas as pd

from src.utils.config import RAW_DATA_PATH


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the raw transactions dataset from a CSV file.

    Reads from the path defined in config.RAW_DATA_PATH and logs
    the resulting shape to stdout for quick sanity-checking.

    Returns:
        A DataFrame containing the raw, unprocessed transaction records.

    Raises:
        FileNotFoundError: If the CSV file does not exist at RAW_DATA_PATH.
    """
    # Read the CSV into a DataFrame — no transformations applied at this stage
    df = pd.read_csv(RAW_DATA_PATH)

    # Log shape to stdout as a quick sanity check after loading
    print(f"Data loaded successfully. Shape = {df.shape}")

    return df