"""Load raw, processed, or synthetic art taste data into pandas DataFrames.

This module will hold file readers and schema-aware loading helpers for artwork
metadata, artist records, user preferences, ratings, and other MVP inputs.
"""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)
