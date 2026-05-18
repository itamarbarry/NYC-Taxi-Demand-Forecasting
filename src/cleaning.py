"""
NYC TLC Yellow Taxi -- Demand Forecasting Cleaning
=================================================

Applies targeted cleaning logic for demand forecasting.
Focuses only on tpep_pickup_datetime and PULocationID.
"""

import pandas as pd
from pathlib import Path

# Columns required for the demand forecasting pipeline
CRITICAL_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
]

def drop_critical_nulls(df):
    """Drop rows where pickup time or location is missing."""
    before = len(df)
    df = df.dropna(subset=[c for c in CRITICAL_COLUMNS if c in df.columns])
    dropped = before - len(df)
    if dropped:
        print(f"  drop_critical_nulls  : dropped {dropped:,} rows")
    return df

def filter_training_window(df):
    """
    Restrict data to the 2024-2026 window.
    Removes early outliers (e.g., 2002) and future outliers (beyond Feb 2026).
    """
    before = len(df)
    # Keep only data from 2024 onwards. Upper bound is current year + 1 to allow for monitoring.
    df = df[
        (df["tpep_pickup_datetime"] >= "2024-01-01") & 
        (df["tpep_pickup_datetime"] < "2026-04-01")
    ]
    dropped = before - len(df)
    if dropped:
        print(f"  filter_training_window : dropped {dropped:,} date outliers")
    return df

def filter_valid_locations(df):
    """
    Keep only valid NYC taxi zones (1-263).
    Removes unknown (264) and outside-NYC (265) locations.
    """
    before = len(df)
    if "PULocationID" in df.columns:
        df = df[df["PULocationID"].between(1, 263)]
    dropped = before - len(df)
    if dropped:
        print(f"  filter_valid_locations : dropped {dropped:,} invalid location IDs")
    return df

def select_relevant_columns(df):
    """Keep only the columns needed for aggregation."""
    cols = [c for c in CRITICAL_COLUMNS if c in df.columns]
    return df[cols]

def clean_dataframe(df):
    """Run the full cleaning sequence for demand forecasting."""
    print(f"  Input  rows : {len(df):,}")
    df = drop_critical_nulls(df)
    df = filter_valid_locations(df)
    df = filter_training_window(df)
    df = select_relevant_columns(df)
    print(f"  Output rows : {len(df):,}")
    return df.reset_index(drop=True)

def clean_parquet(input_path, output_path=None):
    """Load, clean, and optionally save a parquet file."""
    # Only load the columns that exist in the raw file
    RAW_COLS = ["tpep_pickup_datetime", "PULocationID"]
    df = pd.read_parquet(input_path, columns=RAW_COLS)
    df_clean = clean_dataframe(df)

    if output_path is None:
        # Default to data/processed as in Act 3
        output_path = Path("data/processed") / f"{Path(input_path).stem}_clean.parquet"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"  Saved cleaned file -> {output_path}")

    return df_clean
