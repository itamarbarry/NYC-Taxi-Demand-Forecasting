"""
NYC TLC Yellow Taxi -- Data Splitting
======================================

Splits the aggregated demand dataset into train and test sets using a DATE-BASED split.
Following the project requirements:
- Train Set: 2024 and 2025 data.
- Test Set: 2026 data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Default sizes for subsampling experiments
TRAIN_SAMPLE_SIZE = 50_000
TEST_SAMPLE_SIZE  = 10_000


def split_train_test(parquet_path, output_dir, cutoff_date='2026-01-01', end_date='2026-03-01'):
    """
    Split the demand dataset into training (before cutoff) and testing (after cutoff).
    The test set is also capped at end_date (e.g., to keep it strictly Jan-Feb 2026).
    """
    df = pd.read_parquet(parquet_path)

    # Sort chronologically
    print(f"  Splitting data: Train < {cutoff_date}, Test {cutoff_date} -> {end_date}...")
    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    
    # Filter based on the cutoff date
    train_df = df[df['pickup_datetime'] < cutoff_date].reset_index(drop=True)
    test_df  = df[
        (df['pickup_datetime'] >= cutoff_date) & 
        (df['pickup_datetime'] < end_date)
    ].reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path  = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)

    print(f"  Total rows : {len(df):>10,}")
    print(f"  Train rows : {len(train_df):>10,}  "
          f"({train_df['pickup_datetime'].min().date()} -> "
          f"{train_df['pickup_datetime'].max().date()})")
    print(f"  Test rows  : {len(test_df):>10,}  "
          f"({test_df['pickup_datetime'].min().date()} -> "
          f"{test_df['pickup_datetime'].max().date()})")

    return str(train_path), str(test_path)


def subsample_splits(train_path, test_path,
                     train_n=TRAIN_SAMPLE_SIZE, test_n=TEST_SAMPLE_SIZE,
                     random_state=42):
    """
    Draw a random subsample from each split to keep training runtime feasible.
    """
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    train_n = min(train_n, len(train_df))
    test_n  = min(test_n,  len(test_df))

    train_sample = train_df.sample(n=train_n, random_state=random_state).reset_index(drop=True)
    test_sample  = test_df.sample( n=test_n,  random_state=random_state).reset_index(drop=True)

    print(f"  Train sample : {len(train_sample):>6,}  (from {len(train_df):,})")
    print(f"  Test sample  : {len(test_sample):>6,}  (from {len(test_df):,})")

    return train_sample, test_sample

def extract_drift_reference_set(train_df: pd.DataFrame, seed: int = 42):
    """
    Take 1 random week from each month in 2024-2025.
    Remove these weeks from the train set and return them as 'drift_evaluation' set.
    """
    np.random.seed(seed)
    df = train_df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Ensure we only look at 2024-2025
    mask_24_25 = (df['pickup_datetime'] >= '2024-01-01') & (df['pickup_datetime'] < '2026-01-01')
    df_24_25 = df[mask_24_25].copy()
    
    drift_eval_indices = []
    
    # Group by year and month
    years_months = df_24_25.groupby([df_24_25['pickup_datetime'].dt.year, df_24_25['pickup_datetime'].dt.month]).groups.keys()
    
    for year, month in sorted(years_months):
        group = df_24_25[(df_24_25['pickup_datetime'].dt.year == year) & (df_24_25['pickup_datetime'].dt.month == month)]
        
        # Randomly pick a week (1-4)
        week_num = np.random.randint(1, 5) # 1, 2, 3, or 4
        
        start_day = (week_num - 1) * 7 + 1
        end_day = week_num * 7
        
        # Get indices for this week
        week_indices = group[
            (group['pickup_datetime'].dt.day >= start_day) & 
            (group['pickup_datetime'].dt.day <= end_day)
        ].index
        
        drift_eval_indices.extend(week_indices)
    
    drift_eval_df = df.loc[drift_eval_indices].reset_index(drop=True)
    train_df = df.drop(index=drift_eval_indices).reset_index(drop=True)
    
    # Strictly limit train_df to 2024-2025
    train_df = train_df[train_df['pickup_datetime'] < '2026-01-01'].reset_index(drop=True)
    
    print(f"  Drift Eval set: {len(drift_eval_df):>10,} rows (extracted from 2024-2025)")
    print(f"  Reduced Train : {len(train_df):>10,} rows")
    
    return train_df, drift_eval_df
