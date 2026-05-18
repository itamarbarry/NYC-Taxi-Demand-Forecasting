"""
NYC TLC Yellow Taxi -- Demand Transformation
============================================

This module transforms trip-level records into aggregated demand buckets.
It implements the logic from Act 4 of the EDA notebook:
1. 30-minute temporal rounding.
2. Demand aggregation per Time-Location bucket.
3. Full grid reindexing to capture zero-demand intervals.
4. Basic temporal feature extraction (hour, day_of_week, month).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def transform_to_demand(df):
    """
    Transform trip-level DataFrame into demand-level buckets.
    
    Args:
        df: Cleaned DataFrame with 'tpep_pickup_datetime' and 'PULocationID'.
        
    Returns:
        demand_df: Aggregated DataFrame with 30-min demand buckets.
    """
    df = df.copy()
    
    # 1. 30-minute temporal rounding
    # [0,15)->:00, [15,45)->:30, [45,60)->:00 (next hour)
    print("  Rounding timestamps to 30-minute buckets...")
    df['pickup_datetime'] = df['tpep_pickup_datetime'].dt.round('30min')
    
    # 2. Demand aggregation
    print("  Aggregating demand...")
    demand_df = df.groupby(['pickup_datetime', 'PULocationID']).size().reset_index(name='demand')
    
    # 3. Full grid reindexing (to capture zero-demand)
    print("  Generating complete Time/Location grid...")
    min_time = demand_df['pickup_datetime'].min()
    max_time = demand_df['pickup_datetime'].max()
    
    # Ensure we cover the full range of hours
    all_times = pd.date_range(start=min_time, end=max_time, freq='30min')
    all_locations = sorted(df['PULocationID'].unique())
    
    multi_index = pd.MultiIndex.from_product(
        [all_times, all_locations], 
        names=['pickup_datetime', 'PULocationID']
    )
    
    demand_df = (
        demand_df.set_index(['pickup_datetime', 'PULocationID'])
        .reindex(multi_index, fill_value=0)
        .reset_index()
    )
    
    # 4. Basic temporal features
    print("  Adding basic temporal features...")
    demand_df['hour'] = demand_df['pickup_datetime'].dt.hour
    # day_of_week: 1=Sun, 2=Mon, ..., 7=Sat
    demand_df['day_of_week'] = (demand_df['pickup_datetime'].dt.dayofweek + 1) % 7 + 1
    demand_df['month'] = demand_df['pickup_datetime'].dt.month
    
    return demand_df

def run_transformation_pipeline(input_path, output_path):
    """Load cleaned trips, transform to demand, and save."""
    print(f"Loading cleaned trips from {input_path}...")
    df = pd.read_parquet(input_path)
    
    demand_df = transform_to_demand(df)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    demand_df.to_parquet(output_path, index=False)
    print(f"Saved transformed demand data -> {output_path}")
    print(f"Final shape: {demand_df.shape}")
    
    return demand_df

if __name__ == "__main__":
    # For standalone testing
    CLEANED_PATH = "data/processed/yellow_tripdata_2024-2026_clean.parquet"
    TRANSFORMED_PATH = "data/processed/yellow_tripdata_2024-2026_transformed.parquet"
    if Path(CLEANED_PATH).exists():
        run_transformation_pipeline(CLEANED_PATH, TRANSFORMED_PATH)
    else:
        print(f"Input file not found: {CLEANED_PATH}")
