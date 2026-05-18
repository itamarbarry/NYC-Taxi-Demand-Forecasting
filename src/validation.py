"""
NYC TLC Yellow Taxi -- Demand Forecasting Validation
===================================================

This module provides lean validation for the columns required for demand forecasting:
- tpep_pickup_datetime
- PULocationID
"""

import pandas as pd

def _check_not_null(df, col):
    n_null = df[col].isna().sum()
    passed = int(n_null) == 0
    return {
        "name":   "not_null",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else f"{n_null:,} null values",
    }

def _check_between(df, col, min_value, max_value):
    mask = (df[col] >= min_value) & (df[col] <= max_value)
    fail_rate = (~mask).mean()
    passed = fail_rate == 0
    return {
        "name":   f"between[{min_value}, {max_value}]",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else f"{fail_rate:.2%} outside range",
    }

def _check_min_year(df, col, min_year):
    years = df[col].dt.year.unique()
    invalid = [y for y in years if y < min_year]
    passed = len(invalid) == 0
    return {
        "name":   f"year_at_least({min_year})",
        "column": col,
        "passed": passed,
        "detail": "OK" if passed else f"invalid years found: {invalid}",
    }

def validate_nyc_taxi_parquet(parquet_path):
    """Run the 3 core validation checks matching the notebook."""
    REQUIRED_COLS = ["tpep_pickup_datetime", "PULocationID"]
    df = pd.read_parquet(parquet_path, columns=REQUIRED_COLS)
    
    results = []
    # 1. Null check (on pickup time and location)
    n_null = df[REQUIRED_COLS].isna().any(axis=1).sum()
    passed_null = int(n_null) == 0
    results.append({
        "name":   "not_null",
        "column": "multiple",
        "passed": passed_null,
        "detail": "OK" if passed_null else f"{n_null:,} rows with nulls",
    })
    
    # 2. Location ID range check (NYC zones 1-263)
    results.append(_check_between(df, "PULocationID", 1, 263))
    
    # 3. Year check (Strictly 2024 and above)
    results.append(_check_min_year(df, "tpep_pickup_datetime", min_year=2024))

    success = all(r["passed"] for r in results)
    return {"success": success, "results": results}
