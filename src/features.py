"""
NYC TLC Yellow Taxi -- Feature Engineering for Demand Forecasting
================================================================

Adapts the original feature pipeline for a demand prediction task.
Basic structural features (hour, day_of_week, month) are assumed to 
be present in the input DataFrame (from transform_to_demand.py).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_COL = "demand"

# Numerical features to scale
SCALE_FEATURES = ["PULocationID", "hour", "day_of_week", "month"]


# Categorical encoding mappings
BOROUGH_MAP = {
    'Manhattan': 1, 'Brooklyn': 2, 'Queens': 3, 
    'Bronx': 4, 'Staten Island': 5, 'EWR': 6, 'Unknown': 0
}

SERVICE_ZONE_MAP = {
    'Yellow Zone': 1, 'Boro Zone': 2, 'Airports': 3, 'EWR': 4, 'Unknown': 0
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE CREATION
# ═════════════════════════════════════════════════════════════════════════════

def _add_geographic_enrichment(df):
    """Join with taxi_zone_lookup to add Borough and service_zone."""
    lookup_path = Path("data/raw/taxi_zone_lookup.csv")
    if not lookup_path.exists():
        print(f"  Warning: {lookup_path} not found. Skipping enrichment.")
        return df
    
    lookup = pd.read_csv(lookup_path)
    # Merge on LocationID
    df = df.merge(
        lookup[['LocationID', 'Borough', 'service_zone']], 
        left_on='PULocationID', 
        right_on='LocationID', 
        how='left'
    ).drop(columns=['LocationID'])
    
    # Clean up service_zone as per EDA
    df['service_zone'] = df['service_zone'].replace('EWR', 'Airports')
    df['Borough'] = df['Borough'].fillna('Unknown')
    df['service_zone'] = df['service_zone'].fillna('Unknown')
    return df


def _add_holiday_flag(df):
    """
    Mark major US holidays (fixed-date) to help the model identify 
    anomalous demand days.
    """
    if 'pickup_datetime' in df.columns:
        dates = df['pickup_datetime']
    else:
        return df

    month_day = list(zip(dates.dt.month, dates.dt.day))
    
    holidays = {
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
        (12, 31), # New Year's Eve
    }
    
    df['is_holiday'] = [1 if md in holidays else 0 for md in month_day]
    return df




def _add_nightlife_features(df):
    """Flag major nightlife hotspots during peak surge hours."""
    # General Nightlife Hotspots (excluding East Village)
    HOTSPOT_IDS = [142, 158, 249]
    is_hotspot = df['PULocationID'].isin(HOTSPOT_IDS)
    
    # Window: 22:00 - 04:00
    is_prime_hours = (df['hour'] >= 22) | (df['hour'] < 4)
    
    # Weekend nights: Fri/Sat/Sun nights
    is_weekend_night = df['day_of_week'].isin([1, 2, 7])
    
    # New Year's Day
    is_new_years = (df['month'] == 1) & (df['pickup_datetime'].dt.day == 1) if 'pickup_datetime' in df.columns else False
    
    is_surge_time = is_prime_hours & (is_weekend_night | is_new_years)
    
    # 1. General Hotspot Surge (Now excludes East Village)
    df['is_nightlife_surge'] = (is_hotspot & is_surge_time).astype(int)
    
    # 2. EXCLUSIVE East Village Surge
    is_ev = (df['PULocationID'] == 79)
    df['is_east_village_surge'] = (is_ev & is_surge_time).astype(int)
    
    return df


# Historical profile function removed


def _encode_categorical_features(df):
    """Encode high-cardinality categoricals using stable mappings."""
    df['Borough'] = df['Borough'].map(BOROUGH_MAP).fillna(0).astype(int)
    df['service_zone'] = df['service_zone'].map(SERVICE_ZONE_MAP).fillna(0).astype(int)
    
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE TRANSFORMATION
# ═════════════════════════════════════════════════════════════════════════════

def _add_cyclical_time_features(df):
    """
    Add sin/cos transformations for hour and month.
    We use high-resolution time (hour + minute) to ensure 30-minute buckets 
    are positioned correctly in the cycle.
    """
    if 'pickup_datetime' in df.columns:
        # 8:30 AM becomes 8.5
        fractional_hour = df['pickup_datetime'].dt.hour + df['pickup_datetime'].dt.minute / 60.0
    else:
        # Fallback to integer hour if timestamp is missing
        fractional_hour = df['hour']

    df['hour_sin'] = np.sin(2 * np.pi * fractional_hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * fractional_hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PIPELINE ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_CREATION_STEPS = [
    _add_geographic_enrichment,
    _encode_categorical_features,
    _add_holiday_flag,
    _add_nightlife_features,
]

FEATURE_TRANSFORMATION_STEPS = [
    _add_cyclical_time_features,
]


# Raw columns kept in the baseline (no engineering applied)
BASELINE_FEATURE_COLS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "month",
]


def run_baseline_pipeline(df, scaler=None, is_training=True):
    """Minimal pipeline: basic structural features only."""
    if not is_training and scaler is None:
        raise ValueError("scaler must be provided when is_training=False.")

    df = df.copy()
    
    # Ensure numerical consistency
    for col in BASELINE_FEATURE_COLS:
        df[col] = df[col].astype(float)

    if is_training:
        scaler = StandardScaler()
        df[BASELINE_FEATURE_COLS] = scaler.fit_transform(df[BASELINE_FEATURE_COLS])
    else:
        df[BASELINE_FEATURE_COLS] = scaler.transform(df[BASELINE_FEATURE_COLS])

    keep = BASELINE_FEATURE_COLS + [TARGET_COL]
    return df[keep], scaler


def run_feature_pipeline(df, scaler=None, mappings=None, is_training=True,
                         custom_creation_steps=None):
    """Execute the complete feature engineering pipeline for demand."""
    if not is_training and (scaler is None or mappings is None):
        raise ValueError("scaler and mappings must be provided when is_training=False.")

    df = df.copy()

    # 1. Feature creation (Static)
    creation_steps = custom_creation_steps if custom_creation_steps is not None \
                     else FEATURE_CREATION_STEPS
    for step in creation_steps:
        df = step(df)

    # 2. Historical Profiles - REMOVED
    # df, mappings = _add_historical_profiles(df, mappings)
    mappings = {}

    # 3. Feature transformation (Dynamic/Temporal)
    for step in FEATURE_TRANSFORMATION_STEPS:
        df = step(df)

    # 4. Feature scaling (on numerical columns)
    # We include our new 'scores' in the scaling set
    numeric_to_scale = SCALE_FEATURES + [
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]
    
    if is_training:
        scaler = StandardScaler()
        df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
    else:
        df[numeric_to_scale] = scaler.transform(df[numeric_to_scale])

    # 5. Drop columns that shouldn't be in the feature matrix
    if "pickup_datetime" in df.columns:
        df = df.drop(columns=["pickup_datetime"])

    if is_training:
        return df, scaler, mappings
    else:
        return df, None
