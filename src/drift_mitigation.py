"""
NYC TLC — Drift Mitigation
============================

One public entry point (mitigate) routes to one of three strategies based on
the string returned by drift_detection_evidently.select_mitigation_strategy().

Strategies
----------
  "recalibrate"      Refit the StandardScaler on recent data. Model unchanged.
                     Use when drift is mild and scale-level only.

  "drop_features"    Identify feature creation steps that produce drifted
                     engineered features, remove them, retrain on OLD data.
                     Use when specific features are the root cause of drift.

Usage
-----
    model, scaler, eval_steps = mitigate(
        strategy         = selected_strategy,
        train_df         = train_raw,
        recent_df        = dec_train_raw,
        model_name       = "random_forest",
        model_dir        = "models/mitigated",
        drifted_features = drift_results["drifted_features"],
    )

    # eval_steps is None for recalibrate.
    # For drop_features it is the filtered FEATURE_CREATION_STEPS list —
    # pass it to run_feature_pipeline when engineering the evaluation set
    # so its feature columns match what the mitigated model was trained on.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from src.features import (run_feature_pipeline, TARGET_COL,
                           SCALE_FEATURES, FEATURE_CREATION_STEPS)
from src.models   import save_model, CANDIDATE_MODELS


# ── Configuration ─────────────────────────────────────────────────────────────

RECENCY_WEIGHT = 3.0  # sample weight multiplier for recent rows in reweight_retrain

# Engineered feature name -> the feature creation step function that produces it.
# Used by drop_features to identify which pipeline steps to remove when a
# particular engineered feature is flagged as drifted by Evidently.
FEATURE_SOURCE_MAP = {
    "hour_sin":              "_add_cyclical_time_features",
    "hour_cos":              "_add_cyclical_time_features",
    "month_sin":             "_add_cyclical_time_features",
    "month_cos":             "_add_cyclical_time_features",
    "is_holiday":            "_add_holiday_flag",
    "is_nightlife_surge":    "_add_nightlife_features",
    "is_east_village_surge": "_add_nightlife_features",
    "Borough":               "_encode_categorical_features",
    "service_zone":          "_encode_categorical_features",
}


# ── Public entry point ────────────────────────────────────────────────────────

def mitigate(strategy, train_df, recent_df, model_name, model_dir,
             base_model=None, drifted_features=None):
    """
    Apply the chosen drift mitigation strategy.

    Args:
        strategy         : "none" | "recalibrate" | "drop_features"
        train_df         : original training data (raw, 2024-2025)
        recent_df        : recent data for monitoring (raw, 2026)
        model_name       : key into CANDIDATE_MODELS — used for the save filename
        model_dir        : directory to save the retrained model .pkl
        base_model       : fitted model whose hyperparameters to clone when
                           retraining (e.g. tuned_champion_model). 
        drifted_features : list of drifted engineered feature names from
                           parse_drift_results() — only used by "drop_features"

    Returns:
        (model, scaler, mappings, eval_steps)
        model      — new fitted model, or None when strategy is "none"/"recalibrate"
        scaler     — new fitted StandardScaler, or None when strategy is "none"
        mappings   — new demand score mappings, or None when strategy is "none"
        eval_steps — filtered FEATURE_CREATION_STEPS list for "drop_features",
                     None for all other strategies
    """
    model_template = base_model 

    if strategy == "none":
        print("  No mitigation needed.")
        return None, None, None, None

    if strategy == "recalibrate":
        scaler = _recalibrate(recent_df)
        return None, scaler, None, None

    if strategy == "drop_features":
        model, scaler, mappings, eval_steps = _drop_and_retrain(
            train_df, model_name, model_dir, drifted_features or [], model_template
        )
        return model, scaler, mappings, eval_steps

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Private strategy implementations ─────────────────────────────────────────

def _recalibrate(recent_df: pd.DataFrame) -> StandardScaler:
    """Refit the StandardScaler on recent data. Model weights are unchanged."""
    # We must run the full feature pipeline to generate engineered features 
    # (sin/cos etc) before fitting the new scaler, otherwise it won't 
    # recognize all the columns it needs to transform.
    _, scaler, _ = run_feature_pipeline(recent_df, is_training=True)
    
    print(f"  Scaler recalibrated on {len(recent_df):,} recent rows")
    return scaler


def _drop_and_retrain(train_df, model_name, model_dir, drifted_features, model_template):
    """Remove feature steps that produce drifted features, retrain on OLD data only."""
    steps_to_drop = {
        FEATURE_SOURCE_MAP[feat]
        for feat in drifted_features
        if feat in FEATURE_SOURCE_MAP
    }
    filtered_steps = [s for s in FEATURE_CREATION_STEPS
                      if s.__name__ not in steps_to_drop]
    removed_names  = [s.__name__ for s in FEATURE_CREATION_STEPS
                      if s.__name__ in steps_to_drop]
    print(f"  Dropping feature steps: {removed_names}")

    features, scaler, mappings = run_feature_pipeline(
        train_df, is_training=True, custom_creation_steps=filtered_steps
    )
    X = features.drop(columns=[TARGET_COL])
    y = features[TARGET_COL]

    model = clone(model_template)
    model.fit(X, y)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{model_name}_drop_features", model_dir)
    print(f"  Saved -> {model_dir}/{model_name}_drop_features.pkl")

    return model, scaler, mappings, filtered_steps


# ── Comparison Plots ─────────────────────────────────────────────────────────

def plot_mitigation_comparison(abs_errors_dict: dict, output_dir=None):
    """
    Boxplot comparison of absolute errors across different model versions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = list(abs_errors_dict.keys())
    data   = [errors for errors in abs_errors_dict.values()]
    
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="steelblue"),
               medianprops=dict(color="tomato", linewidth=2))
    
    ax.set_ylabel("Absolute Error (pickups)")
    ax.set_title("Drift Mitigation Comparison: Absolute Error Distributions")
    ax.spines[["top", "right"]].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "mitigation_comparison_boxplot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved -> {path}")
    
    return fig
