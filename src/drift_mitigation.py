"""
NYC TLC — Drift Mitigation
============================

One public entry point (mitigate) routes to one of two strategies based on
the string returned by drift_detection_evidently.select_mitigation_strategy().

Strategies
----------
  "drop_features"    Identify specific engineered features that drifted,
                     drop them, and retrain the model on OLD data.
                     Use when specific features are the root cause of drift.

Usage
-----
    model, scaler, dropped_features = mitigate(
        strategy         = selected_strategy,
        train_df         = train_raw,
        recent_df        = dec_train_raw,
        model_name       = "random_forest",
        model_dir        = "models/mitigated",
        drifted_features = drift_results["drifted_features"],
    )

    # For drop_features, dropped_features is the list of column names to exclude.
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


# ── Public entry point ────────────────────────────────────────────────────────

def mitigate(strategy, train_df, recent_df, model_name, model_dir,
             base_model=None, drifted_features=None):
    """
    Apply the chosen drift mitigation strategy.

    Args:
        strategy         : "none" | "drop_features"
        train_df         : original training data (raw, 2024-2025)
        recent_df        : recent data for monitoring (raw, 2026)
        model_name       : key into CANDIDATE_MODELS — used for the save filename
        model_dir        : directory to save the retrained model .pkl
        base_model       : fitted model whose hyperparameters to clone when
                           retraining (e.g. tuned_champion_model). 
        drifted_features : list of drifted engineered feature names from
                           parse_drift_results() — only used by "drop_features"

    Returns:
        (model, scaler, mappings, dropped_features)
        model            — new fitted model, or None when strategy is "none"
        scaler           — new fitted StandardScaler, or None when strategy is "none"
        mappings         — new demand score mappings, or None when strategy is "none"
        dropped_features — list of dropped feature names, None for strategy "none"
    """
    model_template = base_model 

    if strategy == "none":
        print("  No mitigation needed.")
        return None, None, None, None

    if strategy == "drop_features":
        model, scaler, mappings, dropped_features = _drop_and_retrain(
            train_df, model_name, model_dir, drifted_features or [], model_template
        )
        return model, scaler, mappings, dropped_features

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Private strategy implementations ─────────────────────────────────────────


def _drop_and_retrain(train_df, model_name, model_dir, drifted_features, model_template):
    """Drop specified drifted feature columns and retrain model on OLD data."""
    print(f"  Dropping drifted features: {drifted_features}")

    features, scaler, mappings = run_feature_pipeline(
        train_df, is_training=True
    )
    
    # Drop from features (ensure TARGET_COL is not dropped)
    cols_to_drop = [col for col in drifted_features if col in features.columns and col != TARGET_COL]
    features_dropped = features.drop(columns=cols_to_drop, errors="ignore")
    
    X = features_dropped.drop(columns=[TARGET_COL])
    y = features_dropped[TARGET_COL]

    model = clone(model_template)
    model.fit(X, y)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{model_name}_drop_features", model_dir)
    print(f"  Saved -> {model_dir}/{model_name}_drop_features.pkl")

    return model, scaler, mappings, cols_to_drop


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
