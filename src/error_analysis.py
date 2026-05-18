"""
NYC TLC — Per-Sample Error Analysis for Demand Forecasting
=========================================================

Breaks down model errors by meaningful data segments (Borough, Zone, Time).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from src.features import BOROUGH_MAP, SERVICE_ZONE_MAP


DAY_NAMES = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
BOROUGH_NAMES = {v: k for k, v in BOROUGH_MAP.items()}
SERVICE_ZONE_NAMES = {v: k for k, v in SERVICE_ZONE_MAP.items()}


# ── Error DataFrame ───────────────────────────────────────────────────────────

def build_error_df(y_test, y_pred, feature_df=None, scaler=None):
    """
    Build a per-sample error DataFrame.
    If a scaler is provided, features are inverse-transformed for readable grouping.
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error_df = pd.DataFrame({
        "actual":    y_test,
        "predicted": y_pred,
    })
    error_df["abs_error"] = np.abs(error_df["actual"] - error_df["predicted"])

    # Percentage error — guard against near-zero denominators (demand can be 0)
    mask = error_df["actual"] >= 1.0
    error_df["pct_error"] = np.nan
    error_df.loc[mask, "pct_error"] = (
        error_df.loc[mask, "abs_error"] / error_df.loc[mask, "actual"] * 100
    )

    if feature_df is None:
        return error_df

    feat = feature_df.copy().reset_index(drop=True)

    # ── Inverse scaling for readable labels ───────────────────────────────────
    if scaler is not None:
        try:
            # We only inverse transform columns that the scaler knows about
            scaled_cols = [c for c in feat.columns if c in scaler.get_feature_names_out()]
            if scaled_cols:
                feat[scaled_cols] = scaler.inverse_transform(feat[scaled_cols])
                # Round back to integers for categories
                for col in ["hour", "day_of_week", "month", "PULocationID", "Borough", "service_zone"]:
                    if col in feat.columns:
                        feat[col] = feat[col].round().astype(int)
        except Exception as exc:
            print(f"  Warning: Inverse scaling failed: {exc}")

    # ── Borough ───────────────────────────────────────────────────────────────
    if "Borough" in feat.columns:
        error_df["Borough"] = feat["Borough"].map(BOROUGH_NAMES).fillna("Unknown")

    # ── Service Zone ──────────────────────────────────────────────────────────
    if "service_zone" in feat.columns:
        error_df["service_zone"] = feat["service_zone"].map(SERVICE_ZONE_NAMES).fillna("Unknown")

    # ── Day of week ───────────────────────────────────────────────────────────
    if "day_of_week" in feat.columns:
        error_df["day_name"] = feat["day_of_week"].map(DAY_NAMES).fillna("Unknown")

    # ── Hour ──────────────────────────────────────────────────────────────────
    if "hour" in feat.columns:
        error_df["hour"] = feat["hour"]

    # ── Location ID ───────────────────────────────────────────────────────────
    if "PULocationID" in feat.columns:
        error_df["PULocationID"] = feat["PULocationID"]

    return error_df


# ── Segment plot ──────────────────────────────────────────────────────────────

def plot_error_by_segment(error_df, group_col, title=None, output_dir=None):
    """
    Horizontal bar chart: mean absolute error per value of `group_col`.
    """
    if group_col not in error_df.columns:
        print(f"  Skipping segment plot — '{group_col}' not in error_df.")
        return None

    grouped = (
        error_df.groupby(group_col, observed=True)["abs_error"]
        .agg(mean_mae="mean", n_samples="count")
        .sort_values("mean_mae", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(3, len(grouped) * 0.65)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grouped)))
    bars   = ax.barh(grouped.index.astype(str), grouped["mean_mae"], color=colors)

    # Value labels on bars
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=9)

    # Sample count inside each bar
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(
            grouped["mean_mae"].min() * 0.05, i,
            f"n={row['n_samples']:,}",
            va="center", ha="left", fontsize=8,
            color="white", fontweight="bold",
        )

    ax.set_xlabel("Mean Absolute Error (pickups)")
    ax.set_title(title or f"MAE by {group_col}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_by_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {path}")

    return fig


# ── Layered histogram ────────────────────────────────────────────────────────

def plot_error_histogram_by_segment(error_df, group_col, title=None, output_dir=None, bins=40):
    """
    Layered histogram of abs_error.
    """
    if group_col not in error_df.columns:
        print(f"  Skipping histogram — '{group_col}' not in error_df.")
        return None

    groups = error_df[group_col].dropna().unique()

    # Clip to visible range
    all_vals = error_df["abs_error"].dropna().clip(upper=20)
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10.colors

    for i, grp in enumerate(sorted(groups, key=str)):
        vals = error_df.loc[error_df[group_col] == grp, "abs_error"].dropna().clip(upper=20)
        ax.hist(
            vals,
            bins    = bin_edges,
            alpha   = 0.45,
            label   = f"{grp} (n={len(vals):,})",
            color   = colors[i % len(colors)],
            edgecolor = "none",
            density = True,
        )

    ax.set_xlim(0, 20)
    ax.set_xlabel("Absolute Error (pickups)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Error Distribution by {group_col}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"error_hist_{group_col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {path}")

    return fig


# ── Orchestrator ──────────────────────────────────────────────────────────────

_SEGMENTS = [
    ("Borough",         "Borough"),
    ("service_zone",    "Service Zone"),
    ("day_name",        "Day of Week"),
    ("hour",            "Hour of Day"),
]


def run_error_analysis(X_test, y_test, model, scaler=None, output_dir=None):
    """
    Full error analysis pipeline for demand.
    """
    y_pred   = model.predict(X_test)
    error_df = build_error_df(y_test, y_pred, feature_df=X_test, scaler=scaler)

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  {'Metric':<22} {'Value':>10}")
    print("  " + "-" * 35)
    print(f"  {'Mean Abs Error':<22} {error_df['abs_error'].mean():>9.2f} pickups")
    print(f"  {'Median Abs Error':<22} {error_df['abs_error'].median():>9.2f} pickups")
    print(f"  {'90th pct Error':<22} {error_df['abs_error'].quantile(0.9):>9.2f} pickups")
    print(f"  {'Max Error':<22} {error_df['abs_error'].max():>9.2f} pickups")
    valid_pct = error_df["pct_error"].dropna()
    if len(valid_pct):
        print(f"  {'Mean Pct Error':<22} {valid_pct.mean():>9.1f} %")

    # ── Per-segment breakdown ─────────────────────────────────────────────────
    figs = {}
    for col, title in _SEGMENTS:
        if col not in error_df.columns:
            continue
        if error_df[col].isna().all():
            continue

        print(f"\n  Error by {title}:")
        grouped = (
            error_df.groupby(col, observed=True)["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )
        for group, mae in grouped.items():
            print(f"    {str(group):<25}  MAE = {mae:.2f} pickups")

        fig = plot_error_by_segment(
            error_df, col, title=f"MAE by {title}", output_dir=output_dir
        )
        if fig is not None:
            figs[col] = fig

        hist_fig = plot_error_histogram_by_segment(
            error_df, col, title=f"Error Distribution by {title}", output_dir=output_dir
        )
        if hist_fig is not None:
            figs[f"{col}_hist"] = hist_fig

    return error_df, figs
