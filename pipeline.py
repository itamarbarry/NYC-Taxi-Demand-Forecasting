"""
NYC TLC Taxi Demand Forecasting — Full Pipeline
================================================

Run this script to execute the complete end-to-end pipeline adapted for demand:

  Step 1   — Data Cleaning (All raw files)
  Step 1.1 — Data Transformation (Aggregation to 30-min demand)
  Step 2   — Data Splitting (2024-2025 Train vs 2026 Test) + Subsampling
  Step 3   — Experiment A: Baseline (PULocationID, hour, day_of_week, month)
  Step 4   — Experiment B: Full Engineering (Geographic enrichment, cyclical, demand scores)
  Step 5   — Head-to-head comparison
  Step 6   — Champion + Feature Importance (Logged to W&B)
  Step 7   — Hyperparameter Tuning (Random Search -> Grid Search)
  Step 8   — Error Analysis
  Step 9   — Drift Detection (Monthly monitoring)
  Step 9.1 — Drift Detection (Evidently AI)
  Step 10  — Drift Mitigation
"""

import argparse
import numpy as np
np.float_ = np.float64
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Suppress tkinter GUI errors
import matplotlib.pyplot as plt
import os
from pathlib import Path


from src.cleaning                  import clean_parquet, clean_dataframe
from src.transform_to_demand       import run_transformation_pipeline
from src.splitting                 import split_train_test, subsample_splits, extract_drift_reference_set
from src.features                  import run_feature_pipeline, run_baseline_pipeline, TARGET_COL
from src.models                    import train_all_models, load_model, CANDIDATE_MODELS
from src.evaluation                import evaluate_all_models, select_champion, plot_feature_importance
from src.experiment_tracking       import ExperimentTracker, log_monthly_drift_run
from src.tuning                    import (RANDOM_SEARCH_CONFIG, GRID_SEARCH_CONFIGS,
                                           run_wandb_sweep, retrain_best_model)
from src.error_analysis            import run_error_analysis
from src.drift_detection           import (load_monthly_eval, run_monthly_drift_analysis,
                                           plot_monthly_mae_curve,
                                           plot_label_drift_distribution)
from src.drift_detection_evidently import (run_evidently_drift_report, parse_drift_results,
                                           run_evidently_concept_drift_report,
                                           parse_concept_drift_results,
                                           select_mitigation_strategy)
from src.drift_mitigation          import mitigate, plot_mitigation_comparison
from src.versioning                import (log_data_artifact, log_model_artifact, log_feature_artifact)
from src.transform_to_demand       import transform_to_demand, run_transformation_pipeline


# ── Path configuration ────────────────────────────────────────────────────────

RAW_DIR              = Path("data/raw")
PROCESSED_DIR        = Path("data/processed")
TRANSFORMED_PARQUET  = PROCESSED_DIR / "yellow_tripdata_2024-2026_transformed.parquet"

MODEL_DIR_BASELINE   = "models/baseline"
MODEL_DIR_ENGINEERED = "models/engineered"
MODEL_DIR_TUNED      = "models/tuned"
MODEL_DIR_MITIGATED  = "models/mitigated"
PLOTS_DIR            = "outputs/plots"

MONITORING_DIR       = Path("data/monitoring")
DRIFT_EVAL_PARQUET = MONITORING_DIR / "2026_01-02.parquet"
DRIFT_REFERENCE_PARQUET = MONITORING_DIR / "2024-2025_drift_reference.parquet"

# ── W&B configuration ─────────────────────────────────────────────────────────

WANDB_PROJECT = "Taxi_Demand_Forecasting"
WANDB_ENTITY  = None # Set to a team name if desired, or None to use the default team for your account


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def _comparison_table(baseline_results, engineered_results):
    merged = baseline_results[["model", "mae", "rmse"]].merge(
        engineered_results[["model", "mae", "rmse"]],
        on="model", suffixes=("_baseline", "_engineered"),
    )
    merged["mae_improvement_%"] = (
        (merged["mae_baseline"] - merged["mae_engineered"]) / merged["mae_baseline"] * 100
    ).round(1)

    print(f"\n  {'Model':<25} {'Baseline MAE':>13} {'Engineered MAE':>15} {'Improvement':>12}")
    print("  " + "-" * 68)
    for _, row in merged.iterrows():
        print(
            f"  {row['model']:<25} "
            f"{row['mae_baseline']:>13.2f} "
            f"{row['mae_engineered']:>15.2f} "
            f"{row['mae_improvement_%']:>11.1f}%"
        )
    return merged


def prepare_monitoring_summary(raw_dir, output_path):
    """
    Collect ENTIRE Jan and Feb 2026 data from raw and save to a summary parquet.
    """
    print(f"  Generating full monitoring summary for 2026 (Jan-Feb) ...")
    files = [
        raw_dir / "yellow_tripdata_2026-01.parquet",
        raw_dir / "yellow_tripdata_2026-02.parquet"
    ]
    summary_dfs = []
    for f in files:
        if f.exists():
            df = pd.read_parquet(f, columns=["tpep_pickup_datetime", "PULocationID", "tpep_dropoff_datetime"])
            # Filter strictly to Jan-Feb (no March spillover)
            df = clean_dataframe(df)
            df = df[df["tpep_pickup_datetime"] < "2026-03-01"]
            summary_dfs.append(df)
    
    if summary_dfs:
        full_summary = pd.concat(summary_dfs, ignore_index=True)
        
        # 1. Create temporary raw file for transformation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_raw = output_path.with_name("temp_monitoring_raw.parquet")
        full_summary.to_parquet(temp_raw, index=False)
        
        # 2. Transform to demand buckets
        run_transformation_pipeline(temp_raw, output_path)
        temp_raw.unlink()

        # 3. CRITICAL: Remove any March "spillover" caused by temporal rounding
        df_final = pd.read_parquet(output_path)
        df_final = df_final[df_final["month"] < 3].reset_index(drop=True)
        df_final.to_parquet(output_path, index=False)
        
        print(f"  Monitoring summary saved (Strict Jan-Feb) -> {output_path}")
    else:
        print("  Warning: No 2026 files found for monitoring summary.")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(wandb_project=WANDB_PROJECT, wandb_entity=WANDB_ENTITY):
    # ── W&B Login ─────────────────────────────────────────────────────────────
    key_path = Path("wandb_key.txt")
    if key_path.exists():
        with open(key_path, "r") as f:
            key = f.read().strip()
        if key:
            import os
            import wandb
            os.environ["WANDB_API_KEY"] = key
            wandb.login(key=key)
            print(f"  W&B: Logged in using API key from {key_path.name}")

    # ── W&B Data Preparation Tracker ──
    prep_tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = "data-preparation",
        tags     = ["preprocessing", "aggregation"],
    )

    # ── Step 1: Data Cleaning ──────────────────────────────────────────────────
    _print_header("STEP 1 — Data Cleaning")
    raw_files = list(RAW_DIR.glob("yellow_tripdata_*.parquet"))
    if not raw_files:
        print(f"  Error: No raw data found in {RAW_DIR}")
        return

    cleaned_dfs = []
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_COMBINED_PARQUET = PROCESSED_DIR / "yellow_tripdata_2024-2026_cleaned.parquet"
    
    for rf in raw_files:
        print(f"  Cleaning {rf.name} ...")
        df_raw = pd.read_parquet(rf, columns=["tpep_pickup_datetime", "PULocationID"])
        cleaned_dfs.append(clean_dataframe(df_raw))

    print(f"  Combining {len(cleaned_dfs)} cleaned datasets...")
    full_cleaned_df = pd.concat(cleaned_dfs, ignore_index=True)
    full_cleaned_df.to_parquet(CLEANED_COMBINED_PARQUET, index=False)
    print(f"  Saved single combined cleaned file -> {CLEANED_COMBINED_PARQUET}")

    # ── Step 1.1: Data Transformation ──────────────────────────────────────────
    _print_header("STEP 1.1 — Data Transformation (Aggregation to Demand)")
    run_transformation_pipeline(CLEANED_COMBINED_PARQUET, TRANSFORMED_PARQUET)
    
    # ── Step 2: Data Splitting + Subsampling ───────────────────────────────────
    _print_header("STEP 2 — Data Splitting + Subsampling")
    
    # 2.1 Initial Split: 2024-2025 vs 2026 Jan-Feb
    train_path, test_path = split_train_test(TRANSFORMED_PARQUET, PROCESSED_DIR, 
                                             cutoff_date='2026-01-01', end_date='2026-03-01')
    
    # 2.2 Extract Drift Reference Set (1 random week per month from 2024-2025)
    print("\n  Extracting drift reference set from 2024-2025 training data...")
    train_df_raw = pd.read_parquet(train_path)
    train_reduced_raw, drift_reference_raw = extract_drift_reference_set(train_df_raw)
    
    # Save drift reference set (already aggregated to demand)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    drift_reference_raw.to_parquet(DRIFT_REFERENCE_PARQUET, index=False)
    print(f"  Saved drift reference set -> {DRIFT_REFERENCE_PARQUET}")
    
    # Overwrite the train.parquet with the reduced version
    train_reduced_raw.to_parquet(train_path, index=False)
    print(f"  Updated training set (random weeks removed) -> {train_path}")

    print("\n  Subsampling splits for experiment runtime...")
    train_raw, test_raw = subsample_splits(train_path, test_path)

    log_data_artifact(
        prep_tracker, train_path, "train-split",
        metadata={"description": "Training set (2024-2025)"}
    )
    log_data_artifact(
        prep_tracker, test_path, "test-split",
        metadata={"description": "Test set (2026)"}
    )
    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A — Baseline (structural features only)
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 3 — Experiment A: Baseline (structural features only)")

    baseline_train, baseline_scaler = run_baseline_pipeline(train_raw, is_training=True)
    X_train_base = baseline_train.drop(columns=[TARGET_COL])
    y_train_base = baseline_train[TARGET_COL]
    print(f"  Baseline feature columns ({len(X_train_base.columns)}): "
          f"{X_train_base.columns.tolist()}")

    train_all_models(X_train_base, y_train_base, MODEL_DIR_BASELINE)

    baseline_test, _ = run_baseline_pipeline(
        test_raw, scaler=baseline_scaler, is_training=False
    )
    X_test_base = baseline_test.drop(columns=[TARGET_COL])
    y_test_base = baseline_test[TARGET_COL]

    print("\n  Baseline model results:")
    baseline_results = evaluate_all_models(X_test_base, y_test_base, MODEL_DIR_BASELINE)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B — Full Feature Engineering
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 4 — Experiment B: Full Feature Engineering")

    eng_train, eng_scaler, eng_mappings = run_feature_pipeline(train_raw, is_training=True)
    X_train_eng = eng_train.drop(columns=[TARGET_COL])
    y_train_eng = eng_train[TARGET_COL]
    print(f"  Engineered feature columns ({len(X_train_eng.columns)}): "
          f"{X_train_eng.columns.tolist()}")

    train_all_models(X_train_eng, y_train_eng, MODEL_DIR_ENGINEERED)

    eng_test, _ = run_feature_pipeline(test_raw, scaler=eng_scaler, mappings=eng_mappings, is_training=False)
    X_test_eng  = eng_test.drop(columns=[TARGET_COL])
    y_test_eng  = eng_test[TARGET_COL]

    print("\n  Engineered model results:")
    engineered_results = evaluate_all_models(X_test_eng, y_test_eng, MODEL_DIR_ENGINEERED)

    # Save the fitted scaler and mappings
    Path(MODEL_DIR_ENGINEERED).mkdir(parents=True, exist_ok=True)
    joblib.dump(eng_scaler, Path(MODEL_DIR_ENGINEERED) / "scaler.pkl")
    joblib.dump(eng_mappings, Path(MODEL_DIR_ENGINEERED) / "mappings.joblib")
    print(f"  Scaler and mappings saved -> {MODEL_DIR_ENGINEERED}/")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 5 — Feature Engineering Impact: Head-to-Head Comparison")
    print("  Metric: MAE (pickups).  Lower is better.\n")
    comparison = _comparison_table(baseline_results, engineered_results)

    # ── Step 6: Champion + Feature Importance + W&B ───
    _print_header("STEP 6 — Champion Model + Feature Importance")
    champion_name  = select_champion(engineered_results, metric="mae")
    champion_model = load_model(champion_name, MODEL_DIR_ENGINEERED)
    champion_row   = engineered_results.loc[
        engineered_results["model"] == champion_name
    ].iloc[0]

    plot_feature_importance(
        model         = champion_model,
        feature_names = X_test_eng.columns.tolist(),
        model_name    = champion_name,
        output_dir    = PLOTS_DIR,
    )

    tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = "champion-eval",
        tags     = ["champion", "engineered-features"],
        config   = {"champion_model": champion_name},
    )
    tracker.log_summary({
        "champion_model": champion_name,
        "mae":            float(champion_row["mae"]),
        "rmse":           float(champion_row["rmse"]),
        "mape":           float(champion_row["mape"]),
    })
    fi_path = Path(PLOTS_DIR) / f"feature_importance_{champion_name}.png"
    if fi_path.exists():
        tracker.log_image_file(fi_path, "feature_importance")
    print("\n  DEBUG: Reached Step 6 completion. Finishing W&B run...")
    tracker.log_code()
    url = tracker.finish()
    print(f"  W&B: Run successfully closed -> {url}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Hyperparameter Tuning
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 7 — Hyperparameter Tuning")

    # ── Phase 1: Random search ────────────────────────────────────────────────
    print("\n  Phase 1 — Random Search (all model families, wide grid)")
    random_sweep_id, best_random_config, best_random_mae = run_wandb_sweep(
        X_train     = X_train_eng,
        y_train     = y_train_eng,
        sweep_config = RANDOM_SEARCH_CONFIG,
        project     = wandb_project,
        entity      = wandb_entity,
        n_runs      = 15,
    )
    winning_family = best_random_config.get("model_type", "random_forest")
    print(f"\n  Random search complete. Best model family : {winning_family}")

    # ── Phase 2: Grid search on the winning family ────────────────────────────
    print(f"\n  Phase 2 — Grid Search ({winning_family}, narrow grid)")
    grid_config = GRID_SEARCH_CONFIGS[winning_family]
    _, best_grid_config, best_grid_mae = run_wandb_sweep(
        X_train      = X_train_eng,
        y_train      = y_train_eng,
        sweep_config = grid_config,
        project      = wandb_project,
        entity       = wandb_entity,
        n_runs       = 50,
    )

    # ── Retrain on full training set with best config ─────────────────────────
    tuned_champion_model = retrain_best_model(
        best_config = best_grid_config,
        X_train     = X_train_eng,
        y_train     = y_train_eng,
        model_dir   = MODEL_DIR_TUNED,
    )

    # ── Evaluate tuned model on FULL test set ─────────────────────────────
    print("\n  Loading full test set for final evaluation...")
    full_test_raw = pd.read_parquet(test_path)
    full_test_eng, _ = run_feature_pipeline(full_test_raw, scaler=eng_scaler, mappings=eng_mappings, is_training=False)
    X_full_test_eng = full_test_eng.drop(columns=[TARGET_COL])
    y_full_test_eng = full_test_eng[TARGET_COL]

    y_pred_tuned = tuned_champion_model.predict(X_full_test_eng)
    tuned_mae  = float(np.mean(np.abs(y_full_test_eng.values - y_pred_tuned)))
    tuned_rmse = float(np.sqrt(np.mean((y_full_test_eng.values - y_pred_tuned) ** 2)))

    print(f"\n  Tuned model evaluation:")
    print(f"    MAE  : {tuned_mae:.4f}  (champion baseline: {float(champion_row['mae']):.4f})")
    print(f"    RMSE : {tuned_rmse:.4f}  (champion baseline: {float(champion_row['rmse']):.4f})")

    tuning_tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = f"tuned-{winning_family}",
        tags     = ["tuned", "grid-search"],
        config   = best_grid_config,
    )
    tuning_tracker.log_summary({"mae": tuned_mae, "rmse": tuned_rmse})
    
    tuned_pkl = Path(MODEL_DIR_TUNED) / f"tuned_{winning_family}.pkl"
    log_model_artifact(
        tuning_tracker, tuned_pkl, "tuned-champion",
        metadata={"source": "grid-search", "mae": tuned_mae},
    )
    log_feature_artifact(
        tuning_tracker,
        Path(MODEL_DIR_ENGINEERED) / "scaler.pkl",
        active_feature_steps=X_train_eng.columns.tolist(),
        metadata={"n_features": len(X_train_eng.columns)},
    )

    # ── Final Performance Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_list = ['MAE', 'RMSE']
    values_list  = [tuned_mae, tuned_rmse]
    ax.bar(metrics_list, values_list, color=['#3498db', '#2ecc71'])
    ax.set_title(f'Final Tuned {winning_family} Performance (Test Set)', fontsize=14)
    ax.set_ylabel('Error Value (Pickups)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(values_list):
        ax.text(i, v + (max(values_list)*0.02), f"{v:.4f}", ha='center', fontweight='bold', fontsize=12)
    
    tuning_tracker.log_plot(fig, "final_test_performance_metrics")
    print("  Final performance metrics plot logged to W&B.")

    url = tuning_tracker.finish()
    print(f"  Tuned model W&B run -> {url}")

    # ── Step 8: Error Analysis ────────────────────────────────────────────────
    _print_header("STEP 8 — Error Analysis")
    error_df, error_figs = run_error_analysis(
        X_test     = X_full_test_eng,
        y_test     = y_full_test_eng,
        model      = tuned_champion_model,
        scaler     = eng_scaler,
        output_dir = PLOTS_DIR,
    )
    error_tracker = ExperimentTracker(
        project  = wandb_project,
        entity   = wandb_entity,
        run_name = "error-analysis",
        tags     = ["error-analysis", "tuned"],
        config   = {"model": winning_family, "n_test_samples": len(error_df)},
    )
    # W&B Table — select segments for breakdown
    error_tracker.log_table(
        error_df[["actual", "predicted", "abs_error", "pct_error",
                  "day_name", "hour", "PULocationID"]].head(1000).dropna(how="all"),
        table_name = "per_sample_errors",
    )
    error_tracker.log_summary({
        "mae": float(error_df["abs_error"].mean()),
        "p90_abs_error": float(error_df["abs_error"].quantile(0.9))
    })
    for col, fig in error_figs.items():
        error_tracker.log_plot(fig, f"error_by_{col}")
    url = error_tracker.finish()
    print(f"\n  W&B run logged -> {url}")

    # ── Step 9: Drift Detection (Macro Jan-Feb Monitoring) ─────────────────────────────
    _print_header("STEP 9 — Drift Detection (Macro Jan-Feb Monitoring)")
    
    # Generate monitoring summary for 2026 (Jan/Feb)
    prepare_monitoring_summary(RAW_DIR, DRIFT_EVAL_PARQUET)
    
    if not DRIFT_EVAL_PARQUET.exists():
        print(f"  Monthly eval parquet not found at {DRIFT_EVAL_PARQUET}. Skipping Step 9.")
    else:
        monthly_eval_df = load_monthly_eval(DRIFT_EVAL_PARQUET)
        # Use the 2024-2025 Drift Evaluation set as the REFERENCE
        reference_raw_df = pd.read_parquet(DRIFT_REFERENCE_PARQUET)
        
        print(f"  Monitoring performance: Ref (2024-2025 weeks) vs Current (Full Jan-Feb 2026)")

        monthly_summary, drift_reports = run_monthly_drift_analysis(
            monthly_eval_df  = monthly_eval_df,
            reference_raw_df = reference_raw_df,
            model            = tuned_champion_model,
            scaler           = eng_scaler,
            mappings         = eng_mappings,
            output_dir       = PLOTS_DIR,
        )
        curve_fig = plot_monthly_mae_curve(monthly_summary, output_dir=PLOTS_DIR)
        
        summary_tracker = ExperimentTracker(
            project  = wandb_project,
            entity   = wandb_entity,
            run_name = "drift-summary-2026",
            tags     = ["drift-monitoring", "summary"],
        )
        summary_tracker.log_plot(curve_fig, "2026_monthly_mae_curve")
        summary_tracker.finish()
        
        for _, row in monthly_summary.iterrows():
            month_label = row["month"]
            drift_report = drift_reports.get(month_label)
            if drift_report is None: continue
            
            log_monthly_drift_run(
                month_label  = month_label,
                month_num    = int(row["month_num"]),
                mae          = float(row["mae"]),
                drift_report = drift_report,
                project      = wandb_project,
                entity       = wandb_entity,
                mae_delta    = float(row["mae_delta"]),
                n_trips      = int(row["n_trips"]),
                label_drift  = {
                    "psi":        float(row["label_psi"]),
                    "ks_pvalue":  float(row["label_ks_pvalue"]),
                    "drifted":    bool(row["label_drifted"]),
                    "ref_mean":   float(row["label_ref_mean"]),
                    "cur_mean":   float(row["label_cur_mean"]),
                },
            )

    # ── Step 9.1: Evidently AI (Micro Deep Dive) ──────────────────────────────
    _print_header("STEP 9.1 — Drift Detection (Evidently AI Deep Dive)")
    
    # LOAD FULL AGGREGATED 2026 DATA (to avoid sampling bias in demand)
    current_2026_demand = pd.read_parquet(DRIFT_EVAL_PARQUET)

    # Reference is the "drift_evaluation" set (the random weeks from 2024-2025)
    print(f"  Loading in-distribution reference from {DRIFT_REFERENCE_PARQUET}...")
    ref_demand_df = pd.read_parquet(DRIFT_REFERENCE_PARQUET)

    # Engineer features
    ref_eng, _ = run_feature_pipeline(ref_demand_df, scaler=eng_scaler, mappings=eng_mappings, is_training=False)
    current_2026_eng, _ = run_feature_pipeline(current_2026_demand, scaler=eng_scaler, mappings=eng_mappings, is_training=False)

    # Dataset Drift Report — Exclude time-based features
    print("\n  Running Evidently dataset + label drift report (excluding time features) ...")
    
    exclude_cols = ["month", "month_sin", "month_cos", "hour", "hour_sin", "hour_cos", "day_of_week"]
    ref_eng_filtered   = ref_eng.drop(columns=[c for c in exclude_cols if c in ref_eng.columns])
    current_2026_eng_filtered = current_2026_eng.drop(columns=[c for c in exclude_cols if c in current_2026_eng.columns])
    
    evidently_report = run_evidently_drift_report(ref_eng_filtered, current_2026_eng_filtered)
    drift_results    = parse_drift_results(evidently_report)
    
    print(f"\n  Overall drift detected : {drift_results['overall_drift']}")
    print(f"  Features drifted       : {drift_results['n_drifted']} ({drift_results['share_drifted']:.1%} of feature columns)")
    if drift_results["drifted_features"]:
        print(f"  Drifted feature names  : {drift_results['drifted_features']}")
    print(f"  Target (label) drift   : {drift_results['target_drift']} (score={drift_results['target_drift_score']:.4f})")

    outputs_path = Path("outputs")
    outputs_path.mkdir(exist_ok=True)
    evidently_html = outputs_path / "evidently_drift_report.html"
    evidently_report.save_html(str(evidently_html))
    print(f"\n  Evidently dataset drift HTML -> {evidently_html}")

    # Concept Drift Report
    print("\n  Running Evidently concept drift report ...")
    # REFERENCE: The 24 weeks from 2024-2025
    ref_perf_df = ref_eng.copy()
    X_ref_eval = ref_eng.drop(columns=[TARGET_COL])
    ref_perf_df["prediction"] = tuned_champion_model.predict(X_ref_eval)
    
    # CURRENT: Jan-Feb 2026 evaluation slice
    cur_perf_df = current_2026_eng.copy()
    X_cur_eval = current_2026_eng.drop(columns=[TARGET_COL])
    cur_perf_df["prediction"] = tuned_champion_model.predict(X_cur_eval)
    
    concept_report = run_evidently_concept_drift_report(ref_perf_df, cur_perf_df)
    concept_results = parse_concept_drift_results(concept_report)
    
    print(f"\n  Concept drift detected : {concept_results['concept_drift_detected']}")
    print(f"  Reference MAE (Jan)    : {concept_results['ref_mae']:.4f} pickups")
    concept_drift_html = outputs_path / "evidently_concept_drift_report.html"
    concept_report.save_html(str(concept_drift_html))
    print(f"\n  Evidently concept drift HTML -> {concept_drift_html}")

    # SAFETY FILTER: Protect time-based features from being dropped.
    # Even if Evidently flags them, we know it's a calendar artifact.
    PROTECTED = ["hour", "month", "day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos"]
    original_drifted = drift_results["drifted_features"]
    drift_results["drifted_features"] = [f for f in original_drifted if f not in PROTECTED]
    
    if len(original_drifted) != len(drift_results["drifted_features"]):
        shielded = [f for f in original_drifted if f in PROTECTED]
        print(f"  Shielded time features from mitigation: {shielded}")
        # Update counts for the selection logic
        drift_results["n_drifted"] = len(drift_results["drifted_features"])
        drift_results["share_drifted"] = drift_results["n_drifted"] / (len(ref_eng.columns) - 1)

    selected_strategy = select_mitigation_strategy(drift_results, concept_results)
    print(f"\n  Selected strategy      : {selected_strategy}")

    # ── Step 10: Drift Mitigation + Before / After Comparison ─────────────────
    _print_header("STEP 10 — Drift Mitigation + Before / After Comparison")
    
    # 1. Ensure the 2026 dataset is saved for W&B versioning
    # (Already saved as DRIFT_EVAL_PARQUET in Step 9)
    drift_2026_parquet = DRIFT_EVAL_PARQUET

    # Baseline: tuned champion evaluated on 2026 evaluation set
    y_drift_eval = current_2026_eng[TARGET_COL].values
    y_pred_drift_base = tuned_champion_model.predict(current_2026_eng.drop(columns=[TARGET_COL]))
    baseline_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_base)))
    print(f"\n  Tuned champion MAE — 2026 eval (pre-mitigation) : {baseline_drift_mae:.4f} pickups")

    if selected_strategy == "none":
        print("  No mitigation required — skipping Step 10.")
    else:
        print(f"\n  Applying strategy: {selected_strategy}")
        mitigated_model, mitigated_scaler, mitigated_mappings, eval_steps = mitigate(
            strategy         = selected_strategy, 
            train_df         = train_raw,
            recent_df        = current_2026_demand,
            model_name       = winning_family,
            model_dir        = MODEL_DIR_MITIGATED,
            base_model       = tuned_champion_model,
            drifted_features = drift_results["drifted_features"]
        )
        
        if mitigated_model is None:
            mitigated_model = tuned_champion_model

        active_scaler   = mitigated_scaler if mitigated_scaler is not None else eng_scaler
        active_mappings = mitigated_mappings if mitigated_mappings is not None else eng_mappings
        
        current_2026_eng_mit, _ = run_feature_pipeline(
            current_2026_demand, scaler=active_scaler, mappings=active_mappings,
            is_training=False, custom_creation_steps=eval_steps,
        )
        
        y_pred_drift_mit = mitigated_model.predict(current_2026_eng_mit.drop(columns=[TARGET_COL]))
        mitigated_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_mit)))
        improvement_pct = (baseline_drift_mae - mitigated_drift_mae) / baseline_drift_mae * 100
        
        print(f"  Mitigated model MAE — Drift eval (post-mitigation) : {mitigated_drift_mae:.4f} pickups")
        print(f"  Improvement                                       : {improvement_pct:+.1f}%")

        # Build the comparison plot
        comparison_fig = plot_mitigation_comparison(
            {
                "24-week baseline (2024-25)":         np.abs(ref_eng[TARGET_COL].values - tuned_champion_model.predict(ref_eng.drop(columns=[TARGET_COL]))),
                "2026 Jan & Feb (Drifted)":           np.abs(y_drift_eval - y_pred_drift_base),
                f"Mitigated ({selected_strategy}) — 2026": np.abs(y_drift_eval - y_pred_drift_mit),
            },
            output_dir = PLOTS_DIR,
        )

        # Log to W&B
        mitigation_tracker = ExperimentTracker(
            project  = wandb_project,
            entity   = wandb_entity,
            run_name = f"mitigation-{selected_strategy}-2026",
            tags     = ["drift-mitigation", "2026", selected_strategy],
            config   = {
                "strategy":          selected_strategy,
                "drifted_features":  drift_results["drifted_features"],
                "n_current_2026":    len(current_2026_demand),
                "drift_seed":        42,
            },
        )
        mitigation_tracker.log_summary({
            "reference_mae":       float(concept_results["ref_mae"]),
            "baseline_drift_mae":   float(baseline_drift_mae),
            "mitigated_drift_mae":  float(mitigated_drift_mae),
            "mae_improvement_pct": float(improvement_pct),
        })
        mitigation_tracker.log_plot(comparison_fig, "mitigation_comparison")

        # Version the 2026 evaluation dataset artifact
        log_data_artifact(
            mitigation_tracker, DRIFT_EVAL_PARQUET, "2026-jan-feb-full",
            metadata={"period": "Jan-Feb 2026", "n_rows": len(current_2026_demand)},
        )

        # Save the mitigated scaler
        mitigated_scaler_path = Path(MODEL_DIR_MITIGATED) / "scaler_mitigated.pkl"
        joblib.dump(active_scaler, mitigated_scaler_path)

        # Version the mitigated feature pipeline artifact
        mitigated_feature_cols = [c for c in current_2026_eng_mit.columns if c != TARGET_COL]
        log_feature_artifact(
            mitigation_tracker,
            mitigated_scaler_path,
            active_feature_steps=mitigated_feature_cols,
            metadata={"strategy": selected_strategy, "n_features": len(mitigated_feature_cols)},
        )

        # Version the mitigated model artifact
        mitigated_pkl_name = {
            "drop_features":    f"{winning_family}_drop_features.pkl",
        }.get(selected_strategy)

        if mitigated_pkl_name:
            log_model_artifact(
                mitigation_tracker,
                Path(MODEL_DIR_MITIGATED) / mitigated_pkl_name,
                "mitigated-model",
                metadata={
                    "strategy":          selected_strategy,
                    "mae":               mitigated_drift_mae,
                    "improvement_pct":   improvement_pct,
                    "drifted_features":  drift_results["drifted_features"],
                },
            )

        # Attach the Evidently HTML report
        mitigation_tracker.log_artifact(
            evidently_html, artifact_name="evidently-drift-report", artifact_type="report",
        )

        url = mitigation_tracker.finish()
        print(f"\n  W&B run logged -> {url}")

    print("\n  Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help="W&B project name to log runs into")
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY,
                        help="W&B entity (team) name")
    args = parser.parse_args()
    
    # Use command-line args if provided, otherwise fallback to script constants
    # This allows GitHub Actions to pass empty strings and still use these defaults
    project = args.wandb_project if args.wandb_project else WANDB_PROJECT
    entity  = args.wandb_entity if args.wandb_entity else WANDB_ENTITY
    
    run_pipeline(wandb_project=project, wandb_entity=entity)
