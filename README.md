# NYC Taxi Demand Forecasting & Drift Monitoring

This project was completed as part of the **"Data Science Methods and Applications"** B.Sc. course in the **Industrial Engineering & Management Department** at **Ben-Gurion University**.

It presents an end-to-end machine learning pipeline for forecasting NYC yellow taxi demand. It features robust data validation, feature engineering, automated model tuning, and an advanced drift detection and mitigation system to handle non-stationary data distributions.

## 🚖 Project Overview

The objective is to predict the number of taxi pickups (demand) across different NYC zones in 30-minute intervals. The project is designed to be "industrial-grade," incorporating experiment tracking with **Weights & Biases (W&B)** and drift monitoring with **Evidently AI**.

### Key Features
- **Temporal Aggregation**: Converts raw trip records into time-series demand buckets.
- **Geographic Enrichment**: Incorporates borough and service zone metadata.
- **Automated ML Lifecycle**: From raw data cleaning to hyperparameter optimization.
- **Drift Management**: Detects both feature drift (covariate shift) and concept drift, triggering automated mitigation strategies.

---

## 🛠️ Requirements & Setup

### Technical Stack
- **Language**: Python 3.11+
- **Core Libraries**: `pandas`, `scikit-learn`, `xgboost`, `numpy`
- **Experiment Tracking**: `wandb`
- **Drift Analysis**: `evidently`
- **Visualization & UI**: `pydeck`, `streamlit`, `folium`

### Installation
```bash
pip install -r requirements.txt
```

### Experiment Tracking Setup
To log metrics and artifacts to W&B:
1. Create a [Weights & Biases](https://wandb.ai/) account.
2. Create a file named `wandb_key.txt` in the root directory and paste your API key inside.

---

## 🚀 Pipeline Steps

The pipeline is orchestrated in `pipeline.py` and consists of 10 sequential steps:

| Step | Phase | Description |
| :--- | :--- | :--- |
| **1** | **Cleaning** | Validates and cleans raw Parquet files (removes invalid IDs, dates, etc). |
| **1.1** | **Transformation** | Aggregates raw trips into 30-minute demand counts per `PULocationID`. |
| **2** | **Splitting** | Separates data into Training (2024-2025) and Test (2026, Jan-Feb) sets. |
| **3** | **Baseline** | Trains various ML models on the train dataset using the baseline features (no feature engineering). |
| **4** | **Engineering** | Trains various ML models on the train dataset with feature engineering. |
| **5** | **Comparison** | Evaluates the impact of feature engineering on MAE/RMSE. |
| **6** | **Champion** | Selects the best model and logs feature importance to W&B. |
| **7** | **Tuning** | Performs W&B Sweeps (Random Search → Grid Search) for optimal hyperparameters. |
| **8** | **Error Analysis** | Breaks down errors by time and location to identify model weaknesses. |
| **9 & 9.1**| **Drift Detection**| Monitors 2026 data for distribution shifts using PSI and Evidently AI reports. |
| **10** | **Mitigation** | Automatically triggers feature dropping if drift is detected. |

---

## 📊 Data Strategy & Splitting

The project uses a sophisticated temporal splitting strategy to simulate a real-world production environment where a model trained on historical data must handle future shifts.

### 1. The Temporal Split
*   **Training Set**: Data from **January 2024 through December 2025**. This is used for all feature engineering and initial model training.
*   **Test Set**: Data from **January and February 2026**. This simulates "incoming" production data where seasonality and demand patterns may have shifted.

### 2. The Drift Reference Set
To detect drift accurately, we extract a **Reference Set** from the 2024-2025 training data:
*   **Method**: 1 random week is extracted from every month across the 2-year training period.
*   **Purpose**: This creates a stable "In-Distribution" baseline (24 weeks total) that represents the variety of historical demand.
*   **Separation**: These weeks are strictly removed from the training set to ensure the drift baseline is never "seen" during the learning phase of the models.

### 3. Subsampling for Experimentation
During the iterative development steps (steps 3–6 & tuning part of step 7), the pipeline uses subsamples to keep runtimes fast:
*   **Training Sample**: 50,000 random rows from the 2024-2025 set.
*   **Test Sample**: 10,000 random rows from the 2026 set.

### 4. Step-by-Step Data Flow
| Step | Data Source | Purpose |
| :--- | :--- | :--- |
| **3-4 (Experiments)** | Subsampled 2024-25 | Rapid iteration and feature engineering experimentation on a 50k-row subset from the 2024-2025 train dataset. |
| **7 (Tuning)** | Subsampled 2024-25 | Performing grid search and training the optimized model on a 50k-row subset from the 2024-2025 train dataset. |
| **7 (Evaluation)** | 2026, Jan-Feb | Evaluating the final tuned model on the test dataset. |
| **8 (Error Analysis)**| 2026, Jan-Feb | Identifying where the model fails on the test dataset. |
| **9-10 (Drift Monitoring & Mitigation)** | **Ref**: 24-week baseline <br> **Cur**: 2026, Jan-Feb | Measuring distribution drift using PSI and Jensen–Shannon distance, followed by drift mitigation strategies when significant drift is detected. |

---

## 📂 Directory Structure

```text
├── data/
│   ├── raw/                # Original data from nyc.gov 
│   ├── monitoring/         # Processed data used for drift detection 
│   └── processed/          # Processed data used for model training & evaluation
├── src/                    # Modular source code
│   ├── cleaning.py         # Data validation & cleaning logic
│   ├── features.py         # Feature engineering pipelines
│   ├── drift_detection.py  # Monitoring & PSI calculations
│   └── ...                 
├── models/                 # Saved model artifacts (.pkl)
├── outputs/                # Generated plots and Evidently HTML reports
├── pipeline.py             # Main orchestration script
├── interactive_heatmap.py    # Interactive Streamlit dashboard
└── .github/workflows/      # CI/CD pipeline automation
```

---

## ⚙️ How to Run

### Execute Full Pipeline Locally
```bash
python pipeline.py --wandb-project "your-project-name"
```

### Run via GitHub Actions
The project includes a CI/CD workflow in `.github/workflows/pipeline.yml`. It is triggered on:
- **Push to main**: When code or data changes.
- **Schedule**: Every Monday at 07:00 UTC to simulate a weekly production monitoring run.

### Interactive Analysis
- `EDA.ipynb`: Exploratory Data Analysis, smart rounding logic, and geographic demand patterns.
- `Assignment Instructions.pdf`: Detailed project requirements and pedagogical context.

---

## 🗺️ Interactive Heatmap Dashboard

The project includes a Streamlit-based interactive dashboard to visualize predicted taxi demand across NYC.

### Features
- **Predictive Heatmap**: Visualizes predicted demand for all 263 NYC taxi zones.
- **Future Forecasting**: Select any date/time between today and 364 days in the future.
- **Auto-Model Selection**: Automatically loads the most recent model from `models/tuned/`.

### How to Run
```bash
streamlit run interactive_heatmap.py
```

---

## 📊 Monitoring & Reports
When the pipeline finishes, it generates:
1. **W&B Dashboard**: Real-time charts for MAE, RMSE, and Hyperparameter Sweeps.
2. **Evidently Reports**: `outputs/evidently_drift_report.html` for deep-dive feature analysis.
3. **Mitigation Comparison**: Plots showing performance "Before vs. After" drift mitigation.
