# Hotel Booking Cancellation Prediction

This repository implements a production-grade machine learning project for predicting hotel booking cancellations using the Kaggle Hotel Booking Demand dataset. 

## Business Value

Hotel cancellations disrupt operations, lead to lost revenue from unoccupied rooms, and complicate resource planning. This end-to-end machine learning pipeline enables revenue managers and hotel operators to:
- **Optimize Overbooking Policies**: Quantify cancellation risks dynamically to determine optimal overbooking buffers without risking overcapacity penalties.
- **Tailor Deposit and Cancellation Policies**: Automatically flag high-risk bookings and apply targeted non-refundable deposit requirements.
- **Improve Resource Allocation**: Schedule staff, order supplies, and manage inventory based on reliable, probability-weighted guest arrival forecasts.

---

## Project Structure

```
d:\finance/
├── data/
│   ├── raw/                 # Raw dataset (hotel_bookings.csv)
│   └── processed/           # Split, cleaned, and engineered features (X_train, X_test, etc.)
├── models/
│   ├── preprocessor.joblib  # Serialized BookingPreprocessor object (imputers, encoders, scaler)
│   ├── logistic_regression.joblib  # Serialized Baseline Logistic Regression model
│   └── lightgbm.joblib      # Serialized Optimized LightGBM model
├── outputs/
│   ├── evaluation_metrics.json  # Raw evaluation metrics JSON
│   ├── evaluation_summary.txt   # Human-readable comparison log
│   └── shap_summary.png     # Global feature importance visualization
├── src/
│   ├── download_data.py     # Script to download dataset with synthetic offline fallback
│   ├── preprocess.py        # Cleaning, engineering, splitting, and encoding pipeline
│   ├── train.py             # Trains and evaluates LR and LightGBM models
│   └── interpret.py         # SHAP explainability and local risk factors explanation
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

---

## Installation & Setup

1. **Clone or navigate** to the project workspace:
   ```powershell
   cd d:\finance
   ```

2. **Install dependencies** using `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```

---

## Pipeline Execution

The pipeline is split into modular components that can be executed sequentially:

### Step 1: Download Data
Downloads the official Kaggle dataset mirror or runs a synthetic fallback generator if offline.
```powershell
python src/download_data.py
```

### Step 2: Preprocess Data
Cleans missing values, engineers operational features, splits datasets (80/20 stratified), and applies leak-safe target encoding.
```powershell
python src/preprocess.py
```

### Step 3: Train and Evaluate Models
Trains baseline Logistic Regression and optimized LightGBM models, logging performances.
```powershell
python src/train.py
```

### Step 4: Model Interpretation (SHAP)
Generates global feature importance plots and runs local prediction explanation queries.
```powershell
# Run global plot generation & default local explanation
python src/interpret.py

# Explain a specific test sample index (e.g. index 42)
python src/interpret.py --index 42
```

---

## Machine Learning Design & Architecture

### 1. Data Leakage Mitigation
To ensure the model generalizes perfectly to production data:
- **Stratified Split**: We split the dataset (80/20 train/test) before any feature scaling, one-hot encoding, or target encoding.
- **Target Encoding**: High-cardinality features (`country`, `agent`, `company`) are encoded using a custom **Smoothed Target Encoder**. The encoding dictionaries are learned **only** on the training split. Unseen values in the test split are mapped to the global training target mean.
- **Leakage Columns Removed**: Features directly indicating final status (`reservation_status` and `reservation_status_date`) are dropped during preprocessing, preventing target leakage.

### 2. Engineered Operational Features
- `total_stay`: Combined number of week and weekend nights.
- `total_guests`: Combined count of adults, children, and babies.
- `has_special_requests`: Binary flag indicating whether the booking has at least one special request.
- `is_extreme_lead_time`: Binary flag denoting whether `lead_time > 200` days.

---

## Model Performance Summary

The evaluation results on the hold-out test set are summarized below:

| Metric | Logistic Regression (Baseline) | LightGBM (Optimized) | Delta (LGBM - LR) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.57% | **88.25%** | +5.68% |
| **Precision** | 81.36% | **86.26%** | +4.90% |
| **Recall** | 68.66% | **81.22%** | +12.56% |
| **F1-Score** | 74.47% | **83.67%** | +9.20% |
| **ROC-AUC** | 90.42% | **95.50%** | +5.08% |
| **PR-AUC (AP)** | 86.46% | **93.32%** | +6.86% |

### Key Takeaway
The **LightGBM Classifier** significantly outperforms the Logistic Regression model, especially in **Recall (+12.56%)**, indicating it is much more effective at capturing true cancellations. The high **PR-AUC (93.32%)** indicates excellent precision-recall trade-offs, critical for deciding which bookings require deposit upgrades.

---

## Explainability (SHAP Values)

Global model decisions are visualized in `outputs/shap_summary.png`, which plots SHAP values to explain feature contributions.

### Local Booking Risk Explanations
When running `src/interpret.py`, local predictions show exactly which features pushed the model's decision:
- **High-Risk Indicators**: High `lead_time`, `agent` identifiers associated with high cancellation rates, and lack of special requests.
- **Low-Risk Indicators**: Specific `country` origins, bookings with multiple `booking_changes`, and presence of parking space requests.
