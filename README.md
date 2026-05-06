```markdown
# 💳 Fraud Detection AI

> An end-to-end machine learning system for detecting fraudulent credit card
> transactions — with SHAP-powered explainability and Hugging Face language
> generation to turn model decisions into plain English.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Running the Dashboard](#running-the-dashboard)
- [Models Trained](#models-trained)
- [Evaluation Metrics](#evaluation-metrics)
- [Explainability](#explainability)
- [Generated Outputs](#generated-outputs)
- [Configuration](#configuration)
- [Requirements](#requirements)

---

## Project Overview

Credit card fraud is a critical problem in financial services. Fraudulent
transactions are rare — typically less than 0.2% of all activity — which makes
detection extremely difficult using standard machine learning approaches.

This project builds a **complete, production-style AI pipeline** that:

- Ingests raw transaction data from the Kaggle Credit Card Fraud dataset
- Engineers meaningful features from raw inputs
- Balances the heavily skewed dataset using **SMOTE** (Synthetic Minority
  Oversampling Technique)
- Trains and compares **four machine learning models**
- Evaluates them using metrics suited to imbalanced classification
- Explains every prediction using **SHAP** (SHapley Additive exPlanations)
- Converts technical SHAP output into **plain English** using a Hugging Face
  language model
- Presents everything through an interactive **Streamlit dashboard**

---

## Key Features

| Feature | Description |
|---|---|
| Multi-model training | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Class imbalance handling | SMOTE resampling before training |
| Rigorous evaluation | ROC-AUC, PR-AUC, Precision, Recall, F1 |
| SHAP explainability | Per-prediction feature contribution breakdown |
| Human-readable explanations | Hugging Face DistilGPT-2 narrates the SHAP output |
| Interactive UI | Streamlit dashboard with manual, random, and CSV input modes |
| Reproducible pipeline | Seeded RNGs, saved scalers, and saved feature name order |
| Automated reporting | Markdown report generated after every training run |

---

## How It Works

The pipeline runs in eight sequential stages:

```
Raw CSV
   │
   ▼
1. Load Data          → reads transactions.csv from data/raw/
   │
   ▼
2. EDA                → saves exploratory charts to reports/figures/
   │
   ▼
3. Preprocessing      → feature engineering → train/test split → scaling
   │
   ▼
4. SMOTE Resampling   → balances the fraud class before training
   │
   ▼
5. Model Training     → trains 4 models and saves them to models/
   │
   ▼
6. Evaluation         → scores all models and ranks by ROC-AUC
   │
   ▼
7. Explainability     → SHAP values + Hugging Face plain-English summary
   │
   ▼
8. Report             → writes reports/report.md with results table
```

---

## Project Structure

```
fraud-detection-ai/
│
├── app/
│   └── streamlit_app.py          # Streamlit UI dashboard
│
├── data/
│   ├── raw/
│   │   └── transactions.csv      # ← Place Kaggle dataset here
│   ├── processed/
│   │   ├── X_train.csv           # Scaled training features
│   │   ├── X_test.csv            # Scaled test features
│   │   ├── y_train.csv           # Training labels
│   │   ├── y_test.csv            # Test labels
│   │   ├── scaler.pkl            # Fitted StandardScaler
│   │   └── feature_names.json    # Ordered feature column names
│   └── external/
│       └── huggingface_cache/    # Cached HF model weights
│
├── models/
│   ├── fraud_model.pkl           # Best model (chosen by ROC-AUC)
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── lightgbm_model.pkl
│
├── notebooks/                    # Jupyter notebooks for exploration
│
├── reports/
│   ├── figures/                  # All saved charts (EDA + evaluation)
│   └── report.md                 # Auto-generated training report
│
├── src/
│   ├── data/
│   │   ├── load_data.py          # Loads raw CSV from disk
│   │   ├── preprocess.py         # Cleaning, splitting, and scaling
│   │   └── feature_engineering.py # Derives smart features from raw columns
│   │
│   ├── models/
│   │   ├── train_model.py        # Trains all candidate models
│   │   ├── evaluate_model.py     # Scores and ranks all models
│   │   ├── predict.py            # Single-transaction prediction pipeline
│   │   └── huggingface_model.py  # Hugging Face plain-English explanation
│   │
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP values and summary plots
│   │
│   ├── visualization/
│   │   ├── eda.py                # Exploratory data analysis charts
│   │   └── plots.py              # Confusion matrix, ROC, PR, importance plots
│   │
│   └── utils/
│       ├── config.py             # All paths and hyperparameters
│       └── helpers.py            # Shared utility functions
│
├── main.py                       # Pipeline entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the
Machine Learning Group at ULB (Université Libre de Bruxelles).

**Download it here:**
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**After downloading, place the file at:**

```
data/raw/transactions.csv
```

### Dataset Facts

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | 30 (Time, V1–V28 PCA components, Amount) |
| Target column | Class (0 = legitimate, 1 = fraud) |
| File format | CSV |

> **Note:** The V1–V28 columns are the result of PCA transformation applied by
> the dataset authors to protect cardholder privacy. No original feature names
> are available for these columns.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-ai.git
cd fraud-detection-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Download `creditcard.csv` from Kaggle, rename it to `transactions.csv`, and
place it at:

```
data/raw/transactions.csv
```

---

## Running the Project

To run the full pipeline from raw data to saved models and reports:

```bash
python main.py
```

This will execute all eight pipeline stages in order and produce:

- Processed data splits in `data/processed/`
- Trained models in `models/`
- EDA and evaluation charts in `reports/figures/`
- A final Markdown report at `reports/report.md`

---

## Running the Dashboard

Once `main.py` has completed at least once:

```bash
streamlit run app/streamlit_app.py
```

The dashboard offers three input modes:

| Mode | Description |
|---|---|
| Manual | Type raw transaction values directly into the form |
| Random Sample | Pick a random row from the dataset and predict it |
| Upload CSV | Upload a single-row CSV file for prediction |

Each prediction displays:

- Fraud probability score
- Risk level (Low / Medium / High)
- Pass or fail banner
- Plain-English Hugging Face explanation
- Top SHAP feature contributions

---

## Models Trained

| Model | Key Settings |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight=balanced` |
| Random Forest | `n_estimators=200`, `class_weight=balanced_subsample` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=5` |
| LightGBM | `n_estimators=200`, `learning_rate=0.05` |

All models are trained on SMOTE-resampled data where the fraud class is
resampled to 20% of the majority class size (`SMOTE_RATIO = 0.2`).

---

## Evaluation Metrics

Standard accuracy is misleading for fraud detection — a model that always
predicts legitimate would achieve over 99% accuracy while catching zero fraud
cases. This project uses the following metrics instead:

| Metric | Why It Matters |
|---|---|
| **ROC-AUC** | Measures ranking quality across all thresholds — primary sort key |
| **PR-AUC** | Precision-Recall area under curve — best for imbalanced datasets |
| **Precision** | Of all flagged transactions, how many were actually fraud |
| **Recall** | Of all actual fraud cases, how many were caught |
| **F1 Score** | Harmonic mean of precision and recall |
| **Accuracy** | Reported for reference only — not used for model selection |

Models are ranked by **ROC-AUC**. The best model is saved separately to
`models/fraud_model.pkl` for use by the prediction pipeline and dashboard.

---

## Explainability

Fraud detection systems must be explainable — regulators and end users need to
understand why a transaction was flagged. This project uses two layers of
explainability:

### SHAP (SHapley Additive exPlanations)

SHAP assigns each feature a contribution score for every individual prediction.
A positive score pushes the model toward fraud; a negative score pushes it
toward safe.

Example SHAP output:
```
V14 (-2.341) pushes toward safe,
Amount_Log (+1.823) pushes toward fraud,
Hour (+0.912) pushes toward fraud
```

### Hugging Face Plain-English Explanation

The SHAP summary is passed as a prompt to a **DistilGPT-2** language model,
which generates a short, beginner-friendly explanation of the prediction. If
the model is unavailable, a clean static fallback is returned automatically.

Example output:
```
This transaction looks suspicious because the amount is unusually large for
this time of day, and several anonymised signals are elevated. The model
estimated a fraud risk of 0.91.
```

---

## Generated Outputs

After running `main.py`, the following files are created automatically:

| File | Description |
|---|---|
| `data/processed/X_train.csv` | Scaled training feature matrix |
| `data/processed/X_test.csv` | Scaled test feature matrix |
| `data/processed/y_train.csv` | Training labels |
| `data/processed/y_test.csv` | Test labels |
| `data/processed/scaler.pkl` | Fitted StandardScaler for inference |
| `data/processed/feature_names.json` | Ordered feature column list |
| `models/fraud_model.pkl` | Best model selected by ROC-AUC |
| `models/random_forest.pkl` | Saved Random Forest |
| `models/xgboost_model.pkl` | Saved XGBoost |
| `models/lightgbm_model.pkl` | Saved LightGBM |
| `reports/figures/class_balance.png` | Fraud vs legitimate count chart |
| `reports/figures/amount_distribution.png` | Transaction amount histogram |
| `reports/figures/time_distribution.png` | Transaction time histogram |
| `reports/figures/correlation_heatmap.png` | Feature correlation heatmap |
| `reports/figures/*_confusion_matrix.png` | Confusion matrix per model |
| `reports/figures/*_roc_curve.png` | ROC curve per model |
| `reports/figures/*_pr_curve.png` | Precision-Recall curve per model |
| `reports/figures/*_feature_importance.png` | Feature importance per model |
| `reports/figures/shap_summary.png` | SHAP beeswarm plot for best model |
| `reports/report.md` | Full Markdown training summary |
| `reports/metrics.csv` | Model comparison table |

---

## Configuration

All paths and hyperparameters are centralised in `src/utils/config.py`.
Nothing is hardcoded anywhere else in the codebase.

| Setting | Default | Description |
|---|---|---|
| `RANDOM_STATE` | `42` | Seed for all RNGs — ensures reproducibility |
| `TEST_SIZE` | `0.2` | Fraction of data reserved for evaluation |
| `THRESHOLD` | `0.5` | Minimum probability to classify as fraud |
| `SMOTE_RATIO` | `0.2` | Fraud class target ratio after resampling |
| `LARGE_AMOUNT_THRESHOLD` | `200.0` | USD threshold for the Is_Large_Amount flag |
| `HF_MODEL_NAME` | `distilgpt2` | Hugging Face model for text generation |

---

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
transformers
joblib
matplotlib
seaborn
streamlit
tabulate
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```
```
