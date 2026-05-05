# 🛒 Rossmann Store Sales — MLP + Explainable AI

> Predicting retail sales with a deep Multi-Layer Perceptron and explaining every prediction using three complementary XAI methods: **GradientShap**, **LIME**, and **Integrated Gradients**.

---

## 📌 Project Overview

This project applies **explainable machine learning** to the [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset — a real-world retail forecasting task involving over 1,000 German drug stores.

Rather than treating the model as a black box, the notebook goes beyond prediction: it uses three distinct XAI techniques from the [Captum](https://captum.ai/) library to understand *why* the model makes each sales forecast. This dual focus on **performance + interpretability** reflects the demands of real-world decision-support systems in business and healthcare contexts.

---

## 🏗️ Pipeline Summary

```
Data (train.csv + store.csv)
        │
        ▼
 1. Preprocessing
    ├── Merge datasets, fix types, fill missing values
    ├── Filter closed stores & zero-sales rows (IQR outlier removal)
    ├── Feature engineering (calendar, competition, promo, German state)
    ├── Lag & rolling sales features (1, 2, 3, 7, 14-day lags; 7, 14, 30-day windows)
    └── Time-based split → Train / Val (6 wks) / Test (6 wks)
        │
        ▼
 2. Model — SalesMLP (PyTorch)
    ├── 4-layer MLP: 256 → 128 → 64 → 1
    ├── ReLU + Dropout regularisation
    ├── Target: log(1 + Sales) — stabilises variance across stores
    ├── Adam optimiser + ReduceLROnPlateau scheduler
    └── Early stopping (patience = 12)
        │
        ▼
 3. Evaluation (original Sales scale, €)
    ├── MAPE — scale-independent error
    ├── MAE, RMSE — absolute errors in euros
    └── R² — proportion of variance explained
        │
        ▼
 4. XAI Explanations (Captum)
    ├── GradientShap   → global feature importances + beeswarm
    ├── LIME           → per-instance local explanations + multi-instance comparison
    └── Integrated Gradients → magnitude + signed attribution charts
```

---

## 🧠 XAI Methods

### 1. GradientShap
Approximates SHAP values using stochastic gradients sampled between input and a training-mean baseline. Produces both a **global importance bar chart** and a **beeswarm plot** showing how feature values correlate with their attributions.

### 2. LIME (Local Interpretable Model-agnostic Explanations)
Fits a local linear surrogate around each individual prediction by perturbing the input and observing output changes. Generates a **per-instance bar chart** and a **5-instance comparison chart** to reveal consistency or variability in feature influence.

### 3. Integrated Gradients
Integrates gradients along a path from a baseline to the actual input, with a **mathematical completeness guarantee** (attributions sum exactly to `f(x) − f(baseline)`). Generates a **magnitude chart** (what matters) and a **signed chart** (which direction).

> All three methods consistently rank the same top features — **Customers > Promo > LogCompetitionDistance > StoreMonth_avg > Sales lags** — giving high confidence in the model's learned relationships.

---

## 📊 Features Engineered

| Category | Features |
|---|---|
| **Calendar** | Day, Month, Year, Week, Quarter, DayOfWeek, IsWeekend, IsMonthStart/End |
| **Promotion** | Promo, Promo2, Promo2Active, PromoCombo |
| **Competition** | CompetitionDistance, LogCompetitionDistance, CompetitionOpen (months) |
| **Store** | StoreType, Assortment, German State (mapped) |
| **Temporal** | Sales_lag_{1,2,3,7,14}, Sales_roll_mean_{7,14,30}, Sales_roll_std_{7,14,30} |
| **Aggregate** | StoreMonth_avg (per-store monthly mean log-sales) |

---

## 📁 Repository Structure

```
rossmann-xai/
├── rossmann_xai_colab.ipynb     # Main notebook (run on Google Colab)
├── README.md
└── outputs/                     # Generated after running the notebook
    ├── mlp_rossmann.pth         # Trained model weights + metrics
    ├── preprocessor.pkl         # Fitted OHE + StandardScaler
    ├── training_curves.png      # Train MSE & Val RMSE over epochs
    ├── shap_bar_importance.png  # GradientShap — top-25 global importances
    ├── shap_beeswarm.png        # GradientShap — beeswarm (top-15)
    ├── lime_explanation.png     # LIME — single instance
    ├── lime_multi_explanation.png  # LIME — 5-instance comparison
    ├── ig_feature_importance.png   # IG — magnitude chart
    └── ig_signed_importance.png    # IG — signed direction chart
```

---

## ⚙️ Setup & Usage

### 1. Open in Google Colab
This notebook is designed to run on **Google Colab** (free GPU available).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Download the dataset
Download from [Kaggle — Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data):
- `train.csv`
- `store.csv`

Upload both to Google Drive.

### 3. Update the data paths in Cell 0
```python
TRAIN_PATH = '/content/drive/MyDrive/YOUR_FOLDER/train.csv'
STORE_PATH = '/content/drive/MyDrive/YOUR_FOLDER/store.csv'
```

### 4. Install the one extra dependency
```bash
!pip install -q captum
```
All other libraries (`torch`, `sklearn`, `pandas`, `numpy`, `matplotlib`) are pre-installed on Colab.

### 5. Run all cells top-to-bottom
Cells are numbered `0 → 7` following the notebook structure. Each section is self-contained with detailed markdown explanations.

---

## 🔧 Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `EPOCHS` | 80 | With early stopping (patience = 12) |
| `BATCH_SIZE` | 2048 | Large batches → stable tabular gradients |
| `LR` | 1e-3 | Adam; halved on plateau |
| `WEIGHT_DECAY` | 1e-5 | L2 regularisation |
| Dropout | 0.3 / 0.2 | Layers 1 and 2 |
| `N_SHAP_SAMP` | 50 | GradientShap MC draws |
| `N_IG_STEPS` | 50 | Integration steps (≥50 = reliable) |
| `N_LIME_SAMPLES` | 1000 | Perturbations per LIME explanation |

---

## 📦 Dependencies

```
torch
captum
scikit-learn
pandas
numpy
matplotlib
```

---

## 🔑 Key Findings

- **Customers** is by far the most important feature across all three XAI methods — foot traffic dominates sales prediction.
- **Promo** is the strongest actionable lever — promotions consistently raise predictions.
- **LogCompetitionDistance** matters — stores further from competitors tend to sell more.
- **Temporal lag features** (Sales_lag_1, Sales_roll_mean_14) confirm strong autocorrelation in retail sales.
- All three XAI methods (GradientShap, LIME, IG) agree on feature ranking, providing cross-method validation of the model's learned behaviour.

---

## 🔗 References

- [Rossmann Store Sales — Kaggle Competition](https://www.kaggle.com/competitions/rossmann-store-sales)
- [Captum — Model Interpretability for PyTorch](https://captum.ai/)
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Ribeiro et al. (2016) — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME)
- Sundararajan et al. (2017) — *Axiomatic Attribution for Deep Networks* (Integrated Gradients)

---

