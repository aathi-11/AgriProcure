#!/usr/bin/env python3
"""
Generate the Task 6 Final Documentation Notebook for AgriProcure Review 1.

This script creates a single, runnable, well-commented Jupyter notebook (.ipynb)
that combines all Review 1 work (Tasks 1-6) into one coherent document with:
  - Dataset documentation
  - Model evaluation against baselines (macro-F1, per-class precision/recall, confusion matrices)
  - Feature importance analysis
  - Business findings and interpretation

Usage:
    python build_task6_notebook.py
"""

import json, os

# ---------------------------------------------------------------------------
# Helper to build notebook cells
# ---------------------------------------------------------------------------
def md(source):
    """Return a markdown cell dict."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n") if isinstance(source, str) else source}

def code(source, eid=None):
    """Return a code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }

cells = []

# ===================================================================
# SECTION 0 — Title and TOC
# ===================================================================
cells.append(md("""\
# AgriProcure — Final Review 1 Notebook
## From Field to Forecast: Predicting Weekly Price Movements of Tomato, Onion & Potato in Indian Mandis

**Course:** 23CSE452 Business Analytics — Capstone Project  
**Review:** 1 — Data Analysis and Predictive Modelling (Units 1 & 2)

---

### Table of Contents

| # | Section | Review 1 Component |
|---|---------|---------------------|
| 1 | Problem Statement & Business Objective | Problem statement, business objective & dataset description (2 marks) |
| 2 | Dataset Documentation | Dataset source, collection, data dictionary |
| 3 | Data Cleaning & Preprocessing | Data cleaning & preprocessing (3 marks) |
| 4 | Exploratory Data Analysis & Visualization | EDA & visualization (3 marks) |
| 5 | Feature Engineering & Dimensionality Reduction | Feature engineering / dim. reduction (2 marks) |
| 6 | Model Development | Predictive model development (4 marks) |
| 7 | **Model Evaluation & Result Interpretation** | **Model evaluation & result interpretation (3 marks)** |
| 8 | Business Findings & Recommendations | Initial business findings |
| 9 | Dataset Documentation Appendix | Data dictionary & collection details |

> **Note:** Sections 1–6 summarise work completed in the upstream notebooks (Tasks 1–5).  
> **Section 7 (Model Evaluation) and Section 8 (Business Insights)** form the primary additions for this notebook (Task 6)."""))

# ===================================================================
# SECTION 1 — Problem Statement
# ===================================================================
cells.append(md("""\
---
## 1. Problem Statement & Business Objective

### 1.1 Business Problem

Wholesale prices of perishable vegetables — **Tomato**, **Onion** and **Potato** — swing sharply from week to week across India's Agricultural Produce Market Committees (mandis). These swings are driven by seasonality, supply arrivals and weather events (heavy rain, heat waves).

**Buyers** (food processors, supermarket chains) and **sellers** (farmer-producer organisations, FPOs) currently rely on experience and guesswork to time purchases and sales. A processor who buys just before a price spike pays more than necessary; an FPO that sells just before a spike loses income.

### 1.2 Business Objective

> **Develop a predictive model providing food processors and FPOs with data-driven forecasts of short-term price direction, optimizing procurement timing and mitigating market volatility risks.**

| Stakeholder | Decision | How This Project Helps |
|---|---|---|
| Food processors | When to buy raw material | Weekly price-direction signal assists in avoiding procurement immediately prior to spikes |
| FPOs | When to sell or hold produce | Forecasts provide visibility into optimal periods for market liquidation |
| Supply-chain planners | Inventory and budget planning | Weather-linked rules act as early warning indicators for supply disruptions |

### 1.3 Prediction Task

**3-class classification:** Predict whether next week's modal wholesale price will **rise**, **fall** or stay **flat**.

$$\\text{target\\_direction}_{t+1} = \\begin{cases} \\text{fall}, & \\Delta\\% < -2.5\\% \\\\ \\text{flat}, & -2.5\\% \\le \\Delta\\% \\le +2.5\\% \\\\ \\text{rise}, & \\Delta\\% > +2.5\\% \\end{cases}$$

where $\\Delta\\% = (P_{t+1} / P_t - 1) \\times 100$.

### 1.4 Scope

| Item | Choice |
|---|---|
| Crops | Tomato, Onion, Potato |
| Markets | 70 mandis across 14 Indian states |
| History | ~1 year (Sep 2025 – Sep 2026) |
| Granularity | Market × crop × week panel |
| Primary metric | **Macro-F1** (treats all 3 classes equally) |"""))

# ===================================================================
# SECTION 2 — Dataset Documentation
# ===================================================================
cells.append(md("""\
---
## 2. Dataset Documentation

### 2.1 Data Sources

All data was **collected by the team** — no pre-built datasets from Kaggle/UCI/GitHub.

| Data | Source | Method | Period |
|---|---|---|---|
| Mandi prices & arrivals | CEDA Agri Market API (Agmarknet archives) | API collection script (`scripts/`) | Sep 2025 – Sep 2026 |
| Rainfall & temperature | Open-Meteo Historical Weather API | API collection script (`scripts/collect_weather_openmeteo.py`) | Same period |

### 2.2 Dataset Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `data/preprocessed dataset.csv` | ~335K | — | Raw daily mandi prices |
| `data/cleaned_mandi_daily_dataset.csv` | ~228K | — | After deduplication and cleaning |
| `data/cleaned_mandi_crop_week_panel.csv` | 11,130 | — | Weekly aggregated panel (70 mandis × 3 crops × 53 weeks) |
| `data/weather_daily_by_mandi.csv` | 25,830 | — | Daily weather per mandi |
| `data/weather_weekly_by_mandi.csv` | 3,710 | — | Weekly aggregated weather |
| `data/panel_with_weather.csv` | 11,130 | — | Price panel merged with weather |
| `data/features_panel.csv` | 9,030 | 46 | **Modeling dataset** — 38 engineered features + identifiers + target |
| `data/features_pca.csv` | 9,030 | 22 | PCA-reduced features (14 PCs, 91.4% variance) |

### 2.3 Weather Data Dictionary

| Column | Meaning |
|---|---|
| `weekly_rainfall_mm` | Total precipitation over the week (mm) |
| `rainy_days` | Days with ≥ 2.5 mm rain (IMD "rainy day") |
| `max_daily_rainfall_mm` | Rainfall on the wettest day of the week (mm) |
| `weekly_temp_mean_c` | Mean daily temperature (°C) |
| `weekly_temp_max_c` / `weekly_temp_min_c` | Highest daily max / lowest daily min in the week |
| `weekly_humidity_mean_pct` | Mean daily relative humidity (%) |"""))

# ===================================================================
# SECTION 3 — Setup & Data Loading (CODE)
# ===================================================================
cells.append(md("""\
---
## 3. Setup & Data Loading

We load the primary datasets, report results from upstream notebooks, and prepare for the evaluation in Section 7."""))

cells.append(code("""\
# ── Imports ──────────────────────────────────────────────────────────
import os, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Suppress noisy warnings in the notebook
warnings.filterwarnings('ignore')

# ── Plot defaults ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'figure.dpi': 120,
    'font.sans-serif': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.figsize': (10, 5),
})

# ── Colour palette ───────────────────────────────────────────────────
COLOURS = {'fall': '#e74c3c', 'flat': '#f39c12', 'rise': '#27ae60'}
PALETTE = [COLOURS[c] for c in ['fall', 'flat', 'rise']]

# ── Paths ────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.getcwd(), '..')
DATA       = os.path.join(ROOT, 'data')
REPORTS    = os.path.join(ROOT, 'reports')
MODELS     = os.path.join(ROOT, 'models')
FIGURES    = os.path.join(REPORTS, 'figures')

os.makedirs(os.path.join(FIGURES, 'task6'), exist_ok=True)

print("Setup complete ✓")"""))

cells.append(code("""\
# ── Load primary feature panel ───────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, 'features_panel.csv'))
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

# ── Load PCA dataset ────────────────────────────────────────────────
df_pca = pd.read_csv(os.path.join(DATA, 'features_pca.csv'))
df_pca['week_start_date'] = pd.to_datetime(df_pca['week_start_date'])

# ── Chronological train/test split ───────────────────────────────────
train_df = df[df['split'] == 'train'].copy()
test_df  = df[df['split'] == 'test'].copy()

print(f"Total rows      : {len(df):,}")
print(f"Training rows   : {len(train_df):,}  ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test rows       : {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")
print(f"Train period    : {train_df['week_start_date'].min().date()} → {train_df['week_start_date'].max().date()}")
print(f"Test period     : {test_df['week_start_date'].min().date()} → {test_df['week_start_date'].max().date()}")
print(f"Crops           : {sorted(df['crop'].unique())}")
print(f"States          : {df['state'].nunique()}")
print(f"Mandis          : {df['market_name'].nunique()}")
print(f"Features (cols) : {df.shape[1]}")"""))

cells.append(code("""\
# ── Target distribution overview ─────────────────────────────────────
for label, subset in [('Full', df), ('Train', train_df), ('Test', test_df)]:
    vc = subset['target_direction'].value_counts(normalize=True).sort_index()
    counts = subset['target_direction'].value_counts().sort_index()
    print(f"\\n{label} target distribution:")
    for cls in ['fall', 'flat', 'rise']:
        print(f"  {cls:5s}  {counts.get(cls, 0):5,}  ({vc.get(cls, 0)*100:5.1f}%)")"""))

# ===================================================================
# SECTION 4 — Data Cleaning & Preprocessing Summary
# ===================================================================
cells.append(md("""\
---
## 4. Data Cleaning & Preprocessing (Summary)

The raw daily mandi-price dataset was cleaned through the following pipeline:

| Step | Action | Rows Before → After |
|---|---|---|
| 1 | Remove exact duplicates | 335K → ~228K |
| 2 | Standardise mandi/crop names | — |
| 3 | Handle missing prices (median imputation within mandi×crop) | — |
| 4 | Aggregate to weekly panel (Monday-start weeks) | 228K → 11,130 |
| 5 | Flag partial final week (2 trading days) | — |
| 6 | Merge weekly weather by mandi location | 11,130 (panel_with_weather) |
| 7 | Engineer features, drop warm-up weeks (first 8) | 11,130 → 9,030 |

**Data quality:** No missing values or infinities remain in `features_panel.csv`. All 9,030 rows are complete."""))

# ===================================================================
# SECTION 5 — EDA Summary
# ===================================================================
cells.append(md("""\
---
## 5. Exploratory Data Analysis (Summary)

Key EDA findings (full analysis in `notebooks/Task3_EDA_Features_PCA.ipynb`):"""))

cells.append(code("""\
# ── Display key EDA charts from Task 3 ──────────────────────────────
from IPython.display import Image, display

eda_charts = [
    ('chart1_national_price_trend.png',
     'Chart 1: National average modal price by crop.\\n'
     '→ Tomato (≈₹2,380/q) > Onion (≈₹1,995/q) > Potato (≈₹1,550/q). All three move\\n'
     '  together in a quarterly sawtooth with no multi-year trend.'),
    ('chart2_week_of_quarter_seasonality.png',
     'Chart 2: Quarterly seasonality.\\n'
     '→ Prices start each quarter ~6% below average, climb ~1%/week, then drop ~10%\\n'
     '  at the quarter boundary. Business implication: buy early in the quarter.'),
    ('chart4_price_vs_arrivals.png',
     'Chart 4: Price vs. arrivals (supply).\\n'
     '→ Within a mandi, heavier-arrival weeks do NOT show lower prices (r ≈ 0).\\n'
     '  Supply-price dynamics are weak in this data window.'),
    ('chart8_weekly_change_distribution.png',
     'Chart 8: Distribution of weekly price changes.\\n'
     '→ A ±2.5% flat band gives balanced classes: fall 30%, flat 33%, rise 37%.'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (fname, caption) in zip(axes.flat, eda_charts):
    img_path = os.path.join(FIGURES, 'task3', fname)
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(caption.split('\\n')[0], fontsize=10, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'task6', 'eda_summary_grid.png'), dpi=150, bbox_inches='tight')
plt.show()

print("Key EDA takeaways:")
print("  1. Price mean-reversion is the dominant signal (ρ ≈ -0.38 with next week's change)")
print("  2. Calendar quarter sawtooth is the strongest seasonal pattern")
print("  3. Weather and arrivals have weak linear predictive power (|ρ| ≤ 0.06)")
print("  4. ±2.5% flat band yields ~balanced classes")"""))

# ===================================================================
# SECTION 6 — Feature Engineering Summary
# ===================================================================
cells.append(md("""\
---
## 6. Feature Engineering & Dimensionality Reduction (Summary)

### 6.1 Feature Groups (38 numeric features + 2 categorical)

| Group | Count | Key Features | Rationale |
|---|---|---|---|
| **Price history** | 14 | Lags (1,2,4 wk), % changes, rolling mean/std, price vs. rolling mean, cross-sectional ratios | Mean reversion is the dominant signal |
| **Arrivals** | 5 | Current, lag-1, % change, rolling mean, deviation from rolling mean | Supply-side fundamentals |
| **Weather** | 11 | Rainfall (current, lags, rolling sum), rain flags, temperature, heat flag, anomaly, humidity | Threshold effects for tree models |
| **Calendar** | 6 | `week_of_quarter`, `next_week_is_quarter_start`, `quarter`, `month`, sin/cos week-of-year | Captures sawtooth seasonality |
| **Identifiers** | 2 | `crop`, `state` (categorical) | Crop-level and regional differences |

### 6.2 Multicollinearity

- **23 of 38 features** have VIF > 10 (rolling price means reach VIF ≈ 6,692)
- This is expected — price lags and rolling statistics are near-duplicates of the price level
- Tree-based models are unaffected; for logistic regression, PCA-transformed features were used

### 6.3 PCA

- Fitted `StandardScaler → PCA` on training data only (no leakage)
- **14 principal components** retain **91.4%** of numeric feature variance
- PC1 captures price level, PC2 captures momentum/% change, PC3–4 capture weather/calendar"""))

cells.append(code("""\
# ── Feature importance from VIF analysis ─────────────────────────────
vif = pd.read_csv(os.path.join(REPORTS, 'feature_vif.csv'), index_col=0)
print("Top 10 features by VIF (multicollinearity indicator):")
print(vif.head(10).to_string())
print(f"\\nFeatures with VIF > 10: {(vif['VIF'] > 10).sum()} / {len(vif)}")"""))

# ===================================================================
# SECTION 7 — MODEL EVALUATION (CORE TASK 6 CONTRIBUTION)
# ===================================================================
cells.append(md("""\
---
## 7. Model Evaluation & Result Interpretation

This section details the performance of all trained models against the baseline frameworks, analyzing macro-F1 scores, per-class precision and recall metrics, and confusion matrices to synthesize final interpretations.

### 7.1 Cross-Validation Summary

All models used **expanding-window rolling-origin cross-validation** (3 folds) on the training data, with a strictly chronological split. The final held-out test set (Q3 2026, monsoon season) was **never seen** during training or validation."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.1  Load all model results from reports/
# ══════════════════════════════════════════════════════════════════════
model_results = pd.read_csv(os.path.join(REPORTS, 'model_results.csv'))

# Clean up for display
display_cols = ['baseline', 'mean_accuracy', 'mean_macro_f1', 'std_macro_f1',
                'mean_fall_recall', 'mean_flat_recall', 'mean_rise_recall']
results_display = model_results[display_cols].copy()
results_display.columns = ['Model', 'Accuracy', 'Macro-F1', 'Std F1',
                           'Fall Recall', 'Flat Recall', 'Rise Recall']

# Round for readability
results_display = results_display.round(4)
print("╔══════════════════════════════════════════════════════════════╗")
print("║     Cross-Validation Results (3-Fold Rolling Origin)       ║")
print("╚══════════════════════════════════════════════════════════════╝")
display(results_display.style
    .background_gradient(subset=['Macro-F1'], cmap='Greens')
    .set_caption('Models sorted by experiment order')
)"""))

cells.append(md("""\
### 7.2 Model Comparison: Macro-F1 (Primary Metric)

Macro-F1 treats all three classes equally — important because we care about detecting **falls** (buy signal) and **rises** (sell signal) equally, not just overall accuracy."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.2  Macro-F1 comparison bar chart
# ══════════════════════════════════════════════════════════════════════
# Separate baselines from ML models
baselines = model_results[model_results['experiment'].str.contains('Baseline')]
ml_models = model_results[~model_results['experiment'].str.contains('Baseline')]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot all models
models_sorted = model_results.sort_values('mean_macro_f1', ascending=True)
bars = ax.barh(
    models_sorted['baseline'],
    models_sorted['mean_macro_f1'],
    xerr=models_sorted['std_macro_f1'],
    color=['#bdc3c7' if 'Baseline' in e else '#3498db'
           for e in models_sorted['experiment']],
    edgecolor='white',
    linewidth=0.5,
    capsize=4,
    height=0.65
)

# Highlight the best model
best_idx = models_sorted['mean_macro_f1'].idxmax()
best_model = models_sorted.loc[best_idx, 'baseline']
best_f1 = models_sorted.loc[best_idx, 'mean_macro_f1']

# Mark best model
for bar, model_name in zip(bars, models_sorted['baseline']):
    if model_name == best_model:
        bar.set_color('#e74c3c')
        bar.set_edgecolor('#c0392b')
        bar.set_linewidth(2)

# Annotate values
for bar, (_, row) in zip(bars, models_sorted.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{row['mean_macro_f1']:.3f}", va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Mean Macro-F1 (± 1 std)', fontsize=12)
ax.set_title('Model Comparison: Macro-F1 Score\\n(Grey = Baselines, Blue = ML Models, Red = Best)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 0.75)
ax.axvline(x=baselines['mean_macro_f1'].max(), color='#95a5a6', linestyle='--',
           alpha=0.7, label=f'Best baseline ({baselines["mean_macro_f1"].max():.3f})')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'task6', 'macro_f1_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# Calculate improvement
best_baseline_f1 = baselines['mean_macro_f1'].max()
best_ml_f1 = ml_models['mean_macro_f1'].max()
improvement = (best_ml_f1 - best_baseline_f1) / best_baseline_f1 * 100
print(f"\\n{'='*60}")
print(f"Best baseline   : {baselines.loc[baselines['mean_macro_f1'].idxmax(), 'baseline']}  (Macro-F1 = {best_baseline_f1:.4f})")
print(f"Best ML model   : {ml_models.loc[ml_models['mean_macro_f1'].idxmax(), 'baseline']}  (Macro-F1 = {best_ml_f1:.4f})")
print(f"Improvement     : +{improvement:.1f}% over best baseline")
print(f"{'='*60}")"""))

# ── 7.3 Per-Class Precision & Recall ────────────────────────────────
cells.append(md("""\
### 7.3 Per-Class Recall Analysis

Per-class recall shows how well each model detects falls, flat periods, and rises. A model that is useful for procurement timing must detect **falls** (buy opportunities) and **rises** (sell signals) reliably."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.3  Per-class recall comparison
# ══════════════════════════════════════════════════════════════════════
recall_data = model_results[['baseline', 'mean_fall_recall', 'mean_flat_recall', 'mean_rise_recall']].copy()
recall_data.columns = ['Model', 'Fall', 'Flat', 'Rise']
recall_data = recall_data.set_index('Model')

fig, ax = plt.subplots(figsize=(12, 6))
recall_data.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='white', width=0.75)
ax.set_ylabel('Mean Recall', fontsize=12)
ax.set_title('Per-Class Recall by Model\\n(Rolling-Origin Cross-Validation)', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
ax.legend(title='Class', fontsize=10, title_fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1.05)
ax.axhline(y=0.333, color='grey', linestyle=':', alpha=0.5, label='Random (33.3%)')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'task6', 'per_class_recall.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\\nKey observations on per-class recall:")
print("  • Naive baselines have 0% recall on most classes (majority/always-flat predict one class only)")
print("  • All ML models achieve >40% recall on all three classes")
print("  • Tree-based models (RF, XGBoost, LightGBM, CatBoost) detect FALLS better (~58-67%)")
print("  • Logistic Regression excels at detecting RISES (~77%) but is weaker on FALLS (~46%)")
print("  • FLAT class is hardest to predict across all models (~35-44%)")"""))

# ── 7.4 Retrain best model on full training set and evaluate on test ──
cells.append(md("""\
### 7.4 Final Model: Hold-Out Test Evaluation

We now **retrain the best model (CatBoost) on the full training set** and evaluate on the held-out Q3 2026 test set — data the model has never seen. This gives us the **unbiased estimate** of real-world performance.

> The Calendar-Ablated CatBoost is used because it matches the full CatBoost on Macro-F1 but has lower variance (std 0.058 vs 0.079) and reduces risk from calendar-based distribution shift in the Q3 test set."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.4  Retrain CatBoost (calendar-ablated) on full train, evaluate on test
# ══════════════════════════════════════════════════════════════════════
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠ CatBoost not installed — skipping retrain, using fold results only")

# Define feature exclusions (matching Experiment 8)
ID_COLS = ['district', 'market_name', 'week_start_date',
           'target_pct_change', 'target_direction', 'split']
CALENDAR_DROP = ['month', 'quarter', 'week_of_year_sin', 'week_of_year_cos']
EXCLUDE = ID_COLS + CALENDAR_DROP

# Numerical and categorical features
NUM_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in ['crop', 'state']]
CAT_FEATURES = ['crop', 'state']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

print(f"Features used: {len(ALL_FEATURES)}")
print(f"  Numerical : {len([f for f in ALL_FEATURES if f not in CAT_FEATURES])}")
print(f"  Categorical: {len(CAT_FEATURES)}")
print(f"  Dropped calendar features: {CALENDAR_DROP}")"""))

cells.append(code("""\
# ── Train and evaluate ───────────────────────────────────────────────
if HAS_CATBOOST:
    X_train = train_df[ALL_FEATURES].copy()
    y_train = train_df['target_direction'].copy()
    X_test  = test_df[ALL_FEATURES].copy()
    y_test  = test_df['target_direction'].copy()
    
    # CatBoost with same hyperparameters as Experiment 6/8
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        cat_features=['crop', 'state'],
        random_seed=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()
    
    # Classification report
    CLASS_ORDER = ['fall', 'flat', 'rise']
    report = classification_report(y_test, y_pred, target_names=CLASS_ORDER, output_dict=True)
    report_df = pd.DataFrame(report).T
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CatBoost (Calendar-Ablated) — Hold-Out Test Results      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_ORDER, digits=4))
    
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Macro-F1  : {test_macro_f1:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
else:
    print("Skipping CatBoost test evaluation — package not available")
    y_pred = None
    y_test = test_df['target_direction'].copy()"""))

# ── 7.5 Confusion Matrix ────────────────────────────────────────────
cells.append(md("""\
### 7.5 Confusion Matrix (Hold-Out Test Set)

The confusion matrix shows exactly where the model makes mistakes — which classes get confused with each other."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.5  Confusion matrix — hold-out test set
# ══════════════════════════════════════════════════════════════════════
if y_pred is not None:
    CLASS_ORDER = ['fall', 'flat', 'rise']
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(
        ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Normalised by row (recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_ORDER).plot(
        ax=axes[1], cmap='Oranges', values_format='.2%')
    axes[1].set_title('Confusion Matrix (Row-Normalised = Recall)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.suptitle('CatBoost (Calendar-Ablated) — Hold-Out Test Set (Q3 2026)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'task6', 'confusion_matrix_test.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\nConfusion matrix interpretation:")
    for i, cls in enumerate(CLASS_ORDER):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"  {cls:5s}: {correct}/{total} correct ({cm_norm[i,i]*100:.1f}% recall)")
        # Show most common misclassification
        mistakes = [(CLASS_ORDER[j], cm[i,j]) for j in range(3) if j != i]
        worst = max(mistakes, key=lambda x: x[1])
        print(f"         Most confused with: {worst[0]} ({worst[1]} times, {worst[1]/total*100:.1f}%)")
else:
    print("Confusion matrix skipped — no test predictions available")"""))

# ── 7.6 Baseline Comparison on Test Set ──────────────────────────────
cells.append(md("""\
### 7.6 Baselines vs. CatBoost on Hold-Out Test Set

We compare the final CatBoost model against the three naive baselines on the same test set."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.6  Baseline predictions on test set
# ══════════════════════════════════════════════════════════════════════
CLASS_ORDER = ['fall', 'flat', 'rise']

# 1. Majority class baseline
majority_class = train_df['target_direction'].mode()[0]
y_baseline_majority = pd.Series([majority_class] * len(test_df), index=test_df.index)

# 2. Always flat baseline
y_baseline_flat = pd.Series(['flat'] * len(test_df), index=test_df.index)

# 3. Directional persistence (same direction as this week)
y_baseline_persist = test_df['pct_change_1w'].apply(
    lambda x: 'rise' if x > 2.5 else ('fall' if x < -2.5 else 'flat')
)

baselines_test = {
    'Majority Class': y_baseline_majority,
    'Always Flat': y_baseline_flat,
    'Directional Persistence': y_baseline_persist,
}

if y_pred is not None:
    baselines_test['CatBoost (Calendar-Ablated)'] = pd.Series(y_pred, index=test_df.index)

comparison_rows = []
for name, preds in baselines_test.items():
    comparison_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Macro-F1': f1_score(y_test, preds, average='macro'),
        'Fall Recall': recall_score(y_test, preds, average=None, labels=CLASS_ORDER)[0],
        'Flat Recall': recall_score(y_test, preds, average=None, labels=CLASS_ORDER)[1],
        'Rise Recall': recall_score(y_test, preds, average=None, labels=CLASS_ORDER)[2],
        'Fall Precision': precision_score(y_test, preds, average=None, labels=CLASS_ORDER, zero_division=0)[0],
        'Flat Precision': precision_score(y_test, preds, average=None, labels=CLASS_ORDER, zero_division=0)[1],
        'Rise Precision': precision_score(y_test, preds, average=None, labels=CLASS_ORDER, zero_division=0)[2],
    })

test_comparison = pd.DataFrame(comparison_rows).round(4)
print("╔══════════════════════════════════════════════════════════════╗")
print("║   Hold-Out Test Set Comparison (Q3 2026 — Monsoon)         ║")
print("╚══════════════════════════════════════════════════════════════╝")
display(test_comparison.style
    .background_gradient(subset=['Macro-F1'], cmap='Greens')
    .set_caption('All models evaluated on the same unseen test set')
)"""))

# ── 7.7 Feature Importance ──────────────────────────────────────────
cells.append(md("""\
### 7.7 Feature Importance

Feature importance from the CatBoost model reveals **which features the model actually relies on** for predictions. This tells us where the predictive signal lives."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.7  Feature importance (CatBoost)
# ══════════════════════════════════════════════════════════════════════
if HAS_CATBOOST and 'model' in dir():
    importances = model.get_feature_importance()
    feat_imp = pd.DataFrame({
        'feature': ALL_FEATURES,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Assign feature groups
    def get_group(feat):
        price_feats = ['price', 'price_lag', 'pct_change', 'roll_mean', 'roll_std',
                       'price_vs_roll', 'price_spread', 'price_vs_state', 'price_vs_national']
        arrival_feats = ['arrivals', 'trading_days']
        weather_feats = ['rainfall', 'rainy', 'heavy_rain', 'temp_', 'heat_flag', 'humidity']
        calendar_feats = ['week_of_quarter', 'next_week', 'quarter', 'month', 'week_of_year']
        
        for pf in price_feats:
            if pf in feat: return 'Price'
        for af in arrival_feats:
            if af in feat: return 'Arrivals'
        for wf in weather_feats:
            if wf in feat: return 'Weather'
        for cf in calendar_feats:
            if cf in feat: return 'Calendar'
        return 'Other'
    
    feat_imp['group'] = feat_imp['feature'].apply(get_group)
    group_colors = {'Price': '#3498db', 'Arrivals': '#e67e22', 'Weather': '#2ecc71',
                    'Calendar': '#9b59b6', 'Other': '#95a5a6'}
    
    # Plot top 20 features
    top_n = 20
    top_feats = feat_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        range(top_n),
        top_feats['importance'].values,
        color=[group_colors.get(g, '#95a5a6') for g in top_feats['group']],
        edgecolor='white',
        height=0.7
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (CatBoost)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features\\nCatBoost (Calendar-Ablated)',
                 fontsize=13, fontweight='bold')
    
    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[g], label=g)
                       for g in ['Price', 'Calendar', 'Weather', 'Arrivals', 'Other']
                       if g in top_feats['group'].values]
    ax.legend(handles=legend_elements, title='Feature Group', fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'task6', 'feature_importance_top20.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Group-level summary
    print("\\nFeature importance by group:")
    group_imp = feat_imp.groupby('group')['importance'].sum().sort_values(ascending=False)
    for grp, imp in group_imp.items():
        pct = imp / feat_imp['importance'].sum() * 100
        print(f"  {grp:10s}  {pct:5.1f}%  (total importance: {imp:.1f})")
    
    print(f"\\nTop 5 individual features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"  {row['feature']:30s}  importance={row['importance']:.1f}  ({row['group']})")
else:
    print("Feature importance analysis requires CatBoost — skipped")"""))

# ── 7.8 Cross-validation fold stability ──────────────────────────────
cells.append(md("""\
### 7.8 Cross-Validation Fold Stability

We examine how stable each model's performance is across the 3 rolling-origin folds. Large variation suggests the model is sensitive to the training window."""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 7.8  Fold-level stability analysis
# ══════════════════════════════════════════════════════════════════════
fold_files = {
    'Logistic Regression': 'logistic_regression_fold_results.csv',
    'Random Forest': 'random_forest_fold_results.csv',
    'XGBoost': 'xgboost_fold_results.csv',
    'LightGBM': 'lightgbm_fold_results.csv',
    'CatBoost': 'catboost_fold_results.csv',
    'CatBoost (Ablated)': 'catboost_calendar_ablation_fold_results.csv',
}

all_folds = []
for model_name, fname in fold_files.items():
    fpath = os.path.join(REPORTS, fname)
    if os.path.exists(fpath):
        fold_df = pd.read_csv(fpath)
        fold_df['model_label'] = model_name
        all_folds.append(fold_df)

if all_folds:
    folds_combined = pd.concat(all_folds, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, (model_name, grp) in enumerate(folds_combined.groupby('model_label')):
        folds = grp['macro_f1'].values
        ax.scatter([model_name]*len(folds), folds, s=60, alpha=0.7, zorder=3)
        ax.plot([model_name, model_name], [folds.min(), folds.max()],
                color='grey', linewidth=2, alpha=0.5, zorder=2)
        ax.scatter([model_name], [folds.mean()], s=120, marker='D',
                   color='red', zorder=4, edgecolors='darkred')
    
    ax.set_ylabel('Macro-F1', fontsize=12)
    ax.set_title('Cross-Validation Fold Stability\\n(Dots = fold values, ♦ = mean)',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'task6', 'fold_stability.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Fold stability (std of Macro-F1):")
    stability = folds_combined.groupby('model_label')['macro_f1'].agg(['mean', 'std'])
    stability = stability.sort_values('mean', ascending=False)
    for model, row in stability.iterrows():
        print(f"  {model:25s}  mean={row['mean']:.4f}  std={row['std']:.4f}")"""))

# ── 7.9 Interpretation summary ──────────────────────────────────────
cells.append(md("""\
### 7.9 Evaluation Summary & Interpretation

| Aspect | Finding |
|---|---|
| **Best model** | CatBoost (with or without calendar ablation), Macro-F1 ≈ 0.569 |
| **Improvement over best baseline** | ~157% over Directional Persistence (0.222) |
| **Accuracy** | ~58% (vs 36% for majority-class baseline) |
| **Strongest class** | **Rise** — best detected (~67-69% recall) |
| **Weakest class** | **Flat** — hardest to predict (~41-43% recall) |
| **Fall detection** | ~58-67% recall — important for procurement timing |
| **Feature importance** | Price-history features dominate; `week_of_quarter` is the top calendar feature |
| **Calendar ablation** | Removing `month`, `quarter`, sin/cos has negligible effect (∆F1 < 0.001), suggesting these features don't add signal beyond `week_of_quarter` |
| **Stability** | CatBoost (ablated) has the lowest std across folds (0.058) |

**Strengths:**
- The model demonstrates a significant performance increase over the baseline frameworks.
- It effectively captures mean-reversion and quarterly cyclical patterns.
- It maintains balanced recall across all three classes (mitigating single-class bias).

**Limitations:**
- With an accuracy of ~58%, the model provides directional guidance but requires supplementary domain expertise for high-stakes decision-making.
- The "flat" class remains the most challenging to predict due to the narrow ±2.5% band.
- The 1-year historical window limits robust seasonal generalisation.
- The test set evaluates exclusively Q3 (monsoon season); performance across other seasons is estimated purely via cross-validation.
- The observed quarterly seasonality presents an unusually consistent pattern; should this reflect data aggregation methodologies rather than market realities, real-time generalizability may be affected."""))

# ===================================================================
# SECTION 8 — Business Findings
# ===================================================================
cells.append(md("""\
---
## 8. Business Findings & Strategic Recommendations

Based on the exploratory data analysis and model evaluation, the following actionable business insights are presented:"""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 8.1  Feature importance → business insight mapping
# ══════════════════════════════════════════════════════════════════════
print("╔══════════════════════════════════════════════════════════════╗")
print("║          Business Findings from Model Analysis             ║")
print("╚══════════════════════════════════════════════════════════════╝")

findings = [
    ("1. Front-Loaded Quarterly Procurement Strategy",
     "Prices consistently begin each quarter ~6% below average and exhibit an upward trend of ~1%/week.\\n"
     "   The `week_of_quarter` feature emerged as a primary predictive factor.\\n"
     "   → Recommendation: Schedule bulk procurement operations during weeks 1-3 of each quarter."),
    
    ("2. Market Mean Reversion Following Volatility",
     "The most significant predictive signal is the 1-week historical price change (ρ ≈ -0.38).\\n"
     "   Substantial price increases are consistently followed by market corrections.\\n"
     "   → Recommendation: Following a confirmed price spike, non-urgent procurement should be deferred."),
    
    ("3. Limited Direct Impact of Meteorological Variables",
     "Weather variables constitute only ~10-15% of total predictive importance within the model.\\n"
     "   Linear correlations with target variables are negligible (|ρ| ≤ 0.06).\\n"
     "   → However, extreme events (heavy rain, heat waves) may trigger non-linear threshold\\n"
     "   effects. Subsequent association rule mining (Review 2) will investigate this further."),
    
    ("4. Negligible Predictive Power of Arrival Volumes",
     "Market arrivals exhibit near-zero correlation with price movements in the current dataset.\\n"
     "   This deviation from fundamental economic theory may indicate upstream data quality issues.\\n"
     "   → Caveat: Supply dynamics should not be completely discounted without multi-year validation."),
    
    ("5. Model Utility as a Supplementary Decision Tool",
     "An accuracy of ~58% and Macro-F1 of 0.57 indicates the model successfully identifies market\\n"
     "   direction in approximately 3 out of 5 instances.\\n"
     "   → Recommendation: Deploy the model as an augmentation to domain expertise, rather than a\\n"
     "   standalone automated trading framework."),
]

for title, detail in findings:
    print(f"\\n  {title}")
    print(f"  {'─'*58}")
    for line in detail.split('\\n'):
        print(f"    {line}")"""))

cells.append(md("""\
### 8.1 Procurement Timing Heuristic

Based on the quantitative analysis, the following heuristic is proposed for procurement optimization:

| Market Condition | Recommended Action | Confidence Level |
|---|---|---|
| Week 1-3 of quarter + model predicts "flat" or "fall" | **Execute procurement** — prices are below average and likely to climb | High |
| Week 10-13 of quarter + model predicts "rise" | **Defer procurement** — prices are elevated; expect correction | Moderate |
| Model predicts "fall" following a spike week | **Delay procurement** — mean reversion highly probable | Moderate |
| Model predicts "rise" during monsoon + heavy rain | **Anticipatory procurement** — potential supply disruption | Low (pending Review 2 validation) |

### 8.2 Limitations & Next Steps for Review 2

1. **Time-series forecasting** (SARIMA with weather regressors) will provide point forecasts, not just directional signals
2. **Association rule mining** will identify specific weather→price-spike trigger rules with support, confidence and lift
3. **Interactive Power BI dashboard** will let stakeholders explore prices, forecasts and rules by crop, mandi and date
4. **Longer data history** would improve seasonal generalisation — the 1-year window is a known limitation"""))

# ===================================================================
# SECTION 9 — Dataset Documentation Appendix
# ===================================================================
cells.append(md("""\
---
## 9. Dataset Documentation Appendix

### 9.1 Complete Feature Dictionary

The modeling dataset `features_panel.csv` contains 46 columns (38 numeric features + 2 categorical + 4 identifiers + 2 targets):"""))

cells.append(code("""\
# ══════════════════════════════════════════════════════════════════════
# 9.1  Feature dictionary
# ══════════════════════════════════════════════════════════════════════
feature_dict = {
    # Price history (14 features)
    'price': 'Current week modal price (₹/quintal)',
    'price_lag_1': 'Modal price 1 week ago',
    'price_lag_2': 'Modal price 2 weeks ago',
    'price_lag_4': 'Modal price 4 weeks ago',
    'pct_change_1w': 'Price % change from last week — strongest predictor (ρ ≈ -0.38 with target)',
    'pct_change_2w': 'Price % change over 2 weeks',
    'pct_change_4w': 'Price % change over 4 weeks',
    'roll_mean_4': '4-week rolling mean of price',
    'roll_mean_8': '8-week rolling mean of price',
    'roll_std_4': '4-week rolling std of weekly % change (recent volatility)',
    'price_vs_roll_mean_4': 'Price / 4-week rolling mean - 1 (%)',
    'price_vs_roll_mean_8': 'Price / 8-week rolling mean - 1 (%)',
    'price_spread_pct': '(max_price - min_price) / modal_price (%)',
    'price_vs_state_mean': 'Price relative to same-crop state average this week',
    'price_vs_national_mean': 'Price relative to same-crop national average this week',
    # Arrivals (5 features)
    'arrivals': 'Total arrivals this week (tonnes)',
    'arrivals_lag_1': 'Arrivals 1 week ago',
    'arrivals_pct_change_1w': 'Arrivals % change from last week',
    'arrivals_roll_mean_4': '4-week rolling mean of arrivals',
    'arrivals_vs_roll_mean_4': 'Arrivals / 4-week rolling mean - 1 (%)',
    # Weather (11 features)
    'rainfall_mm': 'Weekly total precipitation (mm)',
    'rainfall_lag_1': 'Rainfall 1 week ago',
    'rainfall_lag_2': 'Rainfall 2 weeks ago',
    'rainfall_roll_sum_4': 'Total rainfall over past 4 weeks (mm)',
    'rainy_days': 'Days with ≥ 2.5 mm rain (IMD rainy day)',
    'heavy_rain_flag': '1 if any day ≥ 64.5 mm (IMD heavy rain)',
    'temp_mean_c': 'Weekly mean temperature (°C)',
    'temp_max_c': 'Hottest daily max in the week (°C)',
    'heat_flag': '1 if temp_max ≥ 40°C (IMD heat-wave threshold)',
    'temp_anomaly_c': 'Temperature anomaly vs. previous 8-week mean',
    'humidity_mean_pct': 'Weekly mean relative humidity (%)',
    'humidity_roll_mean_4': '4-week rolling mean humidity',
    # Calendar (6 features)
    'week_of_quarter': 'Position 1-13 within calendar quarter',
    'next_week_is_quarter_start': '1 if next week is week 1 of a new quarter',
    'quarter': 'Calendar quarter (1-4)',
    'month': 'Calendar month (1-12)',
    'week_of_year_sin': 'sin(2π × ISO_week / 52) — smooth yearly cycle',
    'week_of_year_cos': 'cos(2π × ISO_week / 52) — smooth yearly cycle',
    # Identifiers
    'crop': 'Tomato / Onion / Potato',
    'state': '14 Indian states',
    'district': 'District of the mandi',
    'market_name': 'Mandi name (70 markets)',
    'week_start_date': 'Monday of the observation week',
    # Targets
    'target_pct_change': 'Next week price % change (continuous)',
    'target_direction': 'Next week direction: fall / flat / rise (±2.5% band)',
    'split': 'Chronological partition: train / test',
}

feat_df = pd.DataFrame([
    {'Feature': k, 'Description': v} for k, v in feature_dict.items()
])
display(feat_df.style.set_caption('Complete Feature Dictionary — features_panel.csv'))"""))

cells.append(md("""\
### 9.2 Leakage Safeguards

| Rule | Implementation |
|---|---|
| All lags/rolling windows use `shift`/`rolling` within mandi×crop series | No future information leaks |
| Cross-sectional features use only same-week data | `price_vs_state_mean`, `price_vs_national_mean` |
| PCA fitted on training data only | Serialised in `models/pca_pipeline.joblib` |
| Calendar features of week t+1 allowed | The calendar is deterministic and known in advance |
| Chronological train/test split | Train: Nov 2025–Jun 2026; Test: Jul–Sep 2026 |
| No random shuffle in cross-validation | Expanding-window rolling-origin folds |

### 9.3 Data Quality Observations

> **Data Integrity Note:** The upstream exploratory data analysis identified that prices across all 70 mandis display a synchronized quarterly sawtooth wave with an unusually low coefficient of variation (~5.6%). Furthermore, intra-mandi prices appear uncorrelated with local arrivals (r ≈ 0). This degree of regularity is anomalous and may reflect upstream data smoothing, fallback imputation methodologies, or synthetic artifacts within the CEDA/Agmarknet source data. Consequently, all strategic business conclusions should be subjected to appropriate domain validation.

---

## End of Review 1 Notebook

**Files submitted:**
- This notebook (`notebooks/07_Task6_Final_Documentation.ipynb`)
- All datasets in `data/`
- Model results in `reports/`
- Figures in `reports/figures/`
- Documentation in `docs/`
- Collection and processing scripts in `scripts/`"""))

# ===================================================================
# Build the notebook JSON
# ===================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.9",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python"
        }
    },
    "cells": cells
}

# ── Fix cell source format: each element must end with \n except last ──
for cell in notebook["cells"]:
    lines = cell["source"]
    if isinstance(lines, str):
        lines = lines.split("\n")
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + "\n" if not line.endswith("\n") else line)
        else:
            fixed.append(line.rstrip("\n"))
    cell["source"] = fixed
    # Add unique cell ID
    import uuid
    cell["id"] = str(uuid.uuid4())[:8]

# ── Write the notebook ───────────────────────────────────────────────
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'notebooks', '07_Task6_Final_Documentation.ipynb'
)
output_path = os.path.abspath(output_path)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook written to: {output_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
