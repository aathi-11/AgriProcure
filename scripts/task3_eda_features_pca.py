"""
Task 3 (KAN-6): EDA, Feature Engineering, Leakage/Multicollinearity Checks, and PCA
======================================================================================

Runs top-to-bottom without Jupyter. Saves:
  data/features_panel.csv      — full 38-feature table with chronological split
  data/features_pca.csv        — keys + targets + PC scores
  models/pca_pipeline.joblib   — fitted StandardScaler + PCA pipeline
  reports/feature_vif.csv      — VIF table
  reports/pca_loadings.csv     — PCA loadings matrix
  reports/figures/task3/       — all diagnostic charts

Usage:
    python scripts/task3_eda_features_pca.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_features import (
    ARRIVAL_FEATURES,
    CALENDAR_FEATURES,
    CATEGORICAL_FEATURES,
    KEYS,
    NUMERIC_FEATURES,
    PRICE_FEATURES,
    TARGETS,
    WEATHER_FEATURES,
    build_features,
)

DATA   = ROOT / "data"
FIG    = ROOT / "reports" / "figures" / "task3"
MODELS = ROOT / "models"
REP    = ROOT / "reports"

FIG.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

# ── plot style ─────────────────────────────────────────────────────────────────
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CROP_COLORS = {"Onion": "#e05c2e", "Potato": "#2a78d6", "Tomato": "#1baf7a"}
DIR_COLORS  = {"fall": "#184f95", "flat": "#eda100", "rise": "#b52e2e"}
GROUP_COLORS = {
    "Price":    "#2a78d6",
    "Arrivals": "#eb6834",
    "Weather":  "#1baf7a",
    "Calendar": "#eda100",
}
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52e2e"]
)
plt.rcParams.update({
    "figure.facecolor":  SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor":  MUTED,
    "axes.labelcolor":   INK2,    "xtick.color":     INK2,
    "ytick.color":       INK2,    "axes.spines.top": False,
    "axes.spines.right": False,   "axes.grid":        True,
    "grid.color":        "#e6e5e1","axes.axisbelow":  True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 110,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})

GROUP = (
    {f: "Price"    for f in PRICE_FEATURES}
    | {f: "Arrivals" for f in ARRIVAL_FEATURES}
    | {f: "Weather"  for f in WEATHER_FEATURES}
    | {f: "Calendar" for f in CALENDAR_FEATURES}
)

BAND_PCT      = 2.5        # ±% flat band
TEST_START    = pd.Timestamp("2026-07-06")
HEAVY_RAIN_MM = 64.5       # IMD heavy-rain threshold
HEAT_WAVE_C   = 40.0       # IMD heat-wave threshold

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0  Load data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 0  Load & prepare data")
print("=" * 70)

panel = pd.read_csv(DATA / "panel_with_weather.csv", parse_dates=["week_start_date"])
print(f"Panel loaded: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
print(f"  Crops   : {sorted(panel['crop'].unique())}")
print(f"  Mandis  : {panel['market_name'].nunique()}")
print(f"  Dates   : {panel['week_start_date'].min().date()} → {panel['week_start_date'].max().date()}")

# Drop the partial final week (< 6 trading days)
full_weeks = panel[~panel["is_partial_week"]].copy()
print(f"  Full weeks kept: {full_weeks['week_start_date'].nunique()} "
      f"({panel['is_partial_week'].sum()} partial-week rows dropped)")

# Build the feature table from all full weeks (warm-up NaNs still present)
feats_all = build_features(full_weeks)
print(f"\nFeature table (before dropping warm-up): {feats_all.shape}")

# Drop warm-up rows (first 8 weeks) and last row per series (no target)
feats = feats_all.dropna(subset=NUMERIC_FEATURES + ["target_pct_change"]).reset_index(drop=True)
feats["split"] = np.where(feats["week_start_date"] >= TEST_START, "test", "train")

n_dropped = len(feats_all) - len(feats)
print(f"  Warm-up / no-target rows dropped : {n_dropped}")
print(f"  Usable rows                      : {len(feats)}")
split_counts = feats.groupby("split")["week_start_date"].agg(["nunique", "size"])
split_counts.columns = ["unique_weeks", "rows"]
print("\nChronological split:")
print(split_counts.to_string())
print("\nTarget class share by split:")
print(pd.crosstab(feats["split"], feats["target_direction"], normalize="index").round(3).to_string())

train = feats[feats["split"] == "train"]
test  = feats[feats["split"] == "test"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  Feature design summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1  Feature design")
print("=" * 70)

group_counts = {g: list(GROUP.values()).count(g) for g in GROUP_COLORS}
print(f"Total numeric features : {len(NUMERIC_FEATURES)}")
for g, n in group_counts.items():
    print(f"  {g:10s}: {n} features")
print(f"Categorical features   : {CATEGORICAL_FEATURES}")
print(f"Keys                   : {KEYS}")
print(f"Targets                : {TARGETS}")
print(f"\nSample descriptive statistics (train set):")
print(train[NUMERIC_FEATURES].describe().T[["mean", "std", "min", "max"]].round(2).to_string())

# Chart: feature group overview bar
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(list(group_counts.keys()), list(group_counts.values()),
               color=[GROUP_COLORS[g] for g in group_counts], height=0.55)
ax.bar_label(bars, fmt="%d", padding=4, color=INK2)
ax.set_xlabel("Number of features")
ax.set_title("Feature groups: count of engineered inputs")
ax.set_xlim(0, max(group_counts.values()) + 3)
fig.savefig(FIG / "0_feature_group_counts.png")
plt.close(fig)
print(f"\nSaved: reports/figures/task3/0_feature_group_counts.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  EDA visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2  EDA visualizations")
print("=" * 70)

# Use the full panel (all mandis, all weeks) for EDA — not just training
eda = full_weeks.copy()
eda["price_index"] = eda.groupby(["crop", "market_name"])["weekly_modal_price"].transform(
    lambda s: s / s.mean() * 100
)
eda["pct_change"] = eda.groupby(["crop", "market_name"])["weekly_modal_price"].pct_change() * 100

# Compute week-of-quarter for EDA charts
thursday = eda["week_start_date"] + pd.Timedelta(days=3)
q_start  = thursday.dt.to_period("Q").dt.start_time
first_th  = q_start + pd.to_timedelta((3 - q_start.dt.dayofweek) % 7, unit="D")
first_wk  = first_th - pd.Timedelta(days=3)
eda["week_of_quarter"] = ((eda["week_start_date"] - first_wk).dt.days // 7 + 1).astype(int)

# ── Chart 1: National average weekly price by crop ───────────────────────────
nat = eda.groupby(["week_start_date", "crop"])["weekly_modal_price"].mean().unstack()

fig, ax = plt.subplots(figsize=(11, 4.5))
for crop, col in CROP_COLORS.items():
    if crop in nat.columns:
        ax.plot(nat.index, nat[crop], color=col, lw=2, label=crop)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax.set_xlabel("Week start date")
ax.set_ylabel("Weekly modal price (₹/q)")
ax.set_title("Chart 1: National average weekly modal price by crop")
ax.legend(frameon=False)
fig.savefig(FIG / "1_national_price_by_crop.png")
plt.close(fig)
print("Saved: Chart 1 — national price by crop")

# ── Chart 2: Seasonality by week-of-quarter ──────────────────────────────────
woq = (eda.groupby(["crop", "week_of_quarter"])["price_index"]
         .agg(["mean", "std"]).reset_index())

fig, ax = plt.subplots(figsize=(10, 4.5))
for crop, col in CROP_COLORS.items():
    d = woq[woq["crop"] == crop]
    ax.plot(d["week_of_quarter"], d["mean"], color=col, lw=2, label=crop)
    ax.fill_between(d["week_of_quarter"],
                    d["mean"] - d["std"], d["mean"] + d["std"],
                    color=col, alpha=0.12)
ax.axhline(100, color=MUTED, lw=1, ls="--", label="Mandi mean = 100")
ax.set_xlabel("Week of quarter")
ax.set_ylabel("Price index (mandi mean = 100)")
ax.set_title("Chart 2: Intra-quarter price seasonality by crop")
ax.legend(frameon=False)
fig.savefig(FIG / "2_seasonality_week_of_quarter.png")
plt.close(fig)
print("Saved: Chart 2 — intra-quarter seasonality")

# ── Chart 3: Price vs arrivals (within-mandi de-meaned) ──────────────────────
scatter_df = eda.copy()
for col in ["weekly_modal_price", "weekly_total_arrivals"]:
    scatter_df[col] = scatter_df.groupby(["crop", "market_name"])[col].transform(
        lambda s: s - s.mean()
    )

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
for ax, (crop, col) in zip(axes, CROP_COLORS.items()):
    d = scatter_df[scatter_df["crop"] == crop].dropna(
        subset=["weekly_modal_price", "weekly_total_arrivals"]
    )
    r = d["weekly_modal_price"].corr(d["weekly_total_arrivals"])
    sample = d.sample(min(len(d), 800), random_state=42)
    ax.scatter(sample["weekly_total_arrivals"], sample["weekly_modal_price"],
               s=8, alpha=0.35, color=col, edgecolors="none")
    ax.set_xlabel("Arrivals deviation (tonnes)")
    ax.set_ylabel("Price deviation (₹/q)")
    ax.set_title(f"{crop}\n(r = {r:+.2f})")
fig.suptitle("Chart 3: Price vs arrivals within each mandi (de-meaned)",
             y=1.01, fontsize=12, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(FIG / "3_price_vs_arrivals.png")
plt.close(fig)
print("Saved: Chart 3 — price vs arrivals")

# ── Chart 4: Next-week price change by rainfall/heat ─────────────────────────
# Compute next-week pct change from the full panel (non-detrended)
eda_sorted = eda.sort_values(["crop", "market_name", "week_start_date"])
eda_sorted["next_pct_change"] = eda_sorted.groupby(
    ["crop", "market_name"])["weekly_modal_price"].pct_change().shift(-1) * 100

rain_bins  = [-0.1, 2.4, 15.5, 64.5, 9999]
rain_labels = ["Dry\n(0–2.4 mm)", "Light rain\n(2.5–15.5)", "Moderate\n(15.5–64.5)", "Heavy\n(>64.5)"]
eda_sorted["rain_cat"] = pd.cut(
    eda_sorted["max_daily_rainfall_mm"], bins=rain_bins, labels=rain_labels
)
eda_sorted["heat_cat"] = np.where(
    eda_sorted["weekly_temp_max_c"] >= HEAT_WAVE_C, "Heat wave\n(≥40°C)", "Normal\n(<40°C)"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
order_rain = rain_labels
rain_med = eda_sorted.groupby("rain_cat", observed=True)["next_pct_change"].median()
ax1.bar(rain_med.index, rain_med.values,
        color=[CROP_COLORS["Onion"], CROP_COLORS["Potato"],
               CROP_COLORS["Tomato"], "#8b008b"],
        width=0.55)
ax1.axhline(0, color=MUTED, lw=1)
ax1.set_xlabel("Rainfall category (max daily mm this week)")
ax1.set_ylabel("Median next-week price change (%)")
ax1.set_title("Chart 4a: Next-week price change\nby rainfall category")

heat_med = eda_sorted.groupby("heat_cat")["next_pct_change"].median()
ax2.bar(heat_med.index, heat_med.values,
        color=[CROP_COLORS["Potato"], CROP_COLORS["Onion"]], width=0.35)
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_xlabel("Temperature category (max weekly temp)")
ax2.set_ylabel("Median next-week price change (%)")
ax2.set_title("Chart 4b: Next-week price change\nby heat category")

fig.tight_layout()
fig.savefig(FIG / "4_price_change_by_weather.png")
plt.close(fig)
print("Saved: Chart 4 — next-week price change by weather")

# ── Chart 5: Distribution of weekly price changes + ±BAND flat band ───────────
fig, ax = plt.subplots(figsize=(10, 4.5))
bins = np.arange(-30, 31, 1)
for crop, col in CROP_COLORS.items():
    vals = eda["pct_change"].dropna()
    vals_crop = eda.loc[eda["crop"] == crop, "pct_change"].dropna()
    ax.hist(vals_crop, bins=bins, color=col, alpha=0.45, density=True, label=crop)
ax.axvline(-BAND_PCT, color=INK2, lw=1.5, ls="--")
ax.axvline(+BAND_PCT, color=INK2, lw=1.5, ls="--",
           label=f"±{BAND_PCT}% flat band")
ax.axvspan(-BAND_PCT, +BAND_PCT, color=INK2, alpha=0.06)
ax.set_xlabel("Weekly price change (%)")
ax.set_ylabel("Density")
ax.set_title("Chart 5: Distribution of weekly price changes")
ax.legend(frameon=False)
fig.savefig(FIG / "5_price_change_distribution.png")
plt.close(fig)
print("Saved: Chart 5 — price change distribution")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  Leakage checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3  Leakage checks")
print("=" * 70)

# 3a. Truncation test ─────────────────────────────────────────────────────────
print("\n3a. Truncation test")
print("    If features for week t use only data ≤ t, rebuilding with data")
print("    cut off at t must give identical values for that week.\n")

def truncation_test(cutoff: pd.Timestamp) -> pd.Series:
    """Return features that differ when built from truncated data (leakage flag)."""
    cut = build_features(full_weeks[full_weeks["week_start_date"] <= cutoff])
    a = (cut[cut["week_start_date"] == cutoff]
         .set_index(["crop", "market_name"])[NUMERIC_FEATURES])
    b = (feats_all[feats_all["week_start_date"] == cutoff]
         .set_index(["crop", "market_name"])[NUMERIC_FEATURES])
    diff = (a - b.loc[a.index]).abs().max()
    return diff[diff > 1e-9]

leakage_found = False
for cutoff_str in ["2026-01-12", "2026-04-20", "2026-08-31"]:
    cutoff = pd.Timestamp(cutoff_str)
    bad = truncation_test(cutoff)
    if bad.empty:
        print(f"    cut-off {cutoff_str}  →  PASS ✓  (no feature uses future data)")
    else:
        leakage_found = True
        print(f"    cut-off {cutoff_str}  →  FAIL ✗  leaking features: {bad.to_dict()}")

if not leakage_found:
    print("\n    ✅  All truncation tests passed — zero data leakage detected.\n")
else:
    print("\n    ⚠️  Data leakage detected — review the flagged features.\n")

# 3b. Feature–target Spearman ρ ───────────────────────────────────────────────
print("3b. Feature–target correlation screen (train set only)")
rho = (train[NUMERIC_FEATURES]
       .corrwith(train["target_pct_change"], method="spearman")
       .sort_values(key=abs))
flagged = list(rho[rho.abs() > 0.8].index)
print(f"    Features with |ρ| > 0.8 (leakage signal): {flagged or 'none'}")
print("\n    Top-10 features by |Spearman ρ| with next-week price change:")
print(rho.sort_values(key=abs, ascending=False).head(10).round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(rho.index, rho.values,
        color=[GROUP_COLORS[GROUP[f]] for f in rho.index], height=0.7)
ax.axvline(0, color=MUTED, lw=1)
ax.axvline(0.8, color="#b52e2e", lw=1, ls="--", alpha=0.6)
ax.axvline(-0.8, color="#b52e2e", lw=1, ls="--", alpha=0.6, label="|ρ| > 0.8 threshold")
for f, v in rho.items():
    if abs(v) >= 0.10:
        ax.annotate(f"{v:+.2f}", (v, f),
                    xytext=(4 if v > 0 else -4, 0),
                    textcoords="offset points",
                    ha="left" if v > 0 else "right",
                    va="center", fontsize=8, color=INK2)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
ax.legend(handles, GROUP_COLORS.keys(), frameon=False, loc="lower right")
ax.set_xlabel("Spearman ρ with next week's % price change (train)")
ax.set_title("Feature–target correlation (Spearman ρ) by group")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "6_feature_target_correlation.png")
plt.close(fig)
print("Saved: Chart 6 — feature–target correlation")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  Multicollinearity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4  Multicollinearity")
print("=" * 70)

# 4a. Correlation heatmap ─────────────────────────────────────────────────────
corr = train[NUMERIC_FEATURES].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7.5)
ax.set_yticks(range(len(corr)), corr.index, fontsize=7.5)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.65, label="Pearson r (train)")
ax.set_title("Feature correlation matrix (Pearson r, train set)")
fig.savefig(FIG / "7_feature_correlation_matrix.png")
plt.close(fig)
print("Saved: Chart 7 — correlation matrix")

# High-correlation pairs (|r| > 0.9)
pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1))
             .stack().rename("r").reset_index())
pairs.columns = ["feature_1", "feature_2", "r"]
high = pairs[pairs["r"].abs() > 0.9].sort_values("r", key=abs, ascending=False)
print(f"\n  {len(high)} feature pairs with |r| > 0.9:")
print(high.round(3).to_string(index=False))

# 4b. Variance Inflation Factor ───────────────────────────────────────────────
print("\n4b. Variance Inflation Factor (VIF)")
X  = train[NUMERIC_FEATURES]
Xs = (X - X.mean()) / X.std()
Xs = Xs.assign(const=1.0)
vif = pd.Series(
    [variance_inflation_factor(Xs.values, i) for i in range(len(NUMERIC_FEATURES))],
    index=NUMERIC_FEATURES, name="VIF"
).sort_values(ascending=False)
vif_table = vif.to_frame().assign(
    group=lambda d: d.index.map(GROUP),
    flag=lambda d: np.where(d["VIF"] > 10, "VIF > 10", "")
)
vif_table.round(1).to_csv(REP / "feature_vif.csv")
n_vif_high = (vif > 10).sum()
print(f"  {n_vif_high} of {len(vif)} features have VIF > 10")
print("  Top-15 by VIF:")
print(vif_table.head(15).round(1).to_string())
print("  Saved: reports/feature_vif.csv")

# VIF bar chart (top-20)
fig, ax = plt.subplots(figsize=(8, 7))
top_vif = vif_table.head(20)
ax.barh(top_vif.index[::-1], top_vif["VIF"][::-1],
        color=[GROUP_COLORS[g] for g in top_vif["group"][::-1]], height=0.65)
ax.axvline(10, color="#b52e2e", lw=1.5, ls="--", label="VIF = 10 threshold")
ax.set_xlabel("Variance Inflation Factor")
ax.set_title("Multicollinearity: VIF of top-20 features (train set)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
ax.legend(handles, GROUP_COLORS.keys(), frameon=False, loc="lower right")
fig.savefig(FIG / "8_vif_chart.png")
plt.close(fig)
print("Saved: Chart 8 — VIF chart")

print(f"\n  → Price-level features (price, lags, rolling means) are nearly")
print(f"    perfectly collinear (|r| up to ~1.0, VIF up to ~10,000).")
print(f"  → Motivation for PCA: decorrelate all {len(NUMERIC_FEATURES)} features.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  PCA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5  Principal Component Analysis")
print("=" * 70)

# 5a. Fit scaler + PCA on training data only ─────────────────────────────────
full_pipe = make_pipeline(StandardScaler(), PCA(random_state=42))
full_pipe.fit(train[NUMERIC_FEATURES])
pca_full = full_pipe.named_steps["pca"]
ev       = pca_full.explained_variance_ratio_
cum      = np.cumsum(ev)

# Choose N by explained-variance thresholds
thresholds = {thresh: int(np.searchsorted(cum, thresh) + 1) for thresh in [0.80, 0.85, 0.90, 0.95]}
N_COMPONENTS = thresholds[0.90]

kaiser_n = int((pca_full.explained_variance_ > 1).sum())

print(f"\n  Explained variance thresholds:")
for t, n in thresholds.items():
    print(f"    {t:.0%} → {n} components")
print(f"  Kaiser rule (eigenvalue > 1) → {kaiser_n} components")
print(f"\n  ✅  Choice: {N_COMPONENTS} components (90% threshold)")

# 5b. Scree plot ──────────────────────────────────────────────────────────────
ORANGE = "#eb6834"
x = np.arange(1, len(ev) + 1)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x, ev * 100, color=CROP_COLORS["Potato"], alpha=0.6, label="Per component")
ax.plot(x, cum * 100, color=ORANGE, lw=2, marker="o", ms=3, label="Cumulative")
ax.axhline(90, color=MUTED, lw=1, ls="--")
ax.axvline(N_COMPONENTS + 0.5, color=MUTED, lw=1, ls=":")
ax.annotate(
    f"{N_COMPONENTS} PCs → {cum[N_COMPONENTS - 1]:.0%} variance",
    (N_COMPONENTS + 0.7, 60), color=INK2, fontsize=9
)
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax.set_title("Chart 9: Scree plot — variance explained by each component (train)")
ax.legend(frameon=False)
fig.savefig(FIG / "9_pca_scree.png")
plt.close(fig)
print("Saved: Chart 9 — scree plot")

# 5c. Fit final PCA with N_COMPONENTS ────────────────────────────────────────
pipe = make_pipeline(
    StandardScaler(), PCA(n_components=N_COMPONENTS, random_state=42)
).fit(train[NUMERIC_FEATURES])
pca     = pipe.named_steps["pca"]
pc_names = [f"PC{i + 1}" for i in range(N_COMPONENTS)]
loadings = pd.DataFrame(
    pca.components_.T, index=NUMERIC_FEATURES, columns=pc_names
)
ev_kept = pca.explained_variance_ratio_

# 5d. Loadings heatmap ────────────────────────────────────────────────────────
SHOW = min(8, N_COMPONENTS)
fig, ax = plt.subplots(figsize=(9, 11))
im = ax.imshow(loadings.iloc[:, :SHOW].values, cmap=DIV, vmin=-0.7, vmax=0.7, aspect="auto")
ax.set_yticks(range(len(loadings)), loadings.index, fontsize=8)
ax.set_xticks(
    range(SHOW),
    [f"{c}\n{ev_kept[i]:.0%}" for i, c in enumerate(pc_names[:SHOW])]
)
ax.grid(False)
for (i, j), v in np.ndenumerate(loadings.iloc[:, :SHOW].values):
    if abs(v) >= 0.25:
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                color="#ffffff" if abs(v) > 0.45 else INK)
fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
ax.set_title(f"Chart 10: Loadings of the first {SHOW} components (|loading| ≥ 0.25 labelled)")
fig.savefig(FIG / "10_pca_loadings_heatmap.png")
plt.close(fig)
print("Saved: Chart 10 — loadings heatmap")

print("\n  Component interpretation (top-5 loadings each):")
for c in pc_names[:SHOW]:
    top = loadings[c].sort_values(key=abs, ascending=False).head(5)
    print(f"    {c}: " + ", ".join(f"{f} ({v:+.2f})" for f, v in top.items()))

# 5e. Transform train + test; verify decorrelation ───────────────────────────
Z_train = pd.DataFrame(
    pipe.transform(train[NUMERIC_FEATURES]), columns=pc_names, index=train.index
)
Z_test = pd.DataFrame(
    pipe.transform(test[NUMERIC_FEATURES]), columns=pc_names, index=test.index
)

off = lambda m: np.abs(m.values[~np.eye(len(m), dtype=bool)]).max()
print(f"\n  Max |corr| between PCs — train: {off(Z_train.corr()):.3f}  "
      f"(uncorrelated by construction)")
print(f"  Max |corr| between PCs — test : {off(Z_test.corr()):.3f}  "
      f"(distribution shift in monsoon Q3 expected)")

# 5f. PC–target correlation (train) ──────────────────────────────────────────
rho_pc = Z_train.corrwith(train["target_pct_change"], method="spearman")
print("\n  Spearman ρ of each PC with next-week price change (train):")
print(rho_pc.round(3).to_string())

# 5g. Scatter of top-2 PCs coloured by target direction ──────────────────────
top2 = rho_pc.abs().sort_values(ascending=False).index[:2]
sample = Z_train.join(train["target_direction"]).sample(min(3000, len(Z_train)), random_state=42)

fig, ax = plt.subplots(figsize=(8, 6))
for d, col in DIR_COLORS.items():
    s = sample[sample["target_direction"] == d]
    ax.scatter(s[top2[0]], s[top2[1]], s=10, alpha=0.45, color=col,
               label=d, edgecolors="none")
ax.set_xlabel(f"{top2[0]} (ρ with target = {rho_pc[top2[0]]:+.2f})")
ax.set_ylabel(f"{top2[1]} (ρ with target = {rho_pc[top2[1]]:+.2f})")
ax.legend(title="Next week", frameon=False)
ax.set_title("Chart 11: Training weeks on the two most predictive PCs")
fig.savefig(FIG / "11_pca_top2_components_by_direction.png")
plt.close(fig)
print("Saved: Chart 11 — top-2 PC scatter by direction")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6  Save outputs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6  Save outputs")
print("=" * 70)

# Full feature table
out = feats[KEYS + NUMERIC_FEATURES + TARGETS + ["split"]]
out.to_csv(DATA / "features_panel.csv", index=False)
print(f"  Saved data/features_panel.csv         {out.shape}")

# PCA feature table (keys + targets + PC scores)
pca_table = pd.concat(
    [feats[KEYS + TARGETS + ["split"]],
     pd.concat([Z_train, Z_test]).loc[feats.index].round(6)],
    axis=1,
)
pca_table.to_csv(DATA / "features_pca.csv", index=False)
print(f"  Saved data/features_pca.csv           {pca_table.shape}")

# PCA pipeline (StandardScaler + PCA)
joblib.dump(pipe, MODELS / "pca_pipeline.joblib")
print(f"  Saved models/pca_pipeline.joblib")

# PCA loadings matrix
loadings.round(4).to_csv(REP / "pca_loadings.csv")
print(f"  Saved reports/pca_loadings.csv")

# VIF table (already saved above, confirm)
print(f"  Saved reports/feature_vif.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3 SUMMARY")
print("=" * 70)
print(f"""
  Feature table   : data/features_panel.csv
                    {out.shape[0]:,} rows × {out.shape[1]} columns
                    Train: {len(train):,} rows ({train['week_start_date'].nunique()} weeks)
                    Test : {len(test):,} rows ({test['week_start_date'].nunique()} weeks)

  Feature groups  : {len(NUMERIC_FEATURES)} numeric features
                    Price    {group_counts['Price']:>2} (lags 1/2/4, pct-change 1/2/4w,
                              rolling mean 4/8, std 4, deviation features,
                              state/national mean spread, price spread)
                    Arrivals {group_counts['Arrivals']:>2} (level, lag-1, 1w pct-change,
                              rolling mean 4, deviation from mean)
                    Weather  {group_counts['Weather']:>2} (rainfall mm, lags 1/2, 4w sum,
                              rainy days, heavy-rain flag, temp mean/max,
                              heat flag, temp anomaly, humidity mean/4w)
                    Calendar {group_counts['Calendar']:>2} (week-of-quarter, quarter-start flag,
                              quarter, month, week sin/cos)

  Leakage check   : {'PASSED ✓ — no feature uses future data' if not leakage_found else 'FAILED ✗ — check flagged features'}
  Target corr     : max |ρ| = {rho.abs().max():.2f} (threshold 0.80) — {"PASS ✓" if rho.abs().max() < 0.8 else "FAIL ✗"}
  VIF > 10        : {n_vif_high} of {len(vif)} features (price-level features are collinear)

  PCA             : {N_COMPONENTS} components keep {cum[N_COMPONENTS - 1]:.0%} of training variance
                    Kaiser rule would keep {kaiser_n} components
  PC table        : data/features_pca.csv  {pca_table.shape}
  Figures         : reports/figures/task3/  (11 charts)
""")
