# 4.1 (KAN-9): Feature List and Rationale

Parent task: **Task 3 (KAN-6): EDA and Visualization**. This document turns the EDA findings into a feature list for the next-week price-direction model. The features are built in subtask 4.2 (KAN-10) and reduced with PCA in subtask 4.3 (KAN-11).

## 1. Unit of analysis and target

- **Row:** one mandi × crop × week `t` (Monday-start week) from `data/panel_with_weather.csv`.
- **Prediction time:** end of week `t`. Every feature uses only information available by then.
- **Target (regression form):** `target_pct_change = (modal_price[t+1] / modal_price[t] - 1) × 100`.
- **Target (classification form, Review 1):** `target_direction` = `fall` if the change is < -2.5%, `rise` if > +2.5%, otherwise `flat`.
  - **Why ±2.5%:** the EDA (Chart 8) shows this band gives roughly balanced classes (about 30% fall, 33% flat, 37% rise). A ±2.5% move on a ₹2,000/q tomato price is ₹50/q, which is material for a processor buying hundreds of quintals.
- **Excluded rows:** the partial final week (2 trading days), and the first 8 weeks of each series, which are needed to fill the longest rolling window.

## 2. Feature list

Notation: `P` = weekly modal price, `A` = weekly total arrivals, `g` = the mandi × crop series. All lags and rolling windows are computed **within `g`**, in date order.

### 2a. Price-history features

| Feature | Definition | Rationale / EDA evidence |
|---|---|---|
| `price` | `P[t]` | Current level; the base the next change is measured from. |
| `price_lag_1`, `price_lag_2`, `price_lag_4` | `P[t-1]`, `P[t-2]`, `P[t-4]` | Short price memory. Price is highly persistent (lag-1 r ≈ 0.94). |
| `pct_change_1w` | `P[t]/P[t-1] - 1` (%) | **Strongest single signal in EDA** (Chart 6): ρ ≈ -0.38 with next week's change, i.e. mean reversion. |
| `pct_change_2w`, `pct_change_4w` | `P[t]/P[t-2] - 1`, `P[t]/P[t-4] - 1` (%) | Momentum over longer horizons; captures where the series is in its climb. |
| `roll_mean_4`, `roll_mean_8` | Mean of `P` over weeks `t-3…t` and `t-7…t` | Local price level, a smoothed reference. |
| `roll_std_4` | Std of weekly % change over `t-3…t` | Recent volatility. Volatility differs by mandi (Chart 7: 4.4-7.9%). |
| `price_vs_roll_mean_4`, `price_vs_roll_mean_8` | `P[t] / roll_mean_k - 1` (%) | Distance from the recent average. Large positive gaps tend to revert. |
| `price_spread_pct` | `(max_price - min_price) / P[t]` (%) | Within-week dispersion: a sign of uncertain or mixed-quality arrivals. |
| `price_vs_state_mean` | `P[t]` / mean `P` of the same crop in the same state that week - 1 (%) | Whether this mandi is dear or cheap relative to its neighbours (Chart 3). |
| `price_vs_national_mean` | `P[t]` / national mean `P` of the crop that week - 1 (%) | Same idea at national level; the national cycle explains about 30% of variance. |

### 2b. Arrivals (supply) features

| Feature | Definition | Rationale / EDA evidence |
|---|---|---|
| `arrivals` | `A[t]` (tonnes) | Supply this week. Economic theory says high supply lowers price. |
| `arrivals_lag_1` | `A[t-1]` | Supply effects can reach price with a delay. |
| `arrivals_pct_change_1w` | `A[t]/A[t-1] - 1` (%) | A sudden supply drop is the classic spike trigger. |
| `arrivals_roll_mean_4` | Mean of `A` over `t-3…t` | Normal recent supply. |
| `arrivals_vs_roll_mean_4` | `A[t] / arrivals_roll_mean_4 - 1` (%) | Supply shock relative to normal. |
| `trading_days_active` | Days with trades in week `t` | Market disruptions (holidays, strikes) reduce trading days. *Built but not modelled: it is constant (6) in every full week of the current data (found in 4.2).* |

EDA note: arrivals showed **near-zero correlation** with price in this panel (Charts 4 and 6). They are kept because they are central to the business question and to the association-rule work in Review 2, but they are expected to add little to the model.

### 2c. Weather features

| Feature | Definition | Rationale / EDA evidence |
|---|---|---|
| `rainfall_mm` | Weekly total precipitation at the mandi | Rain disrupts harvesting and transport. |
| `rainfall_lag_1`, `rainfall_lag_2` | Rainfall in `t-1`, `t-2` | Weather affects supply with a delay (crop damage, then lower arrivals). |
| `rainfall_roll_sum_4` | Total rainfall over `t-3…t` | Wet spells matter more than single wet days. |
| `rainy_days` | Days with ≥ 2.5 mm (IMD rainy day) | Persistence of rain within the week. |
| `heavy_rain_flag` | 1 if any day ≥ 64.5 mm (IMD "heavy") | Extreme event indicator for spike rules. |
| `temp_mean_c` | Weekly mean temperature | Heat stresses perishable crops (especially tomato). |
| `temp_max_c` | Hottest daily max in the week | Peak heat. |
| `heat_flag` | 1 if `temp_max_c` ≥ 40 °C (IMD heat-wave threshold for plains) | Extreme heat indicator. |
| `temp_anomaly_c` | `temp_mean_c` minus the mandi's mean temperature over the previous 8 weeks (`t-8…t-1`) | Sudden heat or cold relative to recent conditions, rather than normal seasonal temperature. A trailing window is used because 1 year of data cannot give reliable month-of-year averages. |
| `humidity_mean_pct` | Weekly mean relative humidity | High humidity speeds spoilage (onion, potato storage). |
| `humidity_roll_mean_4` | Mean humidity over `t-3…t` | Sustained humid spells. |

EDA note: weather correlations with next week's price move were small (|ρ| ≤ 0.06 at lags 0-4). The flags and anomalies are included because tree models can pick up threshold effects that a correlation would miss.

### 2d. Calendar features

| Feature | Definition | Rationale / EDA evidence |
|---|---|---|
| `week_of_quarter` | 1-13, position of week `t` in its calendar quarter (quarter assigned by the week's Thursday) | **Dominant pattern in EDA** (Chart 2): prices climb about 1% a week through a quarter. |
| `next_week_is_quarter_start` | 1 if week `t+1` is week 1 of a quarter | The about 10% drop happens at the quarter start. The calendar is known in advance, so this is **not leakage**. |
| `quarter` | 1-4 | Quarter-level differences. |
| `month` | 1-12 (of week `t`'s Thursday) | Monthly seasonality (harvest and lean seasons). |
| `week_of_year_sin`, `week_of_year_cos` | sin/cos of 2π × ISO week / 52 | Smooth yearly seasonality without a jump between week 52 and week 1. |

### 2e. Identifiers (used as categorical inputs, not in PCA)

| Feature | Definition | Rationale |
|---|---|---|
| `crop` | Tomato / Onion / Potato | Price levels differ by crop (Chart 1). |
| `state` | 14 states | Regional differences in level and volatility (Chart 7). |
| `market_name` | 70 mandis | Kept for grouping and splitting, not as a model input (too many levels for 1 year of data). |

## 3. Leakage rules

1. Every lag and rolling window uses `shift`/`rolling` **within the mandi × crop series**, ending at week `t`. Nothing from week `t+1` or later is used except the target.
2. Cross-sectional features (`price_vs_state_mean`, `price_vs_national_mean`) use only week `t` prices.
3. `temp_anomaly_c` uses only the mandi's **previous** 8 weeks of temperature, so it needs no statistics from the test period.
4. Calendar features of `t+1` are allowed because the calendar is known in advance.
5. Model evaluation must use a **chronological split** (train on earlier weeks, test on later weeks), never a random split.

## 4. Multicollinearity plan

Many price features are near-duplicates (`price`, `price_lag_1`, `roll_mean_4`, `roll_mean_8` all track the level). Subtask 4.2 checks:

- the correlation matrix, flagging pairs with |r| > 0.9, and
- the variance inflation factor (VIF), flagging VIF > 10.

Subtask 4.3 then applies **PCA** to the standardised numeric features to obtain uncorrelated components. Logistic regression can use the PCs; tree models can use the raw features, since they are not affected by collinearity.
