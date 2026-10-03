"""
Feature engineering for the next-week price-direction model.

Implements the feature list in docs/feature_list_and_rationale.md
(subtask 4.1, KAN-9). Used by notebooks/04_Feature_Engineering.ipynb.

Every lag / rolling feature is computed within one mandi x crop series and
ends at week t, so a row only uses information available at the end of
week t. The only forward-looking columns are the targets and the calendar
flag for week t+1 (the calendar is known in advance).
"""

import numpy as np
import pandas as pd

SERIES = ["crop", "market_name"]
BAND_PCT = 2.5
HEAVY_RAIN_MM = 64.5  # IMD "heavy rain" (24 h)
HEAT_WAVE_C = 40.0  # IMD heat-wave threshold for the plains

PRICE_FEATURES = [
    "price", "price_lag_1", "price_lag_2", "price_lag_4",
    "pct_change_1w", "pct_change_2w", "pct_change_4w",
    "roll_mean_4", "roll_mean_8", "roll_std_4",
    "price_vs_roll_mean_4", "price_vs_roll_mean_8",
    "price_spread_pct", "price_vs_state_mean", "price_vs_national_mean",
]
ARRIVAL_FEATURES = [
    "arrivals", "arrivals_lag_1", "arrivals_pct_change_1w",
    "arrivals_roll_mean_4", "arrivals_vs_roll_mean_4",
]
# trading_days_active is built but not modelled: it is constant (6) once the
# partial final week is excluded, so it carries no information.
WEATHER_FEATURES = [
    "rainfall_mm", "rainfall_lag_1", "rainfall_lag_2", "rainfall_roll_sum_4",
    "rainy_days", "heavy_rain_flag", "temp_mean_c", "temp_max_c", "heat_flag",
    "temp_anomaly_c", "humidity_mean_pct", "humidity_roll_mean_4",
]
CALENDAR_FEATURES = [
    "week_of_quarter", "next_week_is_quarter_start", "quarter", "month",
    "week_of_year_sin", "week_of_year_cos",
]
NUMERIC_FEATURES = PRICE_FEATURES + ARRIVAL_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES
CATEGORICAL_FEATURES = ["crop", "state"]
KEYS = ["crop", "state", "district", "market_name", "week_start_date"]
TARGETS = ["target_pct_change", "target_direction"]


def _calendar(week_start):
    """Calendar position of Monday-start weeks; quarter/month follow the week's Thursday."""
    thursday = week_start + pd.Timedelta(days=3)
    quarter_start = thursday.dt.to_period("Q").dt.start_time
    # first Monday-start week whose Thursday falls in the quarter
    first_thursday = quarter_start + pd.to_timedelta((3 - quarter_start.dt.dayofweek) % 7, unit="D")
    first_week = first_thursday - pd.Timedelta(days=3)
    week_of_quarter = ((week_start - first_week).dt.days // 7 + 1).astype(int)
    iso_week = thursday.dt.isocalendar().week.astype(int)
    return pd.DataFrame({
        "week_of_quarter": week_of_quarter,
        "quarter": thursday.dt.quarter,
        "month": thursday.dt.month,
        "week_of_year_sin": np.sin(2 * np.pi * iso_week / 52),
        "week_of_year_cos": np.cos(2 * np.pi * iso_week / 52),
    }, index=week_start.index)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table from data/panel_with_weather.csv (full weeks only)."""
    df = panel.sort_values(SERIES + ["week_start_date"]).reset_index(drop=True).copy()
    g = df.groupby(SERIES, observed=True)
    P, A = "weekly_modal_price", "weekly_total_arrivals"

    out = df[KEYS].copy()

    # --- price history
    out["price"] = df[P]
    for k in (1, 2, 4):
        out[f"price_lag_{k}"] = g[P].shift(k)
    for k in (1, 2, 4):
        out[f"pct_change_{k}w"] = (df[P] / g[P].shift(k) - 1) * 100
    out["roll_mean_4"] = g[P].transform(lambda s: s.rolling(4).mean())
    out["roll_mean_8"] = g[P].transform(lambda s: s.rolling(8).mean())
    out["roll_std_4"] = out.groupby(SERIES, observed=True)["pct_change_1w"].transform(
        lambda s: s.rolling(4).std())
    out["price_vs_roll_mean_4"] = (df[P] / out["roll_mean_4"] - 1) * 100
    out["price_vs_roll_mean_8"] = (df[P] / out["roll_mean_8"] - 1) * 100
    out["price_spread_pct"] = (df["weekly_max_price"] - df["weekly_min_price"]) / df[P] * 100
    state_mean = df.groupby(["crop", "state", "week_start_date"], observed=True)[P].transform("mean")
    nat_mean = df.groupby(["crop", "week_start_date"], observed=True)[P].transform("mean")
    out["price_vs_state_mean"] = (df[P] / state_mean - 1) * 100
    out["price_vs_national_mean"] = (df[P] / nat_mean - 1) * 100

    # --- arrivals
    out["arrivals"] = df[A]
    out["arrivals_lag_1"] = g[A].shift(1)
    out["arrivals_pct_change_1w"] = (df[A] / g[A].shift(1) - 1) * 100
    out["arrivals_roll_mean_4"] = g[A].transform(lambda s: s.rolling(4).mean())
    out["arrivals_vs_roll_mean_4"] = (df[A] / out["arrivals_roll_mean_4"] - 1) * 100
    out["trading_days_active"] = df["trading_days_active"]

    # --- weather
    R, T, H = "weekly_rainfall_mm", "weekly_temp_mean_c", "weekly_humidity_mean_pct"
    out["rainfall_mm"] = df[R]
    out["rainfall_lag_1"] = g[R].shift(1)
    out["rainfall_lag_2"] = g[R].shift(2)
    out["rainfall_roll_sum_4"] = g[R].transform(lambda s: s.rolling(4).sum())
    out["rainy_days"] = df["rainy_days"]
    out["heavy_rain_flag"] = (df["max_daily_rainfall_mm"] >= HEAVY_RAIN_MM).astype(int)
    out["temp_mean_c"] = df[T]
    out["temp_max_c"] = df["weekly_temp_max_c"]
    out["heat_flag"] = (df["weekly_temp_max_c"] >= HEAT_WAVE_C).astype(int)
    out["temp_anomaly_c"] = df[T] - g[T].transform(lambda s: s.shift(1).rolling(8).mean())
    out["humidity_mean_pct"] = df[H]
    out["humidity_roll_mean_4"] = g[H].transform(lambda s: s.rolling(4).mean())

    # --- calendar (week t, plus the known position of week t+1)
    cal = _calendar(df["week_start_date"])
    next_cal = _calendar(df["week_start_date"] + pd.Timedelta(days=7))
    out["week_of_quarter"] = cal["week_of_quarter"]
    out["next_week_is_quarter_start"] = (next_cal["week_of_quarter"] == 1).astype(int)
    for c in ["quarter", "month", "week_of_year_sin", "week_of_year_cos"]:
        out[c] = cal[c]

    # --- targets (week t+1)
    out["target_pct_change"] = (g[P].shift(-1) / df[P] - 1) * 100
    out["target_direction"] = pd.cut(
        out["target_pct_change"], [-np.inf, -BAND_PCT, BAND_PCT, np.inf],
        labels=["fall", "flat", "rise"],
    ).astype("object")

    return out
