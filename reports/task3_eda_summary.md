# Task 3 (KAN-6): EDA and Visualization, Summary

Notebook: [`notebooks/03_EDA_and_Visualization.ipynb`](../notebooks/03_EDA_and_Visualization.ipynb)
Charts: [`reports/figures/task3/`](figures/task3/)

## Data used

| File | Rows | Source |
|---|---|---|
| `data/cleaned_mandi_crop_week_panel.csv` | 11,130 (70 mandis × 3 crops × 53 weeks) | CEDA / Agmarknet (Task 2) |
| `data/weather_daily_by_mandi.csv` | 25,830 (70 mandis × 369 days) | Open-Meteo archive API |
| `data/weather_weekly_by_mandi.csv` | 3,710 (70 mandis × 53 weeks) | Aggregated from daily, Monday-start weeks |
| `data/panel_with_weather.csv` | 11,130 | Price panel joined to weekly weather |

Weather is collected by `scripts/collect_weather_openmeteo.py`, using the coordinates of each mandi's district town. The final week (starting 2026-09-28) has only 2 trading days and is flagged `is_partial_week`. It is excluded from the EDA.

### Weather data dictionary (weekly)

| Column | Meaning |
|---|---|
| `weekly_rainfall_mm` | Total precipitation over the week (mm) |
| `rainy_days` | Days with ≥ 2.5 mm rain (IMD "rainy day") |
| `max_daily_rainfall_mm` | Rainfall on the wettest day of the week (mm) |
| `weekly_temp_mean_c` | Mean of daily mean 2 m temperature (°C) |
| `weekly_temp_max_c` / `weekly_temp_min_c` | Highest daily max / lowest daily min in the week (°C) |
| `weekly_humidity_mean_pct` | Mean of daily mean relative humidity (%) |
| `weather_days` | Number of days of weather data in the week (7 for full weeks) |

## Charts and takeaways

| # | Chart | Takeaway |
|---|---|---|
| 1 | `chart1_national_price_trend.png` | Tomato (≈ ₹2,380/q) > onion (≈ ₹1,995/q) > potato (≈ ₹1,550/q). All three move together in a quarterly sawtooth with no yearly trend. |
| 2 | `chart2_week_of_quarter_seasonality.png` | Prices start each quarter about 6% below average, climb about 1% a week to about 5% above average by weeks 12-13, then drop about 10%, so buy early in the quarter. |
| 3 | `chart3_state_week_price_heatmap.png` | The cycle is synchronised across all 14 states; the national weekly level explains about 28-35% of price variance. |
| 4 | `chart4_price_vs_arrivals.png` | Within a mandi, heavier-arrival weeks do not show lower prices (r ≈ 0). |
| 5 | `chart5_next_week_change_by_weather.png` | Rain and temperature categories do not consistently shift next week's price change. |
| 6 | `chart6_lagged_correlation_heatmap.png` | Weather and arrivals at lags 0-4 weeks have \|ρ\| ≤ 0.06 with next week's move; own-price mean reversion (ρ ≈ -0.38) is the main signal. |
| 7 | `chart7_volatility_state_crop.png` | Weekly volatility is about 5.5-7% everywhere; tomato in Maharashtra/Gujarat is the most volatile, and Pune tomato is the most volatile mandi (7.9%). |
| 8 | `chart8_weekly_change_distribution.png` | A ±2.5% flat band gives balanced fall / flat / rise classes (about 30 / 33 / 37%). |

## Implications for Task 4

- Prioritise price-history features (lags, % change, rolling mean/std, distance from rolling mean) and calendar features (week of quarter, quarter, month).
- Keep weather and arrival features, using lags and rolling windows, but expect weak signal.
- Use ±2.5% as the price-direction band.

## Data observations to raise with the team

- The price series is unusually regular: the same quarterly sawtooth appears in every mandi and crop, the within-mandi coefficient of variation is only about 5.6%, and prices do not respond to arrivals. Please confirm the CEDA / Agmarknet extraction returned genuine per-market history before drawing business conclusions.
- History covers 1 year (Sep 2025 to Sep 2026), below the 3-5 years in the README scope, so yearly seasonality cannot be confirmed.
