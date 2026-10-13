# AgriProcure — Power BI Interactive Dashboard Architecture & Implementation Guide

**Project:** From Field to Forecast: A Predictive and Weather-Linked Analytics Framework for Indian Mandi Commodity Prices  
**Module:** Unit 3 Advanced Analytics — Task 11 (KAN-30)  
**Deliverable Version:** Production v2.0  
**Data Directory:** `data/powerbi_exports/`  
**Standalone Interactive Web Companion:** [`reports/dashboards/interactive_agriprocure_dashboard.html`](../dashboards/interactive_agriprocure_dashboard.html)

---

## 1. Overview & Business Purpose

The **AgriProcure Power BI Dashboard** provides supply-chain managers, food processors, and Farmer Producer Organizations (FPOs) with an interactive, evidence-backed decision cockpit. It unites:
1. **Historical Market & Weather Dynamics:** 53-week panel across 70 major mandis in 14 states for Tomato, Onion, and Potato.
2. **SARIMAX Time-Series Forecasts:** Multi-step forward price trajectories ($h = 1$ to $4$ weeks) with 80% and 95% confidence intervals.
3. **Mined Early-Warning Association Rules:** Plain-language weather-and-arrival trigger cards that predict price spikes before they occur.
4. **Procurement Arbitrage & Sourcing Drill-Down:** Inter-mandi price spread analysis to identify low-cost procurement origins.

---

## 2. Power BI Data Model Architecture (Star Schema)

The dashboard is structured on a strict **Star Schema** to ensure optimal performance, sub-second cross-filtering, and DAX measure efficiency.

```
       +--------------------+          +--------------------+
       |      dim_crop      |          |      dim_mandi     |
       |--------------------|          |--------------------|
       | PK: Crop           |          | PK: Market_Name    |
       | Category           |          | State              |
       | Perishability      |          | District           |
       | Storage_Type       |          | Region             |
       +---------+----------+          | Lat / Lon          |
                 |                     +---------+----------+
                 | 1                             | 1
                 |                               |
                 | *                             | *
       +---------+-------------------------------+----------+
       |               fact_mandi_weekly_prices              |
       |----------------------------------------------------|
       | FK: Market_Name                                    |
       | FK: Crop                                           |
       | FK: Week_Start_Date                                |
       | Modal_Price, Min_Price, Max_Price                  |
       | Total_Arrivals_MT, Avg_Daily_Arrivals_MT           |
       | Rainfall_mm, Rainy_Days, Temp_Mean_C, Temp_Max_C   |
       +-------------------------+--------------------------+
                                 | *
                                 |
                                 | 1
                       +---------+----------+
                       |      dim_date      |
                       |--------------------|
                       | PK: Week_Start_Date|
                       | Year, Quarter      |
                       | Month_Name         |
                       | Week_Number        |
                       +--------------------+
```

### Supporting Fact Tables:
- **`fact_sarima_forecasts`**: Connected to `dim_crop` (Crop) and `dim_date` (Week_Start_Date). Contains rolling backtest actuals vs predictions across models and horizons.
- **`fact_sarima_future_projections`**: Connected to `dim_crop` (Crop) and `dim_date` (Week_Start_Date). Contains 4-week forward projections with 80% and 95% confidence intervals.
- **`fact_association_rules`**: Connected to `dim_crop` (Crop). Contains mined trigger rules, support, confidence, lift, and plain-English actionable recommendations.

---

## 3. Step-by-Step Data Import & Model Setup

1. **Open Power BI Desktop** -> Click **Get Data** -> **Folder** (or **Text/CSV**).
2. Set folder path to `data/powerbi_exports/`.
3. Load the following 7 tables:
   - `dim_date_weekly.csv` (Rename to **dim_date**)
   - `dim_crop.csv` (Rename to **dim_crop**)
   - `dim_mandi.csv` (Rename to **dim_mandi**)
   - `fact_mandi_weekly_prices.csv` (Rename to **fact_prices**)
   - `fact_sarima_forecasts.csv` (Rename to **fact_forecasts**)
   - `fact_sarima_future_projections.csv` (Rename to **fact_future_projections**)
   - `fact_association_rules.csv` (Rename to **fact_rules**)
4. **Establish Relationships in the Model View:**
   - `dim_crop[Crop]` $1 \to *$ `fact_prices[Crop]` (Single cross-filter)
   - `dim_mandi[Market_Name]` $1 \to *$ `fact_prices[Market_Name]` (Single cross-filter)
   - `dim_date[Week_Start_Key]` $1 \to *$ `fact_prices[Week_Start_Date]` (Single cross-filter)
   - `dim_crop[Crop]` $1 \to *$ `fact_forecasts[Crop]` (Single cross-filter)
   - `dim_crop[Crop]` $1 \to *$ `fact_future_projections[Crop]` (Single cross-filter)
   - `dim_crop[Crop]` $1 \to *$ `fact_rules[Crop]` (Single cross-filter)
5. **Mark Date Table:** Select `dim_date` -> Table Tools -> **Mark as Date Table** -> Select `Date`.

---

## 4. Production DAX Measure Library

Create a dedicated table named `_Measures` and paste these exact formulas:

### 1. Market Pricing Measures
```dax
Avg_Modal_Price = 
AVERAGE(fact_prices[Modal_Price])
```

```dax
Price_StdDev = 
STDEV.P(fact_prices[Modal_Price])
```

```dax
Price_Volatility_CV_Pct = 
DIVIDE([Price_StdDev], [Avg_Modal_Price], 0) * 100
```

```dax
Price_Spread_Pct = 
AVERAGEX(
    fact_prices,
    DIVIDE(fact_prices[Max_Price] - fact_prices[Min_Price], fact_prices[Modal_Price], 0) * 100
)
```

### 2. Supply & Weather Measures
```dax
Total_Arrivals_Volume_MT = 
SUM(fact_prices[Total_Arrivals_MT])
```

```dax
Avg_Weekly_Rainfall_mm = 
AVERAGE(fact_prices[Rainfall_mm])
```

```dax
Heavy_Rain_Weeks_Count = 
CALCULATE(
    COUNTROWS(fact_prices),
    fact_prices[Rainfall_mm] > 35
)
```

```dax
Avg_Mean_Temperature_C = 
AVERAGE(fact_prices[Temp_Mean_C])
```

### 3. Forecast Accuracy & Tracking Measures
```dax
Forecast_Price = 
AVERAGE(fact_forecasts[Predicted])
```

```dax
Forecast_MAE = 
AVERAGE(fact_forecasts[Absolute_Error])
```

```dax
Forecast_MAPE_Pct = 
AVERAGE(fact_forecasts[Percent_Error])
```

```dax
Future_Price_Projection = 
AVERAGE(fact_future_projections[Forecast_Price])
```

```dax
Future_Upper_80_Bound = 
AVERAGE(fact_future_projections[Upper_80])
```

```dax
Future_Lower_80_Bound = 
AVERAGE(fact_future_projections[Lower_80])
```

### 4. Early-Warning & Business Logic Measures
```dax
Active_High_Lift_Rules_Count = 
CALCULATE(
    COUNTROWS(fact_rules),
    fact_rules[Lift] >= 1.25,
    fact_rules[Confidence] >= 0.65
)
```

```dax
Spike_Risk_Status = 
VAR RecentRain = [Avg_Weekly_Rainfall_mm]
VAR RecentArrivals = [Total_Arrivals_Volume_MT]
RETURN
    IF(RecentRain > 35, "CRITICAL: High Rain Spike Triggered",
    IF(RecentRain > 20, "WARNING: Elevated Rain Advisory", "NORMAL: Stable Procurement Window"))
```

```dax
Procurement_Recommendation_Text = 
VAR TopRuleAction = SELECTEDVALUE(fact_rules[Recommended_Action], "Monitor standard spot procurement.")
RETURN TopRuleAction
```

---

## 5. Dashboard Page Layout & Visual Specifications

### Page 1: Executive KPI & Volatility Cockpit
* **Top Slicers:** Commodity (`dim_crop[Crop]`), State (`dim_mandi[State]`), Date Range (`dim_date[Year]`, `Quarter`).
* **KPI Cards Row:**
  1. Average Modal Price (₹/qtl)
  2. Weekly Arrival Volume (MT)
  3. Price Volatility Index (CV %)
  4. Active Early-Warning Rules Count
* **Visual 1 (Line Chart):** Weekly Price Trend vs Prior Year by Crop.
* **Visual 2 (Map Visual):** Geographic Mandi Distribution colored by Price Level, sized by Total Arrivals.
* **Visual 3 (Matrix Visual):** Mandi-level Volatility Rankings (Market Name, State, Mean Price, Volatility %, Peak Spread %).

### Page 2: Price Forecasting & Scenario Explorer (SARIMAX)
* **Top Slicers:** Crop Selector, Forecast Horizon ($1, 2, 3, 4$ Weeks), Model Comparison (`SARIMA_Univariate` vs `SARIMAX_Weather` vs `Seasonal_Naive`).
* **Visual 1 (Line & Shaded Area Chart):** Historical Price Line + 4-Week Forward Forecast + 80% Confidence Interval Band (Lower to Upper).
* **Visual 2 (Clustered Column Chart):** Model Performance Benchmarking (MAPE & MASE across horizons).
* **Visual 3 (KPI Gauges):** Out-of-sample MAPE target indicator ($< 5.0\%$ Green, $5-10\%$ Amber, $> 10\%$ Red).

### Page 3: Weather-Linked Early Warning Matrix
* **Top Slicers:** Crop, Rainfall Threshold (>20mm, >35mm), Alert Type (Price Spike vs Price Crash).
* **Visual 1 (Dual-Axis Chart):** Weekly Rainfall (Column) & Temperature (Line) vs Modal Price Overlay.
* **Visual 2 (Scatter Chart):** Mined Association Rules — X-axis: Support, Y-axis: Confidence, Size: Lift, Color: Consequent Event.
* **Visual 3 (Table / Card Grid):** Filtered Active Early Warning Rules:
  - `Rule_ID`
  - `Trigger_Condition` (WHEN ...)
  - `Target_Outcome` (THEN ...)
  - `Confidence` (%) | `Lift` (x)
  - `Recommended_Action` (Immediate operational directive).

### Page 4: Mandi Drill-Through & Arbitrage Explorer
* **Target:** Activated by right-clicking any Mandi from Page 1 or Page 2.
* **Visual 1 (Line Chart):** Mandi Price vs State Average vs National Average.
* **Visual 2 (Scatter Plot):** Price vs Arrivals Elasticity Curve for the specific mandi.
* **Visual 3 (Savings Calculator):** Sourcing Cost Arbitrage Calculator (Compares selected mandi to state average to estimate bulk procurement savings).

---

## 6. Verification & Companion Files

- **Live Standalone Interactive Dashboard:** Open [`reports/dashboards/interactive_agriprocure_dashboard.html`](../dashboards/interactive_agriprocure_dashboard.html) in any modern browser for immediate interactive exploration.
- **Export Data Location:** `data/powerbi_exports/` contains all verified, schema-aligned CSV tables ready for Power BI Desktop.
