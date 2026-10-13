"""
AgriProcure - Task 11: Power BI Star Schema Data Preparation & Export
-----------------------------------------------------------------------
Exports fully normalized and structured dimension and fact tables for Power BI:
1. dim_date.csv: Calendar dimension with hierarchical date attributes.
2. dim_crop.csv: Agronomic profiles, perishability indices, and storage attributes.
3. dim_mandi.csv: Geographic hierarchy, coordinates, and market tiers.
4. fact_mandi_weekly_prices.csv: Complete weekly market observations with weather.
5. fact_sarima_forecasts.csv: Backtest forecast vs actuals across horizons.
6. fact_sarima_future_projections.csv: 4-week forward projections with 80% & 95% CIs.
7. fact_association_rules.csv: Mined early-warning rules with trigger metrics.
8. mandi_kpis_summary.csv: Pre-aggregated executive KPIs by mandi and crop.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PBI_DIR = os.path.join(DATA_DIR, "powerbi_exports")
os.makedirs(PBI_DIR, exist_ok=True)

def generate_dim_date(start_date="2025-09-01", end_date="2026-11-30"):
    """Creates a standard dimension date table with weekly aggregation keys."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"Date": dates})
    
    df["Date_Key"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["Quarter_Year"] = df["Quarter"] + " " + df["Year"].astype(str)
    df["Month_Number"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Month_Year"] = df["Date"].dt.strftime("%b %Y")
    df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)
    # Week start date (Monday)
    df["Week_Start_Date"] = df["Date"].apply(lambda d: d - pd.Timedelta(days=d.weekday()))
    df["Week_Start_Key"] = df["Week_Start_Date"].dt.strftime("%Y-%m-%d")
    df["Is_Quarter_Start_Week"] = df["Week_Start_Date"].dt.month.isin([1, 4, 7, 10]) & (df["Week_Start_Date"].dt.day <= 7)
    
    # Save unique weekly date rows as well
    df_weekly = df.drop_duplicates(subset=["Week_Start_Key"]).copy()
    df.to_csv(os.path.join(PBI_DIR, "dim_date.csv"), index=False)
    df_weekly.to_csv(os.path.join(PBI_DIR, "dim_date_weekly.csv"), index=False)
    print(f"dim_date.csv generated: {len(df)} daily rows, {len(df_weekly)} weekly entries.")
    return df

def generate_dim_crop():
    """Generates agronomic and commercial commodity metadata."""
    crops_data = [
        {
            "Crop_ID": 1,
            "Crop": "Tomato",
            "Scientific_Name": "Solanum lycopersicum",
            "Category": "Perishable Vegetable",
            "Perishability_Level": "High",
            "Shelf_Life_Days": 5,
            "Storage_Type": "Ambient / Short Cold Chain",
            "Moisture_Sensitivity": "Extremely High (Fungal rot / Cracking)",
            "Heat_Sensitivity": "High (Sunscald / Premature softening)",
            "Target_Procurement_Band_Pct": 0.05,
            "Volatility_Category": "Severe"
        },
        {
            "Crop_ID": 2,
            "Crop": "Onion",
            "Scientific_Name": "Allium cepa",
            "Category": "Semi-Perishable Vegetable",
            "Perishability_Level": "Medium",
            "Shelf_Life_Days": 35,
            "Storage_Type": "Ventilated Storage / Chawl",
            "Moisture_Sensitivity": "High (Sprouting / Basal rot)",
            "Heat_Sensitivity": "Medium (Desiccation)",
            "Target_Procurement_Band_Pct": 0.05,
            "Volatility_Category": "High"
        },
        {
            "Crop_ID": 3,
            "Crop": "Potato",
            "Scientific_Name": "Solanum tuberosum",
            "Category": "Semi-Perishable Tuber",
            "Perishability_Level": "Low",
            "Shelf_Life_Days": 90,
            "Storage_Type": "Commercial Cold Storage (2-4 C)",
            "Moisture_Sensitivity": "Medium (Blight / Tuber rot)",
            "Heat_Sensitivity": "Low (Under cold storage)",
            "Target_Procurement_Band_Pct": 0.03,
            "Volatility_Category": "Moderate"
        }
    ]
    df = pd.DataFrame(crops_data)
    df.to_csv(os.path.join(PBI_DIR, "dim_crop.csv"), index=False)
    print("dim_crop.csv generated.")
    return df

def generate_dim_mandi(raw_df):
    """Extracts geographic coordinates and state metadata for all mandis."""
    # State-level centroids & regions for mapping
    state_meta = {
        "Maharashtra": {"Region": "West", "Lat": 19.7515, "Lon": 75.7139},
        "Gujarat": {"Region": "West", "Lat": 22.2587, "Lon": 71.1924},
        "Rajasthan": {"Region": "West", "Lat": 27.0238, "Lon": 74.2179},
        "Karnataka": {"Region": "South", "Lat": 15.3173, "Lon": 75.7139},
        "Tamil Nadu": {"Region": "South", "Lat": 11.1271, "Lon": 78.6569},
        "Andhra Pradesh": {"Region": "South", "Lat": 15.9129, "Lon": 79.7400},
        "Telangana": {"Region": "South", "Lat": 18.1124, "Lon": 79.0193},
        "Uttar Pradesh": {"Region": "North", "Lat": 26.8467, "Lon": 80.9462},
        "Haryana": {"Region": "North", "Lat": 29.0588, "Lon": 76.0856},
        "Punjab": {"Region": "North", "Lat": 31.1471, "Lon": 75.3412},
        "Bihar": {"Region": "East", "Lat": 25.0961, "Lon": 85.3131},
        "West Bengal": {"Region": "East", "Lat": 22.9868, "Lon": 87.8550},
        "Odisha": {"Region": "East", "Lat": 20.9517, "Lon": 85.0985},
        "Madhya Pradesh": {"Region": "Central", "Lat": 22.9734, "Lon": 78.6569}
    }
    
    unique_mandis = raw_df[["market_name", "district", "state"]].drop_duplicates().reset_index(drop=True)
    records = []
    
    for idx, row in unique_mandis.iterrows():
        st = row["state"]
        meta = state_meta.get(st, {"Region": "Other", "Lat": 20.5937, "Lon": 78.9629})
        # Add slight jitter for visualization if multiple mandis share state centroid
        jitter_lat = meta["Lat"] + (idx % 7 - 3) * 0.15
        jitter_lon = meta["Lon"] + ((idx // 7) % 7 - 3) * 0.15
        
        tier = "Primary APMC Hub" if any(h in row["market_name"].lower() for h in ["pune", "agra", "ahmedabad", "bangalore", "mumbai", "delhi", "amritsar", "jaipur"]) else "Secondary APMC Mandi"
        
        records.append({
            "Mandi_ID": idx + 1,
            "Market_Name": row["market_name"],
            "District": row["district"],
            "State": st,
            "Region": meta["Region"],
            "Mandi_Tier": tier,
            "Latitude": round(jitter_lat, 4),
            "Longitude": round(jitter_lon, 4)
        })
        
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(PBI_DIR, "dim_mandi.csv"), index=False)
    print(f"dim_mandi.csv generated: {len(df)} mandis.")
    return df

def generate_fact_tables(raw_df):
    """Exports fact tables matching the dimensional keys."""
    # 1. Weekly mandi fact table
    fact_prices = raw_df[[
        "market_name", "crop", "week_start_date", "weekly_modal_price",
        "weekly_min_price", "weekly_max_price", "weekly_total_arrivals",
        "weekly_avg_daily_arrivals", "trading_days_active", "weekly_rainfall_mm",
        "rainy_days", "max_daily_rainfall_mm", "weekly_temp_mean_c",
        "weekly_temp_max_c", "weekly_temp_min_c", "weekly_humidity_mean_pct"
    ]].copy()
    
    fact_prices.rename(columns={
        "market_name": "Market_Name",
        "crop": "Crop",
        "week_start_date": "Week_Start_Date",
        "weekly_modal_price": "Modal_Price",
        "weekly_min_price": "Min_Price",
        "weekly_max_price": "Max_Price",
        "weekly_total_arrivals": "Total_Arrivals_MT",
        "weekly_avg_daily_arrivals": "Avg_Daily_Arrivals_MT",
        "trading_days_active": "Trading_Days",
        "weekly_rainfall_mm": "Rainfall_mm",
        "rainy_days": "Rainy_Days",
        "max_daily_rainfall_mm": "Max_Daily_Rainfall_mm",
        "weekly_temp_mean_c": "Temp_Mean_C",
        "weekly_temp_max_c": "Temp_Max_C",
        "weekly_temp_min_c": "Temp_Min_C",
        "weekly_humidity_mean_pct": "Humidity_Pct"
    }, inplace=True)
    
    fact_prices.to_csv(os.path.join(PBI_DIR, "fact_mandi_weekly_prices.csv"), index=False)
    print(f"fact_mandi_weekly_prices.csv generated: {len(fact_prices)} rows.")
    
    # 2. Executive Mandi KPIs Summary
    kpi_records = []
    for (mandi, crop), grp in fact_prices.groupby(["Market_Name", "Crop"]):
        state = raw_df[raw_df["market_name"] == mandi]["state"].iloc[0]
        prices = grp["Modal_Price"].values
        arrivals = grp["Total_Arrivals_MT"].values
        mean_p = float(np.mean(prices))
        std_p = float(np.std(prices))
        volatility = (std_p / mean_p) * 100.0 if mean_p > 0 else 0.0
        
        kpi_records.append({
            "Market_Name": mandi,
            "State": state,
            "Crop": crop,
            "Annual_Mean_Price": round(mean_p, 2),
            "Price_Standard_Deviation": round(std_p, 2),
            "Price_Volatility_CV_Pct": round(volatility, 2),
            "Min_Historical_Price": round(float(np.min(prices)), 2),
            "Max_Historical_Price": round(float(np.max(prices)), 2),
            "Mean_Weekly_Arrivals_MT": round(float(np.mean(arrivals)), 2),
            "Total_Annual_Arrivals_MT": round(float(np.sum(arrivals)), 2),
            "Mean_Weekly_Rainfall_mm": round(float(grp["Rainfall_mm"].mean()), 2),
            "Mean_Weekly_Temp_C": round(float(grp["Temp_Mean_C"].mean()), 2),
            "Peak_Spread_Pct": round(float(((grp["Max_Price"] - grp["Min_Price"]) / grp["Modal_Price"]).mean() * 100.0), 2)
        })
        
    kpi_df = pd.DataFrame(kpi_records)
    kpi_df.to_csv(os.path.join(PBI_DIR, "mandi_kpis_summary.csv"), index=False)
    print(f"mandi_kpis_summary.csv generated: {len(kpi_df)} pairs.")
    
    return fact_prices, kpi_df

def main():
    print("=" * 60)
    print("AgriProcure: Generating Power BI Star Schema Data Exports")
    print("=" * 60)
    
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "panel_with_weather.csv"))
    
    generate_dim_date()
    generate_dim_crop()
    generate_dim_mandi(raw_df)
    generate_fact_tables(raw_df)
    
    # Check that forecasting and rules files are mirrored into powerbi_exports
    for f in ["sarima_forecasts_vs_actuals.csv", "sarima_future_forecasts.csv", "association_rules.csv"]:
        src = os.path.join(DATA_DIR, f)
        dst = os.path.join(PBI_DIR, f)
        if os.path.exists(src):
            pd.read_csv(src).to_csv(dst, index=False)
            print(f"Mirrored {f} to powerbi_exports.")
            
    print("\nPower BI Star Schema Preparation Complete! All tables saved in data/powerbi_exports/.")

if __name__ == "__main__":
    main()
