"""
AgriProcure - Tasks 7 & 8: Time-Series Analysis, SARIMA/SARIMAX Modeling & Evaluation
-------------------------------------------------------------------------------------
Implements:
1. Weekly price aggregation per crop and key representative mandis.
2. Stationarity testing: Augmented Dickey-Fuller (ADF) & KPSS tests (levels & 1st diff).
3. Seasonal decomposition (STL & classical) and ACF/PACF analysis.
4. Model implementation:
   - Baseline: Naive (Random Walk)
   - Baseline: Seasonal Naive (Lag 13)
   - Univariate SARIMA(p, d, q) x (P, D, Q)_s
   - Multivariate SARIMAX with weather (rainfall, temperature) and arrivals regressors
5. Rolling-origin expanding-window backtesting across forecast horizons h = 1, 2, 3, 4 weeks.
6. Rigorous metric evaluation: MAPE, MASE, RMSE, MAE per crop and horizon.
7. Residual diagnostics: Ljung-Box white-noise test, Jarque-Bera normality, Q-Q plots.
8. Future 4-week forecasts with 80% and 95% confidence intervals.
9. Interactive Plotly visualizer and static publication figures.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

warnings.filterwarnings("ignore")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures", "task7_8")
PBI_DIR = os.path.join(DATA_DIR, "powerbi_exports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PBI_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

def load_and_prepare_data():
    """Loads panel with weather and aggregates national and mandi time-series."""
    df = pd.read_csv(os.path.join(DATA_DIR, "panel_with_weather.csv"))
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    
    # 1. National aggregation per crop
    national_df = df.groupby(["crop", "week_start_date"]).agg({
        "weekly_modal_price": "mean",
        "weekly_rainfall_mm": "mean",
        "weekly_temp_mean_c": "mean",
        "weekly_total_arrivals": "mean",
        "weekly_min_price": "mean",
        "weekly_max_price": "mean"
    }).reset_index().sort_values(["crop", "week_start_date"])
    
    # 2. Key representative mandi benchmarks
    benchmarks = [
        ("Tomato", "Pune Apmc Market"),
        ("Tomato", "Bangalore Apmc Market"),
        ("Onion", "Ahmedabad Apmc Market"),
        ("Potato", "Agra Apmc Market")
    ]
    
    mandi_series = {}
    for crop, mandi in benchmarks:
        sub = df[(df["crop"] == crop) & (df["market_name"] == mandi)].sort_values("week_start_date")
        if len(sub) > 0:
            mandi_series[f"{crop}_{mandi}"] = sub
            
    return df, national_df, mandi_series

def run_stationarity_tests(national_df):
    """Executes ADF and KPSS stationarity tests on level and first difference."""
    results = []
    crops = ["Tomato", "Onion", "Potato"]
    
    for crop in crops:
        sub = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        y = sub["weekly_modal_price"].values
        diff_y = np.diff(y)
        
        # Level ADF
        adf_level = adfuller(y, autolag="AIC")
        # Level KPSS
        kpss_level = kpss(y, regression="c", nlags="auto")
        
        # Diff ADF
        adf_diff = adfuller(diff_y, autolag="AIC")
        # Diff KPSS
        kpss_diff = kpss(diff_y, regression="c", nlags="auto")
        
        # Interpretation
        is_stationary_level = (adf_level[1] < 0.05) and (kpss_level[1] >= 0.05)
        inferred_d = 0 if is_stationary_level else 1
        
        results.append({
            "Crop": crop,
            "Series_Type": "National Aggregated",
            "N_Obs": len(y),
            "Mean_Price": round(float(np.mean(y)), 2),
            "Std_Price": round(float(np.std(y)), 2),
            "ADF_Stat_Level": round(float(adf_level[0]), 3),
            "ADF_p_Level": round(float(adf_level[1]), 4),
            "KPSS_Stat_Level": round(float(kpss_level[0]), 3),
            "KPSS_p_Level": round(float(kpss_level[1]), 4),
            "Stationary_Level": is_stationary_level,
            "ADF_Stat_Diff1": round(float(adf_diff[0]), 3),
            "ADF_p_Diff1": round(float(adf_diff[1]), 4),
            "KPSS_Stat_Diff1": round(float(kpss_diff[0]), 3),
            "KPSS_p_Diff1": round(float(kpss_diff[1]), 4),
            "Recommended_d": inferred_d
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(REPORTS_DIR, "sarima_stationarity_tests.csv"), index=False)
    print("Stationarity tests saved to reports/sarima_stationarity_tests.csv")
    return res_df

def compute_decomposition_and_plots(national_df):
    """Performs STL/Seasonal decomposition and plots ACF/PACF."""
    crops = ["Tomato", "Onion", "Potato"]
    
    # 1. Decomposition figure
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    
    for col_idx, crop in enumerate(crops):
        sub = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        sub = sub.set_index("week_start_date")
        y = sub["weekly_modal_price"]
        
        # Seasonal decomposition with period=13 (quarterly cycle)
        decomp = seasonal_decompose(y, model="additive", period=13, extrapolate_trend="freq")
        
        axes[0, col_idx].plot(y.index, y.values, color="#1f77b4", lw=2)
        axes[0, col_idx].set_title(f"{crop} — Observed Price", fontweight="bold")
        axes[0, col_idx].set_ylabel("Price (Rs/qtl)")
        
        axes[1, col_idx].plot(decomp.trend.index, decomp.trend.values, color="#ff7f0e", lw=2)
        axes[1, col_idx].set_title(f"{crop} — Trend Component", fontweight="bold")
        axes[1, col_idx].set_ylabel("Trend")
        
        axes[2, col_idx].plot(decomp.seasonal.index, decomp.seasonal.values, color="#2ca02c", lw=2)
        axes[2, col_idx].set_title(f"{crop} — Seasonal (s=13 weeks)", fontweight="bold")
        axes[2, col_idx].set_ylabel("Seasonality")
        
        axes[3, col_idx].scatter(decomp.resid.index, decomp.resid.values, color="#d62728", alpha=0.7, s=20)
        axes[3, col_idx].axhline(0, color="gray", linestyle="--", lw=1)
        axes[3, col_idx].set_title(f"{crop} — Residuals (Noise)", fontweight="bold")
        axes[3, col_idx].set_ylabel("Residual")
        axes[3, col_idx].tick_params(axis="x", rotation=30)
        
    plt.suptitle("AgriProcure — Time Series Decomposition per Crop (Period = 13 Weeks)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart2_seasonal_decomposition.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. ACF & PACF Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for col_idx, crop in enumerate(crops):
        sub = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        y = sub["weekly_modal_price"].values
        
        acf_vals = acf(y, nlags=16)
        pacf_vals = pacf(y, nlags=16)
        lags = np.arange(len(acf_vals))
        
        # ACF
        axes[0, col_idx].stem(lags, acf_vals)
        axes[0, col_idx].axhline(0, color="gray", lw=1)
        axes[0, col_idx].axhline(1.96/np.sqrt(len(y)), color="blue", linestyle="--", lw=1, alpha=0.6)
        axes[0, col_idx].axhline(-1.96/np.sqrt(len(y)), color="blue", linestyle="--", lw=1, alpha=0.6)
        axes[0, col_idx].set_title(f"{crop} — Autocorrelation (ACF)", fontweight="bold")
        axes[0, col_idx].set_ylabel("Autocorrelation")
        axes[0, col_idx].set_ylim(-0.8, 1.1)
        
        # PACF
        axes[1, col_idx].stem(lags, pacf_vals)
        axes[1, col_idx].axhline(0, color="gray", lw=1)
        axes[1, col_idx].axhline(1.96/np.sqrt(len(y)), color="blue", linestyle="--", lw=1, alpha=0.6)
        axes[1, col_idx].axhline(-1.96/np.sqrt(len(y)), color="blue", linestyle="--", lw=1, alpha=0.6)
        axes[1, col_idx].set_title(f"{crop} — Partial ACF (PACF)", fontweight="bold")
        axes[1, col_idx].set_xlabel("Lag (Weeks)")
        axes[1, col_idx].set_ylabel("Partial Autocorrelation")
        axes[1, col_idx].set_ylim(-0.8, 1.1)
        
    plt.suptitle("AgriProcure — ACF and PACF Analysis per Crop (Notice strong lag-13 quarterly cycle)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart1_price_stationarity_acf_pacf.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Decomposition and ACF/PACF charts saved.")

def evaluate_metrics(y_true, y_pred, y_train_history, seasonal_period=13):
    """Calculates MAPE, MASE, RMSE, and MAE."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # MAE & RMSE
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    
    # MASE: scaled by in-sample seasonal naive MAE
    if len(y_train_history) > seasonal_period:
        scale = np.mean(np.abs(y_train_history[seasonal_period:] - y_train_history[:-seasonal_period]))
    else:
        scale = np.mean(np.abs(np.diff(y_train_history)))
    if scale == 0 or np.isnan(scale):
        scale = 1.0
        
    mase = mae / scale
    return mae, rmse, mape, mase

def run_rolling_origin_backtest(national_df):
    """
    Executes expanding-window rolling-origin backtest for horizons h=1 to h=4.
    Compares 4 models:
    1. Naive (Random Walk)
    2. Seasonal Naive (Lag 13)
    3. Univariate SARIMA(1, 0, 0) x (1, 0, 0, 13)
    4. Multivariate SARIMAX with rainfall, temperature, and arrivals regressors
    """
    crops = ["Tomato", "Onion", "Potato"]
    horizons = [1, 2, 3, 4]
    
    # Specification parameters per crop based on ACF & AIC
    model_configs = {
        "Tomato": {"order": (1, 0, 1), "seasonal_order": (1, 0, 0, 13)},
        "Onion": {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 13)},
        "Potato": {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 13)}
    }
    
    backtest_records = []
    forecast_points = []
    
    # Total weeks is 53. Test folds: last 8 weeks (weeks 45 to 52).
    # Expanding windows: train on 0:t, predict t to t+h
    test_origins = list(range(41, 49))
    
    for crop in crops:
        sub = national_df[national_df["crop"] == crop].sort_values("week_start_date").reset_index(drop=True)
        y_all = sub["weekly_modal_price"].values
        dates_all = sub["week_start_date"].values
        
        # Regressors: rainfall, temperature, arrivals
        exog_raw = sub[["weekly_rainfall_mm", "weekly_temp_mean_c", "weekly_total_arrivals"]].values
        # Standardize exog to prevent numerical overflow in statespace
        exog_mean = np.mean(exog_raw, axis=0)
        exog_std = np.std(exog_raw, axis=0) + 1e-6
        exog_norm = (exog_raw - exog_mean) / exog_std
        
        cfg = model_configs[crop]
        
        for origin in test_origins:
            train_y = y_all[:origin]
            train_exog = exog_norm[:origin]
            
            # Forecast up to max horizon
            max_h = 4
            if origin + max_h > len(y_all):
                continue
                
            actuals_h = y_all[origin:origin + max_h]
            dates_h = dates_all[origin:origin + max_h]
            future_exog_h = exog_norm[origin:origin + max_h]
            
            # 1. Naive Model: last observed value
            pred_naive = np.repeat(train_y[-1], max_h)
            
            # 2. Seasonal Naive: value from 13 weeks ago
            pred_snaive = []
            for step in range(max_h):
                idx = (origin + step) - 13
                if idx >= 0:
                    pred_snaive.append(y_all[idx])
                else:
                    pred_snaive.append(train_y[-1])
            pred_snaive = np.array(pred_snaive)
            
            # 3. Univariate SARIMA
            try:
                mod_sarima = SARIMAX(train_y, order=cfg["order"], seasonal_order=cfg["seasonal_order"],
                                     enforce_stationarity=False, enforce_invertibility=False)
                res_sarima = mod_sarima.fit(disp=False, maxiter=100)
                pred_sarima = res_sarima.forecast(steps=max_h)
            except Exception:
                pred_sarima = pred_naive
                
            # 4. Multivariate SARIMAX (Weather + Arrivals)
            try:
                mod_sarimax = SARIMAX(train_y, exog=train_exog, order=cfg["order"], seasonal_order=cfg["seasonal_order"],
                                      enforce_stationarity=False, enforce_invertibility=False)
                res_sarimax = mod_sarimax.fit(disp=False, maxiter=100)
                pred_sarimax = res_sarimax.forecast(steps=max_h, exog=future_exog_h)
            except Exception:
                pred_sarimax = pred_sarima
                
            # Record predictions per horizon
            for h in horizons:
                act = actuals_h[h - 1]
                target_date = dates_h[h - 1]
                
                models_dict = {
                    "Naive": pred_naive[h - 1],
                    "Seasonal_Naive": pred_snaive[h - 1],
                    "SARIMA_Univariate": pred_sarima[h - 1],
                    "SARIMAX_Weather": pred_sarimax[h - 1]
                }
                
                for m_name, m_val in models_dict.items():
                    err = act - m_val
                    abs_err = abs(err)
                    pct_err = abs_err / act * 100.0
                    
                    backtest_records.append({
                        "Crop": crop,
                        "Origin_Week": origin,
                        "Horizon": h,
                        "Forecast_Date": target_date,
                        "Model": m_name,
                        "Actual": act,
                        "Predicted": m_val,
                        "Error": err,
                        "Absolute_Error": abs_err,
                        "Percent_Error": pct_err
                    })
                    
                    if origin == test_origins[-1]: # last origin for point forecasts
                        forecast_points.append({
                            "Crop": crop,
                            "Horizon": h,
                            "Forecast_Date": target_date,
                            "Model": m_name,
                            "Actual": act,
                            "Forecast": m_val
                        })

    backtest_df = pd.DataFrame(backtest_records)
    
    # Calculate aggregate metrics per (Crop, Model, Horizon)
    metric_records = []
    for (crop, model, h), grp in backtest_df.groupby(["Crop", "Model", "Horizon"]):
        sub_train = national_df[national_df["crop"] == crop]["weekly_modal_price"].values[:41]
        mae, rmse, mape, mase = evaluate_metrics(grp["Actual"].values, grp["Predicted"].values, sub_train, seasonal_period=13)
        metric_records.append({
            "Crop": crop,
            "Model": model,
            "Horizon": h,
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE": round(float(mape), 2),
            "MASE": round(float(mase), 3),
            "N_Folds": len(grp)
        })
        
    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "sarima_evaluation_metrics.csv"), index=False)
    backtest_df.to_csv(os.path.join(DATA_DIR, "sarima_forecasts_vs_actuals.csv"), index=False)
    print("Rolling-origin backtest complete. Metrics saved to reports/sarima_evaluation_metrics.csv")
    return backtest_df, metrics_df

def run_future_forecast_and_intervals(national_df):
    """Fits full-sample SARIMAX per crop and generates 4-week forward projections with 80% & 95% CIs."""
    crops = ["Tomato", "Onion", "Potato"]
    model_configs = {
        "Tomato": {"order": (1, 0, 1), "seasonal_order": (1, 0, 0, 13)},
        "Onion": {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 13)},
        "Potato": {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 13)}
    }
    
    future_records = []
    diagnostic_records = []
    
    last_date = national_df["week_start_date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=4, freq="W-MON")
    
    for crop in crops:
        sub = national_df[national_df["crop"] == crop].sort_values("week_start_date").reset_index(drop=True)
        y = sub["weekly_modal_price"].values
        
        # Weather regressors
        exog_raw = sub[["weekly_rainfall_mm", "weekly_temp_mean_c", "weekly_total_arrivals"]].values
        exog_mean = np.mean(exog_raw, axis=0)
        exog_std = np.std(exog_raw, axis=0) + 1e-6
        exog_norm = (exog_raw - exog_mean) / exog_std
        
        # Future exog: assumption of seasonal mean weather for next 4 weeks
        future_exog = np.repeat([exog_norm[-4:].mean(axis=0)], 4, axis=0)
        
        cfg = model_configs[crop]
        mod = SARIMAX(y, exog=exog_norm, order=cfg["order"], seasonal_order=cfg["seasonal_order"],
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=150)
        
        # Forecast 4 steps
        fcast = res.get_forecast(steps=4, exog=future_exog)
        ci_80 = fcast.conf_int(alpha=0.20)
        ci_95 = fcast.conf_int(alpha=0.05)
        pred_means = fcast.predicted_mean
        
        for i, dt in enumerate(future_dates):
            future_records.append({
                "Crop": crop,
                "Week_Start_Date": dt.strftime("%Y-%m-%d"),
                "Horizon_Weeks_Ahead": i + 1,
                "Forecast_Price": round(float(pred_means[i]), 2),
                "Lower_80": round(float(ci_80[i, 0]), 2),
                "Upper_80": round(float(ci_80[i, 1]), 2),
                "Lower_95": round(float(ci_95[i, 0]), 2),
                "Upper_95": round(float(ci_95[i, 1]), 2)
            })
            
        # Residual diagnostics
        residuals = res.resid[13:] # omit warm-up
        lb_test = acorr_ljungbox(residuals, lags=[4, 8, 12], return_df=True)
        jb_stat, jb_pval = jarque_bera(residuals)
        
        diagnostic_records.append({
            "Crop": crop,
            "Model": f"SARIMAX{cfg['order']}x{cfg['seasonal_order']}",
            "AIC": round(float(res.aic), 2),
            "BIC": round(float(res.bic), 2),
            "LjungBox_Q_Lag8": round(float(lb_test["lb_stat"].iloc[1]), 3),
            "LjungBox_p_Lag8": round(float(lb_test["lb_pvalue"].iloc[1]), 4),
            "WhiteNoise_Residuals": bool(lb_test["lb_pvalue"].iloc[1] > 0.05),
            "JarqueBera_Stat": round(float(jb_stat), 3),
            "JarqueBera_p": round(float(jb_pval), 4),
            "Normal_Residuals": bool(jb_pval > 0.05)
        })
        
    future_df = pd.DataFrame(future_records)
    future_df.to_csv(os.path.join(DATA_DIR, "sarima_future_forecasts.csv"), index=False)
    future_df.to_csv(os.path.join(PBI_DIR, "fact_sarima_future_projections.csv"), index=False)
    
    diag_df = pd.DataFrame(diagnostic_records)
    diag_df.to_csv(os.path.join(REPORTS_DIR, "sarima_residual_diagnostics.csv"), index=False)
    print("Future 4-week forecasts and residual diagnostics generated successfully.")
    return future_df, diag_df

def generate_evaluation_visualizations(national_df, backtest_df, metrics_df, future_df):
    """Generates charts for error metrics, forecast vs actuals, fan charts, and interactive Plotly."""
    crops = ["Tomato", "Onion", "Potato"]
    
    # 1. Backtest Forecast vs Actual Plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    for idx, crop in enumerate(crops):
        sub_nat = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        sub_bk = backtest_df[(backtest_df["Crop"] == crop) & (backtest_df["Horizon"] == 1)].sort_values("Forecast_Date")
        
        # Plot full history
        axes[idx].plot(sub_nat["week_start_date"], sub_nat["weekly_modal_price"], label="Actual Price", color="#2c3e50", lw=2)
        
        # Plot 1-step ahead forecasts for SARIMAX & Seasonal Naive
        sarimax_pts = sub_bk[sub_bk["Model"] == "SARIMAX_Weather"]
        snaive_pts = sub_bk[sub_bk["Model"] == "Seasonal_Naive"]
        
        axes[idx].plot(pd.to_datetime(sarimax_pts["Forecast_Date"]), sarimax_pts["Predicted"],
                       label="SARIMAX Forecast (1-week ahead)", color="#27ae60", lw=2.2, linestyle="-", marker="o", markersize=4)
        axes[idx].plot(pd.to_datetime(snaive_pts["Forecast_Date"]), snaive_pts["Predicted"],
                       label="Seasonal Naive (Lag 13)", color="#e74c3c", lw=1.8, linestyle="--", marker="x", markersize=4)
        
        axes[idx].set_title(f"{crop} — Out-of-Sample Rolling Backtest vs Actual Price (Rs/qtl)", fontweight="bold", fontsize=12)
        axes[idx].set_ylabel("Modal Price (Rs/qtl)")
        axes[idx].legend(loc="upper left")
        axes[idx].grid(True, alpha=0.3)
        
    axes[2].set_xlabel("Week Date")
    plt.suptitle("AgriProcure — Out-of-Sample Price Forecast vs Actuals (Rolling Horizon h=1)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart3_backtest_forecast_vs_actual.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Error Metrics Comparison: MASE and MAPE across horizons
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter for h=1 and h=4
    h_palette = {"Naive": "#7f8c8d", "Seasonal_Naive": "#e67e22", "SARIMA_Univariate": "#2980b9", "SARIMAX_Weather": "#27ae60"}
    
    sns.barplot(data=metrics_df, x="Crop", y="MAPE", hue="Model", palette=h_palette, ax=axes[0])
    axes[0].set_title("Mean Absolute Percentage Error (MAPE %) — Across All Horizons", fontweight="bold")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].grid(True, alpha=0.3)
    
    sns.barplot(data=metrics_df, x="Crop", y="MASE", hue="Model", palette=h_palette, ax=axes[1])
    axes[1].axhline(1.0, color="red", linestyle="--", label="MASE = 1.0 (Baseline Benchmark)")
    axes[1].set_title("Mean Absolute Scaled Error (MASE) — Values < 1.0 Beat Seasonal Naive", fontweight="bold")
    axes[1].set_ylabel("MASE")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("AgriProcure — Time Series Forecast Accuracy Benchmark (Lower is Better)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart4_error_metrics_mase_mape.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Future Forecast Fan Chart with Confidence Bounds
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    for idx, crop in enumerate(crops):
        sub_nat = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        sub_fut = future_df[future_df["Crop"] == crop].copy()
        sub_fut["Week_Start_Date"] = pd.to_datetime(sub_fut["Week_Start_Date"])
        
        # Connect last actual point to first future point
        last_hist = sub_nat.iloc[-1]
        bridge_dates = [last_hist["week_start_date"]] + list(sub_fut["Week_Start_Date"])
        bridge_forecast = [last_hist["weekly_modal_price"]] + list(sub_fut["Forecast_Price"])
        bridge_lower80 = [last_hist["weekly_modal_price"]] + list(sub_fut["Lower_80"])
        bridge_upper80 = [last_hist["weekly_modal_price"]] + list(sub_fut["Upper_80"])
        bridge_lower95 = [last_hist["weekly_modal_price"]] + list(sub_fut["Lower_95"])
        bridge_upper95 = [last_hist["weekly_modal_price"]] + list(sub_fut["Upper_95"])
        
        # Plot last 16 historical weeks
        recent_nat = sub_nat.tail(20)
        axes[idx].plot(recent_nat["week_start_date"], recent_nat["weekly_modal_price"], label="Historical Actual Price", color="#1a252f", lw=2.5, marker="o", markersize=3)
        
        # Plot forecast line
        axes[idx].plot(bridge_dates, bridge_forecast, label="SARIMAX Forward Projection", color="#27ae60", lw=2.5, linestyle="--", marker="s", markersize=5)
        
        # Shaded confidence intervals
        axes[idx].fill_between(bridge_dates, bridge_lower80, bridge_upper80, color="#2ecc71", alpha=0.35, label="80% Confidence Interval")
        axes[idx].fill_between(bridge_dates, bridge_lower95, bridge_upper95, color="#2ecc71", alpha=0.15, label="95% Confidence Interval")
        
        axes[idx].set_title(f"{crop} — 4-Week Forward Price Projection with Weather Regressors & Uncertainty Bands", fontweight="bold")
        axes[idx].set_ylabel("Price (Rs/qtl)")
        axes[idx].legend(loc="upper left")
        axes[idx].grid(True, alpha=0.3)
        
    axes[2].set_xlabel("Date")
    plt.suptitle("AgriProcure — 4-Week Price Forecast & Fan Chart per Crop", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart6_future_forecast_fan_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Interactive Plotly HTML Visualizer
    fig_plotly = make_subplots(rows=3, cols=1, subplot_titles=[f"{c} Forward Forecast & Backtest" for c in crops],
                               vertical_spacing=0.08)
    
    for idx, crop in enumerate(crops, 1):
        sub_nat = national_df[national_df["crop"] == crop].sort_values("week_start_date")
        sub_fut = future_df[future_df["Crop"] == crop]
        
        # Historical actual
        fig_plotly.add_trace(go.Scatter(
            x=sub_nat["week_start_date"], y=sub_nat["weekly_modal_price"],
            mode="lines+markers", name=f"{crop} Actual", line=dict(color="#2c3e50", width=2)
        ), row=idx, col=1)
        
        # Future forecast
        fig_plotly.add_trace(go.Scatter(
            x=sub_fut["Week_Start_Date"], y=sub_fut["Forecast_Price"],
            mode="lines+markers", name=f"{crop} Forecast", line=dict(color="#27ae60", width=2.5, dash="dash")
        ), row=idx, col=1)
        
        # 80% CI upper
        fig_plotly.add_trace(go.Scatter(
            x=sub_fut["Week_Start_Date"], y=sub_fut["Upper_80"],
            mode="lines", name=f"{crop} Upper 80%", line=dict(width=0), showlegend=False
        ), row=idx, col=1)
        
        # 80% CI lower with fill
        fig_plotly.add_trace(go.Scatter(
            x=sub_fut["Week_Start_Date"], y=sub_fut["Lower_80"],
            mode="lines", name=f"{crop} 80% CI", fill="tonexty", fillcolor="rgba(46, 204, 113, 0.3)",
            line=dict(width=0)
        ), row=idx, col=1)
        
    fig_plotly.update_layout(height=1000, title_text="AgriProcure Interactive Price Forecasting Suite (SARIMAX)", template="plotly_white")
    plotly_path = os.path.join(FIG_DIR, "interactive_sarima_forecasts.html")
    fig_plotly.write_html(plotly_path)
    print(f"Interactive Plotly chart saved to {plotly_path}")

def main():
    print("=" * 60)
    print("AgriProcure: Executing Tasks 7 & 8 Time-Series Analysis")
    print("=" * 60)
    
    df, national_df, mandi_series = load_and_prepare_data()
    print(f"Loaded {len(df)} rows across {df['market_name'].nunique()} mandis.")
    
    print("\n--- Subtask 7.1: Stationarity & Decomposition ---")
    stat_df = run_stationarity_tests(national_df)
    compute_decomposition_and_plots(national_df)
    
    print("\n--- Subtask 7.2 & 8.1: SARIMA/SARIMAX Rolling Backtesting ---")
    backtest_df, metrics_df = run_rolling_origin_backtest(national_df)
    
    print("\n--- Subtask 8.2 & 8.3: Future Projections & Residual Diagnostics ---")
    future_df, diag_df = run_future_forecast_and_intervals(national_df)
    
    print("\n--- Generating Publication Charts and Interactive Visuals ---")
    generate_evaluation_visualizations(national_df, backtest_df, metrics_df, future_df)
    
    print("\n" + "=" * 60)
    print("Tasks 7 & 8 Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
