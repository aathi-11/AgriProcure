"""
Collect daily rainfall and temperature for every mandi in the weekly panel
from the Open-Meteo historical weather (archive) API, then aggregate to the
same Monday-start weeks used in data/cleaned_mandi_crop_week_panel.csv.

Source : https://archive-api.open-meteo.com/v1/archive (free, no API key)
Output : data/weather_daily_by_mandi.csv
         data/weather_weekly_by_mandi.csv

Each mandi is matched to weather by the coordinates of its district
headquarters town (see MANDI_COORDS). This is a point estimate, so it may not
fully reflect conditions in the growing regions that supply the mandi
(see README, Assumptions and Limitations).

Run from the repository root:
    python scripts/collect_weather_openmeteo.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "cleaned_mandi_crop_week_panel.csv"
DAILY_OUT = ROOT / "data" / "weather_daily_by_mandi.csv"
WEEKLY_OUT = ROOT / "data" / "weather_weekly_by_mandi.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "precipitation_sum",
    "rain_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
]
TIMEZONE = "Asia/Kolkata"

# (latitude, longitude) of the district headquarters / market town.
MANDI_COORDS = {
    # Andhra Pradesh
    "Guntur Apmc Market": (16.3067, 80.4365),
    "Kurnool Apmc Market": (15.8281, 78.0373),
    "Rajahmundry Apmc Market": (17.0005, 81.8040),
    "Vijayawada Apmc Market": (16.5062, 80.6480),
    "Visakhapatnam Apmc Market": (17.6868, 83.2185),
    # Bihar
    "Bhagalpur Apmc Market": (25.2425, 86.9842),
    "Gaya Apmc Market": (24.7914, 85.0002),
    "Muzaffarpur Apmc Market": (26.1209, 85.3647),
    "Patna Apmc Market": (25.5941, 85.1376),
    "Purnia Apmc Market": (25.7771, 87.4753),
    # Gujarat
    "Ahmedabad Apmc Market": (23.0225, 72.5714),
    "Gondal Apmc Market": (21.9619, 70.7923),
    "Mahuva Apmc Market": (21.0902, 71.7568),
    "Rajkot Apmc Market": (22.3039, 70.8022),
    "Surat Apmc Market": (21.1702, 72.8311),
    # Haryana
    "Ambala Apmc Market": (30.3782, 76.7767),
    "Hisar Apmc Market": (29.1492, 75.7217),
    "Karnal Apmc Market": (29.6857, 76.9905),
    "Panipat Apmc Market": (29.3909, 76.9635),
    "Sonipat Apmc Market": (28.9931, 77.0151),
    # Karnataka
    "Bangalore Apmc Market": (12.9716, 77.5946),
    "Belgaum Apmc Market": (15.8497, 74.4977),
    "Chitradurga Apmc Market": (14.2251, 76.3980),
    "Davangere Apmc Market": (14.4644, 75.9218),
    "Hubli Apmc Market": (15.3647, 75.1240),
    # Madhya Pradesh
    "Bhopal Apmc Market": (23.2599, 77.4126),
    "Gwalior Apmc Market": (26.2183, 78.1828),
    "Indore Apmc Market": (22.7196, 75.8577),
    "Jabalpur Apmc Market": (23.1815, 79.9864),
    "Ujjain Apmc Market": (23.1765, 75.7885),
    # Maharashtra
    "Lasalgaon Apmc Market": (20.1500, 74.2333),
    "Nagpur Apmc Market": (21.1458, 79.0882),
    "Pimpalgaon Apmc Market": (20.1667, 73.9833),
    "Pune Apmc Market": (18.5204, 73.8567),
    "Solapur Apmc Market": (17.6599, 75.9064),
    # Odisha
    "Balasore Apmc Market": (21.4942, 86.9317),
    "Berhampur Apmc Market": (19.3150, 84.7941),
    "Bhubaneswar Apmc Market": (20.2961, 85.8245),
    "Cuttack Apmc Market": (20.4625, 85.8830),
    "Sambalpur Apmc Market": (21.4669, 83.9812),
    # Punjab
    "Amritsar Apmc Market": (31.6340, 74.8723),
    "Bathinda Apmc Market": (30.2110, 74.9455),
    "Jalandhar Apmc Market": (31.3260, 75.5762),
    "Ludhiana Apmc Market": (30.9010, 75.8573),
    "Patiala Apmc Market": (30.3398, 76.3869),
    # Rajasthan
    "Alwar Apmc Market": (27.5530, 76.6346),
    "Bikaner Apmc Market": (28.0229, 73.3119),
    "Jaipur Apmc Market": (26.9124, 75.7873),
    "Jodhpur Apmc Market": (26.2389, 73.0243),
    "Kota Apmc Market": (25.2138, 75.8648),
    # Tamil Nadu
    "Chennai Apmc Market": (13.0827, 80.2707),
    "Coimbatore Apmc Market": (11.0168, 76.9558),
    "Madurai Apmc Market": (9.9252, 78.1198),
    "Salem Apmc Market": (11.6643, 78.1460),
    "Tiruchirappalli Apmc Market": (10.7905, 78.7047),
    # Telangana
    "Hyderabad Apmc Market": (17.3850, 78.4867),
    "Karimnagar Apmc Market": (18.4386, 79.1288),
    "Mahbubnagar Apmc Market": (16.7488, 78.0035),
    "Nizamabad Apmc Market": (18.6725, 78.0941),
    "Warangal Apmc Market": (17.9689, 79.5941),
    # Uttar Pradesh
    "Agra Apmc Market": (27.1767, 78.0081),
    "Kanpur Apmc Market": (26.4499, 80.3319),
    "Lucknow Apmc Market": (26.8467, 80.9462),
    "Meerut Apmc Market": (28.9845, 77.7064),
    "Varanasi Apmc Market": (25.3176, 82.9739),
    # West Bengal
    "Burdwan Apmc Market": (23.2324, 87.8615),
    "Hooghly Apmc Market": (22.9089, 88.3967),
    "Kolkata Apmc Market": (22.5726, 88.3639),
    "Malda Apmc Market": (25.0108, 88.1411),
    "Siliguri Apmc Market": (26.7271, 88.3953),
}


def fetch_daily(lat, lon, start, end, retries=4):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            return pd.DataFrame(payload["daily"])
        except Exception as exc:  # rate limit or network hiccup
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt + 1} after error: {exc} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch weather for ({lat}, {lon})")


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["week_start_date"])
    mandis = panel[["state", "district", "market_name"]].drop_duplicates()

    missing = set(mandis["market_name"]) - set(MANDI_COORDS)
    if missing:
        raise ValueError(f"No coordinates for: {sorted(missing)}")

    start = panel["week_start_date"].min().strftime("%Y-%m-%d")
    # Cover the full last week (Mon-Sun) but stay inside the archive's range.
    end_needed = panel["week_start_date"].max() + pd.Timedelta(days=6)
    end_allowed = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    end = min(end_needed, end_allowed).strftime("%Y-%m-%d")
    print(f"Fetching {len(mandis)} mandis, {start} to {end}")

    frames = []
    for row in mandis.itertuples(index=False):
        lat, lon = MANDI_COORDS[row.market_name]
        print(f"- {row.market_name} ({lat}, {lon})")
        daily = fetch_daily(lat, lon, start, end)
        daily = daily.rename(columns={"time": "date"})
        daily.insert(0, "market_name", row.market_name)
        daily.insert(0, "district", row.district)
        daily.insert(0, "state", row.state)
        daily["latitude"] = lat
        daily["longitude"] = lon
        frames.append(daily)
        time.sleep(0.5)  # be polite to the free API

    weather = pd.concat(frames, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.rename(
        columns={
            "precipitation_sum": "precipitation_mm",
            "rain_sum": "rain_mm",
            "temperature_2m_mean": "temp_mean_c",
            "temperature_2m_max": "temp_max_c",
            "temperature_2m_min": "temp_min_c",
            "relative_humidity_2m_mean": "humidity_mean_pct",
        }
    )
    weather.to_csv(DAILY_OUT, index=False)
    print(f"Saved {DAILY_OUT.relative_to(ROOT)}: {weather.shape}")

    # Aggregate to Monday-start weeks to match the price panel.
    weather["week_start_date"] = weather["date"] - pd.to_timedelta(
        weather["date"].dt.dayofweek, unit="D"
    )
    weekly = (
        weather.groupby(
            ["state", "district", "market_name", "week_start_date"]
        )
        .agg(
            weekly_rainfall_mm=("precipitation_mm", "sum"),
            rainy_days=("precipitation_mm", lambda s: int((s >= 2.5).sum())),
            max_daily_rainfall_mm=("precipitation_mm", "max"),
            weekly_temp_mean_c=("temp_mean_c", "mean"),
            weekly_temp_max_c=("temp_max_c", "max"),
            weekly_temp_min_c=("temp_min_c", "min"),
            weekly_humidity_mean_pct=("humidity_mean_pct", "mean"),
            weather_days=("date", "count"),
        )
        .reset_index()
        .round(2)
    )
    weekly.to_csv(WEEKLY_OUT, index=False)
    print(f"Saved {WEEKLY_OUT.relative_to(ROOT)}: {weekly.shape}")


if __name__ == "__main__":
    main()
