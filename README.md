# From Field to Forecast: A Predictive and Weather-Linked Analytics Framework for Indian Mandi Commodity Prices

Team capstone project (Business Analytics): predicting and explaining weekly price movements of tomato, onion and potato in Indian mandis, using self-collected price, arrivals and weather data.

---

## 1. Problem Description

Prices of vegetables such as tomato, onion and potato swing sharply from week to week in Indian wholesale markets (mandis). These swings are driven by seasonality, arrivals (how much produce reaches the market) and weather events such as heavy rain or heat waves.

Buyers and sellers usually decide *when* to buy or sell on experience and guesswork. A processor that buys just before a price spike pays more than it needed to, and a farmer group that sells just before a spike loses income. There is no simple, data-backed way for them to see:

1. **What is likely to happen to prices next week**, and
2. **Which weather and market conditions tend to come before a price spike.**

This project builds that capability from data the team collects itself.

### Guiding questions

- Will next week's modal price in a given mandi rise, fall or stay flat?
- How well can prices be forecast per crop over the coming weeks?
- Which combinations of arrivals and price movements are frequently followed by price spikes?
- How can these findings support better procurement and selling timing?

---

## 2. Business Objective

**Objective:** Give food processors and farmer-producer organizations (FPOs) an evidence-based view of short-term price direction and price-spike triggers, so they can time procurement and selling decisions better.

### Who benefits and how

| Stakeholder | Decision they face | How this project helps |
|---|---|---|
| Food processors (e.g. chips, ketchup, millers, grain buyers) | When to buy raw material and how much to stock | Weekly price-direction signal and forecasts help avoid buying right before a spike and plan purchases ahead. |
| Farmer-producer organizations (FPOs) | When to sell or hold produce, and which mandi to sell in | Forecasts and spike rules show when prices are likely to rise, and which mandis behave differently. |
| Procurement / supply-chain planners | How to plan inventory and budgets | Early-warning indicators for seasonal price shifts and arrival bottlenecks. |

---

## 3. Scope

| Item | Choice |
|---|---|
| Commodities (5 Crops) | Tomato, Onion, Potato, Rice, Wheat |
| States (14 States) | Punjab, Haryana, UP, MP, Maharashtra, Karnataka, Gujarat, Rajasthan, WB, Bihar, TN, Telangana, AP, Odisha |
| Mandis | 70 Mandis (5 Mandis per State across 14 States) |
| History | 3 Years (Oct 2023 - Sep 2026) |
| Granularity | Market × crop × week panel |
| Target (Review 1) | Next week's modal price: rise, fall or flat (within a ±X% band) |

---

## 4. Data Sources and Collection

All data is **collected by the team using custom web scraping** (pre-built third-party datasets from Kaggle, UCI, GitHub, or live third-party APIs are not used).

| Data | Source | Method |
|---|---|---|
| Mandi prices and arrivals (3-Year History) | Agmarknet Wholesale Mandi Portal | Custom BeautifulSoup HTML Web Scraper on Apify Actor |

The web scraping code (`actor/main.py`), mandi registry (`docs/mandi_list.csv`), and execution guide (`docs/apify_data_collection_guide.md`) are stored in the repository.

---

## 5. How We Are Going to Do It

### Review 1: Data Analysis and Predictive Modelling (Units 1 and 2)

1. **Problem statement and objective:** define the problem, scope and target.
2. **Data collection:** build the CEDA and Open-Meteo collectors; build the weekly panel.
3. **Cleaning and preprocessing:** handle missing weeks, duplicates, name and unit mismatches, and outliers; align mandis with weather locations.
4. **Exploratory data analysis:** price trends, seasonality, price vs rainfall, temperature and arrivals, and volatility by crop and mandi.
5. **Feature engineering and dimensionality reduction:** price and arrivals lags, rolling statistics, weather and calendar features; check leakage and multicollinearity; apply PCA.
6. **Model development:** naive baseline, logistic regression and a tree-based model, using a chronological train/test split and time-series cross-validation; tune and save the final model.
7. **Evaluation and initial findings:** macro-F1, per-class precision and recall, confusion matrix, feature importance and first business findings.

### Review 2: Advanced Analytics (Unit 3)

**Method 1: Time-series analysis**
- Weekly price series per crop, stationarity tests (ADF/KPSS) and decomposition.
- SARIMA models per crop with weather regressors.
- Rolling-origin backtest against naive and seasonal-naive baselines; MAPE and MASE per crop and horizon; residual diagnostics.

**Method 2: Association rule mining**
- Define events with justified thresholds: price spike, heavy rain, heat spike, low arrivals (with lagged events so causes come before effects).
- Apriori or FP-Growth to mine frequent itemsets and generate rules.
- Filter by support, confidence and lift, check that rules hold across crops, mandis or time periods, and interpret them (association is not causation).

**Interactive dashboard (Power BI)**
- Loads exported CSVs of the panel, forecasts and rules (modelling stays in Python notebooks).
- Slicers for crop, mandi and date; forecast vs actual charts; weather overlays; a rules page; KPI cards; drill-through from mandi to detail.

**Combined business insights**
- Combine forecast and rule findings into 3-5 specific procurement-timing recommendations, with limits and risks.

---

## 6. Tools and Technologies

- **Language and analysis:** Python (pandas, NumPy, scikit-learn, statsmodels, mlxtend), Jupyter Notebook
- **Visualisation:** matplotlib / seaborn / plotly for notebooks, Power BI for the interactive dashboard
- **Data collection:** CEDA Agri Market API, Open-Meteo API
- **Project tracking:** Jira board (Backlog, To Do, In Progress, Review/Testing, Completed)
- **Version control:** GitHub

---


## 7. Assumptions and Limitations

- Prices are weekly modal prices aggregated from daily mandi data; missing weeks are flagged or filled and documented.
- Weather is matched to each mandi by location and may not perfectly reflect conditions in the growing region.
- The price-direction band (±X%) is a project choice and will be justified from the data.
- Association rules show co-occurrence, not causation.
- Results may not generalise beyond the chosen crops, mandis and period.

---

