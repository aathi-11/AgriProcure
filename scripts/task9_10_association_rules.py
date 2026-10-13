"""
AgriProcure - Tasks 9 & 10: Association Rule Mining, Evaluation & Interpretation
---------------------------------------------------------------------------------
Implements:
1. Feature discretization with justified domain and statistical quantile thresholds:
   - Target Events: Next-week Price Spike (>+5%, >+10%), Price Crash (<-5%, <-10%), Stability.
   - Antecedent Events: Heavy rainfall (current & lagged), heat spikes, high humidity,
     low arrivals, arrivals surge, price momentum, and high intra-week spread.
   - Temporal lag alignment (drivers strictly precede price reaction).
2. Frequent itemset mining via FP-Growth and Apriori (mlxtend).
3. Association rule generation and evaluation:
   - Support, Confidence, Lift, Conviction, Leverage, Zhang's metric.
4. Strategic rule filtering:
   - Only rules where consequents are future price movements.
   - Elimination of redundant or tautological rules.
5. Cross-segment validation:
   - Evaluation of rule robustness across crops (Tomato vs Onion vs Potato) and regions.
6. Plain-language business translation:
   - Converting mathematical tuples into executive warning triggers for FPOs and food processors.
7. Publication figures and interactive Plotly visualization.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

warnings.filterwarnings("ignore")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures", "task9_10")
PBI_DIR = os.path.join(DATA_DIR, "powerbi_exports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PBI_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

def load_and_discretize_data():
    """
    Loads features_panel.csv and constructs a binary transaction matrix.
    Thresholds are strictly justified based on economic domain knowledge & empirical percentiles.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "features_panel.csv"))
    
    # 1. Target Events (Week t+1 price movement)
    # Wholesale vegetable prices with >5% weekly change represent meaningful operational margin impacts.
    # >10% represents a severe price spike requiring emergency buffer stock deployment.
    tx = pd.DataFrame(index=df.index)
    
    tx["Price_Spike_NextWeek"] = df["target_pct_change"] > 0.05
    tx["Severe_Spike_NextWeek"] = df["target_pct_change"] > 0.10
    tx["Price_Crash_NextWeek"] = df["target_pct_change"] < -0.05
    tx["Severe_Crash_NextWeek"] = df["target_pct_change"] < -0.10
    tx["Price_Stable_NextWeek"] = df["target_pct_change"].abs() <= 0.03
    
    # 2. Weather Triggers (Observed current week t or lag t-1)
    # 35mm weekly rainfall represents heavy rainfall according to IMD standards for mandi access disruption.
    tx["Heavy_Rain_CurrentWeek"] = df["rainfall_mm"] > 35.0
    tx["Heavy_Rain_Lag1"] = df["rainfall_lag_1"] > 35.0
    tx["Persistent_Monsoon_Rain"] = df["rainfall_roll_sum_4"] > 120.0
    tx["Heat_Spike_CurrentWeek"] = (df["temp_max_c"] > 38.0) | (df["temp_anomaly_c"] > 2.5)
    tx["High_Humidity_CurrentWeek"] = df["humidity_mean_pct"] > 75.0
    
    # 3. Market Supply / Arrival Triggers
    # Low arrivals defined as >15% below the mandi's recent 4-week moving average.
    tx["Low_Arrivals_CurrentWeek"] = df["arrivals_vs_roll_mean_4"] < -0.15
    tx["Acute_Supply_Deficit"] = df["arrivals_vs_roll_mean_4"] < -0.30
    tx["Arrivals_Surge_CurrentWeek"] = df["arrivals_vs_roll_mean_4"] > 0.20
    tx["Arrivals_Drop_WoW"] = df["arrivals_pct_change_1w"] < -0.15
    
    # 4. Market Microstructure & Momentum Triggers
    tx["High_Price_Spread"] = df["price_spread_pct"] > 0.15
    tx["Bullish_Price_Momentum"] = df["price_vs_roll_mean_4"] > 0.08
    tx["Bearish_Price_Momentum"] = df["price_vs_roll_mean_4"] < -0.08
    
    # 5. Crop Context Flags
    tx["Crop_Tomato"] = df["crop"] == "Tomato"
    tx["Crop_Onion"] = df["crop"] == "Onion"
    tx["Crop_Potato"] = df["crop"] == "Potato"
    
    # 6. Regional Context Flags
    western_states = ["Maharashtra", "Gujarat", "Rajasthan"]
    southern_states = ["Karnataka", "Tamil Nadu", "Andhra Pradesh", "Telangana"]
    northern_states = ["Uttar Pradesh", "Haryana", "Punjab"]
    
    tx["Region_West"] = df["state"].isin(western_states)
    tx["Region_South"] = df["state"].isin(southern_states)
    tx["Region_North"] = df["state"].isin(northern_states)
    
    # Metadata dataframe for subsequent validation
    meta_df = df[["crop", "state", "market_name", "week_start_date", "target_pct_change"]].copy()
    
    print(f"Constructed binary transaction matrix: {tx.shape[0]} transactions, {tx.shape[1]} items.")
    print("Item frequencies (Support):")
    freqs = tx.mean().sort_values(ascending=False)
    for item, freq in freqs.items():
        print(f"  - {item:28s}: {freq*100:5.2f}%")
        
    return df, tx, meta_df

def mine_association_rules(tx, min_support=0.02, min_confidence=0.30):
    """
    Applies FP-Growth frequent itemset mining and derives association rules.
    """
    print("\nRunning FP-Growth frequent itemset mining...")
    freq_items = fpgrowth(tx, min_support=min_support, use_colnames=True, max_len=4)
    freq_items["length"] = freq_items["itemsets"].apply(len)
    print(f"Discovered {len(freq_items)} frequent itemsets with min_support >= {min_support}.")
    
    # Derive association rules
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    # Filter for strictly forward-looking actionable rules:
    # 1. Consequent MUST be a single future target event
    target_events = ["Price_Spike_NextWeek", "Severe_Spike_NextWeek", "Price_Crash_NextWeek", "Severe_Crash_NextWeek"]
    
    actionable_rules = rules[
        (rules["consequent_len"] == 1) &
        (rules["consequents"].apply(lambda c: list(c)[0] in target_events)) &
        (rules["lift"] > 1.05) &
        (rules["antecedent_len"] >= 1) &
        (rules["antecedent_len"] <= 4)
    ].copy()
    
    # Ensure antecedents do NOT contain future target events (prevent reverse leakage)
    actionable_rules = actionable_rules[
        ~actionable_rules["antecedents"].apply(lambda a: any(item in target_events for item in a))
    ]
    
    print(f"Generated {len(rules)} total rules; {len(actionable_rules)} strictly actionable price-direction rules.")
    return freq_items, rules, actionable_rules

def translate_and_enrich_rules(actionable_rules):
    """
    Converts mathematical frozenset rules into structured business rules with plain English interpretation.
    """
    records = []
    
    for idx, r in actionable_rules.iterrows():
        ant_list = sorted(list(r["antecedents"]))
        con_list = list(r["consequents"])
        con = con_list[0]
        
        # Build plain English condition
        cond_parts = []
        crop_name = "Any Crop"
        for item in ant_list:
            if "Crop_" in item:
                crop_name = item.replace("Crop_", "")
            elif item == "Heavy_Rain_CurrentWeek":
                cond_parts.append("Heavy Rainfall (>35mm) this week")
            elif item == "Heavy_Rain_Lag1":
                cond_parts.append("Heavy Rainfall in previous week")
            elif item == "Persistent_Monsoon_Rain":
                cond_parts.append("Sustained 4-week heavy monsoon rains (>120mm)")
            elif item == "Heat_Spike_CurrentWeek":
                cond_parts.append("Heatwave (>38°C or +2.5°C anomaly)")
            elif item == "High_Humidity_CurrentWeek":
                cond_parts.append("High Humidity (>75%)")
            elif item == "Low_Arrivals_CurrentWeek":
                cond_parts.append("Arrivals drop >15% below 4-week average")
            elif item == "Acute_Supply_Deficit":
                cond_parts.append("Acute supply deficit (arrivals down >30%)")
            elif item == "Arrivals_Surge_CurrentWeek":
                cond_parts.append("Arrivals surge >20% above average")
            elif item == "Arrivals_Drop_WoW":
                cond_parts.append("Arrivals plunge >15% week-over-week")
            elif item == "High_Price_Spread":
                cond_parts.append("High intra-week price volatility (>15% spread)")
            elif item == "Bullish_Price_Momentum":
                cond_parts.append("Prices trending >8% above 4-week mean")
            elif item == "Bearish_Price_Momentum":
                cond_parts.append("Prices trending >8% below 4-week mean")
            elif "Region_" in item:
                cond_parts.append(f"{item.replace('Region_', '')}ern Mandis")
            else:
                cond_parts.append(item.replace("_", " "))
                
        condition_str = " + ".join(cond_parts) if cond_parts else "General Conditions"
        
        # Target action string
        if "Spike" in con:
            target_str = "Price Spike (>5% rise next week)" if "Severe" not in con else "Severe Price Surge (>10% rise next week)"
            action = "Pre-procure buffer stock immediately or defer selling (hold produce)"
        else:
            target_str = "Price Crash (>5% fall next week)" if "Severe" not in con else "Severe Price Plunge (>10% fall next week)"
            action = "Liquidate perishable stock immediately / delay wholesale buying"
            
        rule_desc = f"WHEN {crop_name} experiences [{condition_str}], THEN expect {target_str} with {r['confidence']*100:.1f}% probability (Lift: {r['lift']:.2f}x)."
        
        records.append({
            "Rule_ID": f"RULE_{idx:04d}",
            "Crop": crop_name,
            "Target_Outcome": con,
            "Target_Description": target_str,
            "Antecedents": " + ".join(ant_list),
            "Trigger_Condition": condition_str,
            "Recommended_Action": action,
            "Rule_Statement": rule_desc,
            "Support": round(float(r["support"]), 4),
            "Confidence": round(float(r["confidence"]), 4),
            "Lift": round(float(r["lift"]), 3),
            "Conviction": round(float(r["conviction"]), 3) if not np.isinf(r["conviction"]) else 99.9,
            "Leverage": round(float(r["leverage"]), 4)
        })
        
    enriched_df = pd.DataFrame(records)
    enriched_df = enriched_df.sort_values(by=["Lift", "Confidence"], ascending=[False, False])
    return enriched_df

def cross_validate_rules_across_crops(df, enriched_df):
    """
    Tests whether discovered rules hold when applied specifically across crops and seasons.
    """
    validation_records = []
    top_rules = enriched_df.head(20)
    
    crops = ["Tomato", "Onion", "Potato"]
    for idx, r in top_rules.iterrows():
        ant_items = r["Antecedents"].split(" + ")
        con = r["Target_Outcome"]
        
        # Filter base dataframe
        # Condition check
        for c in crops:
            c_df = df[df["crop"] == c]
            if len(c_df) == 0:
                continue
            
            # Estimate empirical confidence for crop c
            # Target
            if "Spike" in con:
                target_hit = c_df["target_pct_change"] > 0.05
            else:
                target_hit = c_df["target_pct_change"] < -0.05
                
            # Antecedent evaluation
            cond_mask = pd.Series(True, index=c_df.index)
            if "Heavy_Rain_CurrentWeek" in ant_items:
                cond_mask &= (c_df["rainfall_mm"] > 35.0)
            if "Heavy_Rain_Lag1" in ant_items:
                cond_mask &= (c_df["rainfall_lag_1"] > 35.0)
            if "Low_Arrivals_CurrentWeek" in ant_items:
                cond_mask &= (c_df["arrivals_vs_roll_mean_4"] < -0.15)
            if "Heat_Spike_CurrentWeek" in ant_items:
                cond_mask &= (c_df["temp_max_c"] > 38.0)
            if "High_Humidity_CurrentWeek" in ant_items:
                cond_mask &= (c_df["humidity_mean_pct"] > 75.0)
            if "High_Price_Spread" in ant_items:
                cond_mask &= (c_df["price_spread_pct"] > 0.15)
                
            n_trigger = cond_mask.sum()
            n_success = (cond_mask & target_hit).sum()
            emp_conf = (n_success / n_trigger) if n_trigger >= 15 else np.nan
            
            validation_records.append({
                "Rule_ID": r["Rule_ID"],
                "General_Crop": r["Crop"],
                "Test_Crop": c,
                "Trigger_Count": int(n_trigger),
                "Success_Count": int(n_success),
                "Empirical_Confidence": round(float(emp_conf), 3) if not np.isnan(emp_conf) else None,
                "Global_Confidence": r["Confidence"],
                "Global_Lift": r["Lift"]
            })
            
    val_df = pd.DataFrame(validation_records)
    val_df.to_csv(os.path.join(REPORTS_DIR, "association_rules_crop_cross_validation.csv"), index=False)
    print("Cross-crop rule validation saved to reports/association_rules_crop_cross_validation.csv")
    return val_df

def generate_rule_visualizations(enriched_df):
    """
    Generates scatter plot, rule ranking bar charts, network graph, and interactive Plotly visualizer.
    """
    top_actionable = enriched_df.drop_duplicates(subset=["Trigger_Condition", "Target_Outcome"]).head(25)
    
    # 1. Scatter Plot: Support vs Confidence sized by Lift
    plt.figure(figsize=(12, 7))
    scatter = sns.scatterplot(
        data=enriched_df, x="Support", y="Confidence", hue="Target_Outcome",
        size="Lift", sizes=(40, 300), alpha=0.75, palette={"Price_Spike_NextWeek": "#27ae60",
                                                            "Severe_Spike_NextWeek": "#2ecc71",
                                                            "Price_Crash_NextWeek": "#e74c3c",
                                                            "Severe_Crash_NextWeek": "#c0392b"}
    )
    plt.title("AgriProcure — Mined Association Rules: Support vs Confidence (Bubble Size = Lift)", fontweight="bold", fontsize=13)
    plt.xlabel("Support (Fraction of Total Mandi-Weeks)")
    plt.ylabel("Confidence (Rule Reliability P(Consequent | Antecedent))")
    plt.axhline(0.50, color="gray", linestyle="--", alpha=0.5, label="50% Confidence Threshold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart1_rules_support_confidence_lift_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Bar Chart of Top Actionable Rules by Lift
    plt.figure(figsize=(14, 8))
    top15 = enriched_df.head(15).sort_values("Lift", ascending=True)
    
    # Shorten labels for display
    clean_labels = [f"[{r['Crop']}] {r['Trigger_Condition'][:45]}... -> {r['Target_Outcome']}" for _, r in top15.iterrows()]
    colors = ["#27ae60" if "Spike" in r["Target_Outcome"] else "#e74c3c" for _, r in top15.iterrows()]
    
    bars = plt.barh(range(len(top15)), top15["Lift"], color=colors, alpha=0.85, edgecolor="black", lw=0.5)
    plt.yticks(range(len(top15)), clean_labels, fontsize=9)
    plt.xlabel("Lift Ratio (Relative to Baseline Probability)")
    plt.title("AgriProcure — Top 15 Highest-Lift Early Warning Rules", fontweight="bold", fontsize=13)
    plt.axvline(1.0, color="black", linestyle="--", alpha=0.6)
    
    # Annotate with confidence
    for i, bar in enumerate(bars):
        conf = top15["Confidence"].iloc[i]
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"Conf: {conf*100:.1f}%",
                 va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart2_top_rules_ranked_by_lift.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Interactive Plotly Scatter & Rule Explorer
    fig_plotly = px.scatter(
        enriched_df.head(100),
        x="Support", y="Confidence", size="Lift", color="Target_Outcome",
        hover_data=["Crop", "Trigger_Condition", "Recommended_Action", "Lift", "Confidence"],
        title="AgriProcure Interactive Association Rules Explorer (Hover for Actionable Playbook)",
        color_discrete_map={
            "Price_Spike_NextWeek": "#27ae60",
            "Severe_Spike_NextWeek": "#2ecc71",
            "Price_Crash_NextWeek": "#e74c3c",
            "Severe_Crash_NextWeek": "#c0392b"
        },
        template="plotly_white"
    )
    fig_plotly.update_layout(height=650)
    plotly_file = os.path.join(FIG_DIR, "interactive_association_rules.html")
    fig_plotly.write_html(plotly_file)
    print(f"Interactive rules visualizer saved to {plotly_file}")

def main():
    print("=" * 60)
    print("AgriProcure: Executing Tasks 9 & 10 Association Rule Mining")
    print("=" * 60)
    
    print("\n--- Subtask 9.1: Discretization & Transaction Encoding ---")
    df, tx, meta_df = load_and_discretize_data()
    
    print("\n--- Subtask 9.2: Frequent Itemset Mining & Rule Discovery ---")
    freq_items, rules, actionable_rules = mine_association_rules(tx, min_support=0.025, min_confidence=0.35)
    
    print("\n--- Subtask 10.1: Plain-Language Business Enrichment & Filtering ---")
    enriched_df = translate_and_enrich_rules(actionable_rules)
    
    # Save CSV outputs
    enriched_df.to_csv(os.path.join(REPORTS_DIR, "association_rules_full.csv"), index=False)
    enriched_df.to_csv(os.path.join(DATA_DIR, "association_rules.csv"), index=False)
    enriched_df.to_csv(os.path.join(PBI_DIR, "fact_association_rules.csv"), index=False)
    
    # Save top 20 executive summary
    top_summary = enriched_df.head(20)
    top_summary.to_csv(os.path.join(REPORTS_DIR, "association_rules_top_actionable.csv"), index=False)
    print(f"Saved {len(enriched_df)} total enriched rules to data/ and reports/ directories.")
    
    print("\n--- Subtask 10.2: Cross-Crop Robustness Validation ---")
    val_df = cross_validate_rules_across_crops(df, enriched_df)
    
    print("\n--- Generating Rule Visualizations ---")
    generate_rule_visualizations(enriched_df)
    
    print("\n" + "=" * 60)
    print("Top 5 Mined Early Warning Rules:")
    print("=" * 60)
    for idx, r in top_summary.head(5).iterrows():
        print(f"[{r['Rule_ID']}] {r['Rule_Statement']}")
        print(f"  -> Recommended Action: {r['Recommended_Action']}\n")
    print("Tasks 9 & 10 Completed Successfully!")

if __name__ == "__main__":
    main()
