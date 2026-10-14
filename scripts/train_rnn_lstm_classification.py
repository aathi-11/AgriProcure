"""
AgriProcure - RNN & LSTM Deep Learning for Time-Series Analysis & 5% Threshold Classification
---------------------------------------------------------------------------------------------
Implements:
1. ±5% Threshold Target Definition:
   - 3-Class: Rise (>+5%), Fall (<-5%), Flat ([-5%, +5%]).
   - Binary: Increase / Price Spike (>+5%) vs Rest.
2. Classical & Tree Model Benchmarks on 5% Threshold:
   - Naive Directional Persistence Baseline
   - Logistic Regression
   - Random Forest
   - Tuned XGBoost
   - Tuned CatBoost
3. Deep Learning Sequence Models (PyTorch):
   - Sequence Dataset Generator (Rolling window L=4 weeks per crop-mandi)
   - Deep SimpleRNN (2-layer Recurrent Neural Network with Dropout)
   - Deep LSTM (2-layer Long Short-Term Memory Network with Dropout)
   - Class-weighted Cross-Entropy Loss for class imbalance mitigation
   - Continuous Price Regression LSTM for time-series forecasting
4. Comprehensive Metric Reporting:
   - Accuracy, Macro-F1, Weighted-F1, Per-Class Precision & Recall
   - Multi-Class ROC-AUC (One-vs-Rest) & Binary ROC-AUC
   - Confusion Matrices & Training Loss Curves
5. Visualizations & CSV Exports
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures", "deep_learning")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# -------------------------------------------------------------
# 1. Data Preparation with 5% Threshold
# -------------------------------------------------------------
def load_and_prepare_5pct_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "features_panel.csv"))
    
    # 5% Threshold Direction:
    # 0 = Fall (<-5%), 1 = Rise (>+5%), 2 = Flat ([-5%, +5%])
    conditions = [
        df["target_pct_change"] < -5.0,
        df["target_pct_change"] > 5.0
    ]
    choices = [0, 1] # 0: Fall, 1: Rise
    df["target_5pct"] = np.select(conditions, choices, default=2) # 2: Flat
    
    # Binary Increase Target: 1 if Rise (>+5%), 0 otherwise
    df["target_binary_increase"] = (df["target_pct_change"] > 5.0).astype(int)
    
    # Numeric features
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    exclude = ["price", "target_pct_change", "target_5pct", "target_binary_increase"]
    feature_cols = [c for c in num_cols if c not in exclude]
    
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    
    y_train_3c = train_df["target_5pct"].values
    y_test_3c = test_df["target_5pct"].values
    
    y_train_bin = train_df["target_binary_increase"].values
    y_test_bin = test_df["target_binary_increase"].values
    
    print(f"Loaded dataset: {len(df)} rows. Feature dimension: {len(feature_cols)}.")
    print("Class distribution on full panel with 5% threshold:")
    counts = df["target_5pct"].value_counts().sort_index()
    names = {0: "Fall (<-5%)", 1: "Rise (>+5%)", 2: "Flat ([-5%, +5%])"}
    for k, v in counts.items():
        print(f"  - {names[k]:18s}: {v:5d} ({v/len(df)*100:5.2f}%)")
    print(f"Binary Increase (>+5% vs Rest): {df['target_binary_increase'].sum()} ({df['target_binary_increase'].mean()*100:.2f}%)")
    
    return df, train_df, test_df, feature_cols, X_train, X_test, y_train_3c, y_test_3c, y_train_bin, y_test_bin, scaler

# -------------------------------------------------------------
# 2. PyTorch Sequence Dataset Builder
# -------------------------------------------------------------
class MandiSequenceDataset(Dataset):
    """Constructs rolling sequence windows of length L for RNN/LSTM."""
    def __init__(self, df_subset, feature_cols, scaler, seq_len=4, target_type="3class"):
        self.seq_len = seq_len
        self.X_seqs = []
        self.y_targets = []
        self.prices_next = []
        
        # Group by mandi and crop
        for (crop, mandi), grp in df_subset.groupby(["crop", "market_name"]):
            grp = grp.sort_values("week_start_date").reset_index(drop=True)
            if len(grp) < seq_len:
                continue
                
            scaled_features = scaler.transform(grp[feature_cols])
            targets_3c = grp["target_5pct"].values
            targets_bin = grp["target_binary_increase"].values
            prices = grp["price"].values
            pct_changes = grp["target_pct_change"].values
            
            for i in range(len(grp) - seq_len + 1):
                x_window = scaled_features[i : i + seq_len]
                idx_target = i + seq_len - 1
                
                self.X_seqs.append(x_window)
                if target_type == "3class":
                    self.y_targets.append(targets_3c[idx_target])
                elif target_type == "binary":
                    self.y_targets.append(targets_bin[idx_target])
                elif target_type == "regression":
                    # Next week price = price * (1 + pct_change/100)
                    next_p = prices[idx_target] * (1.0 + pct_changes[idx_target] / 100.0)
                    self.y_targets.append(next_p)
                self.prices_next.append(prices[idx_target])
                
        self.X_seqs = np.array(self.X_seqs, dtype=np.float32)
        if target_type in ["3class", "binary"]:
            self.y_targets = np.array(self.y_targets, dtype=np.int64)
        else:
            self.y_targets = np.array(self.y_targets, dtype=np.float32)
            
    def __len__(self):
        return len(self.X_seqs)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X_seqs[idx]), torch.tensor(self.y_targets[idx])

# -------------------------------------------------------------
# 3. Model Architectures: SimpleRNN & LSTM
# -------------------------------------------------------------
class AgriRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(AgriRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x) # out shape: (batch, seq_len, hidden)
        out = out[:, -1, :]  # last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class AgriLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(AgriLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# -------------------------------------------------------------
# 4. Training Engine for Deep Learning Models
# -------------------------------------------------------------
def train_dl_model(model, train_loader, test_loader, epochs=25, lr=0.001, class_weights=None):
    device = torch.device("cpu")
    model.to(device)
    
    if class_weights is not None:
        weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_t)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)
            
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(y_b)
                cls_preds = torch.argmax(preds, dim=1)
                correct += (cls_preds == y_b).sum().item()
                
        test_loss = val_loss / len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        scheduler.step(test_loss)
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
    return history

def predict_dl_model(model, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x_b, _ in data_loader:
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1).numpy()
            cls_preds = np.argmax(probs, axis=1)
            all_preds.extend(cls_preds)
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)

# -------------------------------------------------------------
# 5. Benchmarking All Models
# -------------------------------------------------------------
def benchmark_all_models(X_train, X_test, y_train_3c, y_test_3c,
                         y_train_bin, y_test_bin,
                         train_df, test_df, feature_cols, scaler):
    results_3c = []
    results_bin = []
    cm_dict = {}
    
    class_names = ["Fall (<-5%)", "Rise (>+5%)", "Flat ([-5%, +5%])"]
    
    # Compute inverse class frequencies for weighting
    class_counts = np.bincount(y_train_3c)
    total_samples = len(y_train_3c)
    class_weights = total_samples / (3.0 * class_counts)
    
    # ---------------------------------------------------------
    # Baseline: Directional Persistence (Naive)
    # ---------------------------------------------------------
    naive_pred_3c = []
    for idx, row in test_df.iterrows():
        pct1w = row["pct_change_1w"]
        if pct1w < -5.0:
            naive_pred_3c.append(0)
        elif pct1w > 5.0:
            naive_pred_3c.append(1)
        else:
            naive_pred_3c.append(2)
    naive_pred_3c = np.array(naive_pred_3c)
    naive_pred_bin = (naive_pred_3c == 1).astype(int)
    
    acc = accuracy_score(y_test_3c, naive_pred_3c)
    macro_f1 = f1_score(y_test_3c, naive_pred_3c, average="macro")
    weighted_f1 = f1_score(y_test_3c, naive_pred_3c, average="weighted")
    prec = precision_score(y_test_3c, naive_pred_3c, average=None, zero_division=0)
    rec = recall_score(y_test_3c, naive_pred_3c, average=None, zero_division=0)
    cm = confusion_matrix(y_test_3c, naive_pred_3c)
    cm_dict["Naive Baseline"] = cm
    
    results_3c.append({
        "Model": "Naive Persistence Baseline",
        "Accuracy": round(acc, 4),
        "Macro_F1": round(macro_f1, 4),
        "Weighted_F1": round(weighted_f1, 4),
        "Precision_Fall": round(prec[0], 4),
        "Precision_Rise": round(prec[1], 4),
        "Precision_Flat": round(prec[2], 4),
        "Recall_Fall": round(rec[0], 4),
        "Recall_Rise": round(rec[1], 4),
        "Recall_Flat": round(rec[2], 4),
        "ROC_AUC_OvR": 0.5000
    })
    
    # Binary
    bin_acc = accuracy_score(y_test_bin, naive_pred_bin)
    bin_f1 = f1_score(y_test_bin, naive_pred_bin)
    results_bin.append({
        "Model": "Naive Persistence Baseline",
        "Binary_Accuracy": round(bin_acc, 4),
        "Binary_F1": round(bin_f1, 4),
        "Precision": round(precision_score(y_test_bin, naive_pred_bin, zero_division=0), 4),
        "Recall": round(recall_score(y_test_bin, naive_pred_bin, zero_division=0), 4),
        "ROC_AUC": 0.5000
    })
    
    # ---------------------------------------------------------
    # Classical & Tree Models
    # ---------------------------------------------------------
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Tuned XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0, random_state=SEED, n_jobs=-1)
    }
    if HAS_CATBOOST:
        models_dict["Tuned CatBoost"] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=False, random_seed=SEED)
    
    for name, clf in models_dict.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train_3c)
        preds_3c = clf.predict(X_test)
        probs_3c = clf.predict_proba(X_test)
        
        acc = accuracy_score(y_test_3c, preds_3c)
        macro_f1 = f1_score(y_test_3c, preds_3c, average="macro")
        weighted_f1 = f1_score(y_test_3c, preds_3c, average="weighted")
        prec = precision_score(y_test_3c, preds_3c, average=None, zero_division=0)
        rec = recall_score(y_test_3c, preds_3c, average=None, zero_division=0)
        roc_ovr = roc_auc_score(y_test_3c, probs_3c, multi_class="ovr")
        cm = confusion_matrix(y_test_3c, preds_3c)
        cm_dict[name] = cm
        
        results_3c.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro_F1": round(macro_f1, 4),
            "Weighted_F1": round(weighted_f1, 4),
            "Precision_Fall": round(prec[0], 4),
            "Precision_Rise": round(prec[1], 4),
            "Precision_Flat": round(prec[2], 4),
            "Recall_Fall": round(rec[0], 4),
            "Recall_Rise": round(rec[1], 4),
            "Recall_Flat": round(rec[2], 4),
            "ROC_AUC_OvR": round(roc_ovr, 4)
        })
        
        # Binary Increase Model (fresh instance to avoid multi-class state carryover)
        if "Logistic" in name:
            clf_bin = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        elif "Random Forest" in name:
            clf_bin = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=SEED, n_jobs=-1)
        elif "XGBoost" in name:
            clf_bin = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0, random_state=SEED, n_jobs=-1)
        elif "CatBoost" in name and HAS_CATBOOST:
            clf_bin = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=False, random_seed=SEED)
        else:
            clf_bin = LogisticRegression()
            
        clf_bin.fit(X_train, y_train_bin)
        preds_bin = clf_bin.predict(X_test)
        probs_bin = clf_bin.predict_proba(X_test)[:, 1]
        
        results_bin.append({
            "Model": name,
            "Binary_Accuracy": round(accuracy_score(y_test_bin, preds_bin), 4),
            "Binary_F1": round(f1_score(y_test_bin, preds_bin), 4),
            "Precision": round(precision_score(y_test_bin, preds_bin, zero_division=0), 4),
            "Recall": round(recall_score(y_test_bin, preds_bin, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_test_bin, probs_bin), 4)
        })
        
    # ---------------------------------------------------------
    # Deep Learning Sequence Models: RNN & LSTM
    # ---------------------------------------------------------
    print("Building rolling sequence datasets (L=4 weeks)...")
    train_seq_ds = MandiSequenceDataset(train_df, feature_cols, scaler, seq_len=4, target_type="3class")
    test_seq_ds = MandiSequenceDataset(test_df, feature_cols, scaler, seq_len=4, target_type="3class")
    
    train_loader = DataLoader(train_seq_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_seq_ds, batch_size=64, shuffle=False)
    y_test_seq = test_seq_ds.y_targets
    
    input_dim = len(feature_cols)
    
    # 1. SimpleRNN
    print("Training 2-layer Deep SimpleRNN...")
    rnn_model = AgriRNN(input_size=input_dim, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2)
    rnn_hist = train_dl_model(rnn_model, train_loader, test_loader, epochs=25, lr=0.001, class_weights=class_weights)
    rnn_preds, rnn_probs = predict_dl_model(rnn_model, test_loader)
    
    acc_rnn = accuracy_score(y_test_seq, rnn_preds)
    macro_f1_rnn = f1_score(y_test_seq, rnn_preds, average="macro")
    weighted_f1_rnn = f1_score(y_test_seq, rnn_preds, average="weighted")
    prec_rnn = precision_score(y_test_seq, rnn_preds, average=None, zero_division=0)
    rec_rnn = recall_score(y_test_seq, rnn_preds, average=None, zero_division=0)
    roc_rnn = roc_auc_score(y_test_seq, rnn_probs, multi_class="ovr")
    cm_dict["SimpleRNN"] = confusion_matrix(y_test_seq, rnn_preds)
    
    results_3c.append({
        "Model": "Deep SimpleRNN (PyTorch)",
        "Accuracy": round(acc_rnn, 4),
        "Macro_F1": round(macro_f1_rnn, 4),
        "Weighted_F1": round(weighted_f1_rnn, 4),
        "Precision_Fall": round(prec_rnn[0], 4),
        "Precision_Rise": round(prec_rnn[1], 4),
        "Precision_Flat": round(prec_rnn[2], 4),
        "Recall_Fall": round(rec_rnn[0], 4),
        "Recall_Rise": round(rec_rnn[1], 4),
        "Recall_Flat": round(rec_rnn[2], 4),
        "ROC_AUC_OvR": round(roc_rnn, 4)
    })
    
    # 2. LSTM
    print("Training 2-layer Deep LSTM...")
    lstm_model = AgriLSTM(input_size=input_dim, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2)
    lstm_hist = train_dl_model(lstm_model, train_loader, test_loader, epochs=25, lr=0.001, class_weights=class_weights)
    lstm_preds, lstm_probs = predict_dl_model(lstm_model, test_loader)
    
    acc_lstm = accuracy_score(y_test_seq, lstm_preds)
    macro_f1_lstm = f1_score(y_test_seq, lstm_preds, average="macro")
    weighted_f1_lstm = f1_score(y_test_seq, lstm_preds, average="weighted")
    prec_lstm = precision_score(y_test_seq, lstm_preds, average=None, zero_division=0)
    rec_lstm = recall_score(y_test_seq, lstm_preds, average=None, zero_division=0)
    roc_lstm = roc_auc_score(y_test_seq, lstm_probs, multi_class="ovr")
    cm_dict["Deep LSTM"] = confusion_matrix(y_test_seq, lstm_preds)
    
    results_3c.append({
        "Model": "Deep LSTM (PyTorch)",
        "Accuracy": round(acc_lstm, 4),
        "Macro_F1": round(macro_f1_lstm, 4),
        "Weighted_F1": round(weighted_f1_lstm, 4),
        "Precision_Fall": round(prec_lstm[0], 4),
        "Precision_Rise": round(prec_lstm[1], 4),
        "Precision_Flat": round(prec_lstm[2], 4),
        "Recall_Fall": round(rec_lstm[0], 4),
        "Recall_Rise": round(rec_lstm[1], 4),
        "Recall_Flat": round(rec_lstm[2], 4),
        "ROC_AUC_OvR": round(roc_lstm, 4)
    })
    
    # Binary evaluations for DL models
    y_test_bin_seq = (y_test_seq == 1).astype(int)
    for dl_name, probs_arr in [("Deep SimpleRNN", rnn_probs), ("Deep LSTM", lstm_probs)]:
        bin_prob = probs_arr[:, 1]
        bin_p = (bin_prob > 0.33).astype(int)
        results_bin.append({
            "Model": dl_name,
            "Binary_Accuracy": round(accuracy_score(y_test_bin_seq, bin_p), 4),
            "Binary_F1": round(f1_score(y_test_bin_seq, bin_p), 4),
            "Precision": round(precision_score(y_test_bin_seq, bin_p, zero_division=0), 4),
            "Recall": round(recall_score(y_test_bin_seq, bin_p, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_test_bin_seq, bin_prob), 4)
        })
        
    df_3c = pd.DataFrame(results_3c)
    df_bin = pd.DataFrame(results_bin)
    
    df_3c.to_csv(os.path.join(REPORTS_DIR, "dl_and_ml_classification_metrics_5pct.csv"), index=False)
    df_bin.to_csv(os.path.join(REPORTS_DIR, "binary_prediction_metrics_5pct.csv"), index=False)
    print("Classification metrics saved to reports/dl_and_ml_classification_metrics_5pct.csv")
    
    return df_3c, df_bin, cm_dict, rnn_hist, lstm_hist, y_test_seq, lstm_preds

# -------------------------------------------------------------
# 6. LSTM Time-Series Price Regression (Continuous Forecast)
# -------------------------------------------------------------
def train_lstm_ts_regressor(train_df, test_df, feature_cols, scaler):
    print("\nTraining LSTM for continuous time-series price forecasting...")
    train_ds = MandiSequenceDataset(train_df, feature_cols, scaler, seq_len=4, target_type="regression")
    test_ds = MandiSequenceDataset(test_df, feature_cols, scaler, seq_len=4, target_type="regression")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    input_dim = len(feature_cols)
    reg_model = AgriLSTM(input_size=input_dim, hidden_size=64, num_layers=2, num_classes=1, dropout=0.2)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reg_model.parameters(), lr=0.002, weight_decay=1e-4)
    
    # Train 20 epochs
    reg_model.train()
    for ep in range(20):
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = reg_model(x_b).squeeze(-1)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    reg_model.eval()
    all_preds = []
    all_acts = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            preds = reg_model(x_b).squeeze(-1).numpy()
            all_preds.extend(preds)
            all_acts.extend(y_b.numpy())
            
    all_preds = np.array(all_preds)
    all_acts = np.array(all_acts)
    
    rmse = np.sqrt(mean_squared_error(all_acts, all_preds))
    mae = mean_absolute_error(all_acts, all_preds)
    mape = np.mean(np.abs((all_acts - all_preds) / all_acts)) * 100.0
    
    ts_metrics = pd.DataFrame([{
        "Model": "Deep LSTM Regressor (PyTorch)",
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE_Pct": round(mape, 2)
    }])
    ts_metrics.to_csv(os.path.join(REPORTS_DIR, "dl_timeseries_forecasting_metrics.csv"), index=False)
    print(f"LSTM Time-Series Regressor -> RMSE: Rs. {rmse:.2f} | MAE: Rs. {mae:.2f} | MAPE: {mape:.2f}%")
    return ts_metrics, all_acts, all_preds

# -------------------------------------------------------------
# 7. Visualization Suite
# -------------------------------------------------------------
def generate_dl_visualizations(df_3c, df_bin, cm_dict, rnn_hist, lstm_hist, actuals, lstm_ts_preds):
    # 1. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(rnn_hist["train_loss"]) + 1)
    
    axes[0].plot(epochs_range, rnn_hist["train_loss"], label="RNN Train Loss", color="#e67e22")
    axes[0].plot(epochs_range, rnn_hist["test_loss"], label="RNN Test Loss", color="#d35400", linestyle="--")
    axes[0].plot(epochs_range, lstm_hist["train_loss"], label="LSTM Train Loss", color="#2980b9")
    axes[0].plot(epochs_range, lstm_hist["test_loss"], label="LSTM Test Loss", color="#16a085", linestyle="--")
    axes[0].set_title("Training & Validation Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    
    axes[1].plot(epochs_range, rnn_hist["test_acc"], label="SimpleRNN Test Accuracy", color="#e67e22")
    axes[1].plot(epochs_range, lstm_hist["test_acc"], label="LSTM Test Accuracy", color="#2980b9", lw=2)
    axes[1].set_title("Test Accuracy Trajectory per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart1_lstm_rnn_loss_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Confusion Matrices Grid
    model_keys = list(cm_dict.keys())
    n_cols = 4 if len(model_keys) > 6 else 3
    n_rows = int(np.ceil(len(model_keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()
    class_labels = ["Fall", "Rise", "Flat"]
    
    for idx, m_name in enumerate(model_keys):
        cm = cm_dict[m_name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=class_labels, yticklabels=class_labels)
        axes[idx].set_title(m_name, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        
    for idx in range(len(model_keys), len(axes)):
        axes[idx].set_visible(False)
        
    plt.suptitle("Confusion Matrices Across Models (+-5% Price Movement Threshold)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart2_confusion_matrices_all_models.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Bar Chart Comparison: Accuracy, Macro-F1, ROC-AUC
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df_3c))
    width = 0.25
    
    ax.bar(x - width, df_3c["Accuracy"], width, label="Accuracy", color="#3498db")
    ax.bar(x, df_3c["Macro_F1"], width, label="Macro-F1", color="#2ecc71")
    ax.bar(x + width, df_3c["ROC_AUC_OvR"], width, label="ROC-AUC (OvR)", color="#9b59b6")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_3c["Model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Metric Score")
    ax.set_title("Comprehensive Model Benchmark (+-5% Price Band: Rise, Fall, Flat)", fontweight="bold", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i in range(len(df_3c)):
        ax.text(i, df_3c["Macro_F1"].iloc[i] + 0.01, f"{df_3c['Macro_F1'].iloc[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart3_classification_metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. LSTM Regression: Actual vs Predicted Price Sample
    plt.figure(figsize=(14, 5))
    sample_idx = range(150) # first 150 test predictions for clean view
    plt.plot(sample_idx, actuals[:150], label="Actual Mandi Price (Rs/qtl)", color="#2c3e50", lw=2)
    plt.plot(sample_idx, lstm_ts_preds[:150], label="LSTM Sequence Prediction", color="#e74c3c", linestyle="--", lw=1.8)
    plt.title("Deep LSTM Time-Series Regressor — Test Set Actual vs Predicted Prices (Rs/qtl)", fontweight="bold")
    plt.xlabel("Sample Test Observation")
    plt.ylabel("Modal Price (Rs/qtl)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart4_lstm_forecast_vs_actual.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("All deep learning visualization charts saved to reports/figures/deep_learning/.")

def main():
    print("=" * 70)
    print("AgriProcure: Executing RNN & LSTM Time-Series & 5% Classification Pipeline")
    print("=" * 70)
    
    df, train_df, test_df, feature_cols, X_train, X_test, y_train_3c, y_test_3c, y_train_bin, y_test_bin, scaler = load_and_prepare_5pct_data()
    
    df_3c, df_bin, cm_dict, rnn_hist, lstm_hist, y_test_seq, lstm_preds = benchmark_all_models(
        X_train, X_test, y_train_3c, y_test_3c, y_train_bin, y_test_bin,
        train_df, test_df, feature_cols, scaler
    )
    
    ts_metrics, actuals, lstm_ts_preds = train_lstm_ts_regressor(train_df, test_df, feature_cols, scaler)
    
    generate_dl_visualizations(df_3c, df_bin, cm_dict, rnn_hist, lstm_hist, actuals, lstm_ts_preds)
    
    print("\n" + "=" * 70)
    print("3-CLASS CLASSIFICATION BENCHMARK (Threshold = +-5%):")
    print("=" * 70)
    print(df_3c[["Model", "Accuracy", "Macro_F1", "ROC_AUC_OvR"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("BINARY SPIKE PREDICTION (Price Increase > +5% vs Rest):")
    print("=" * 70)
    print(df_bin[["Model", "Binary_Accuracy", "Binary_F1", "Precision", "Recall", "ROC_AUC"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
