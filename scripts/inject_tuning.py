import nbformat as nbf
import json

path = 'notebooks/06_Model_Experimentation.ipynb'
nb = nbf.read(path, as_version=4)

# Markdown cell
md_content = """---

## Experiment 9 — Hyperparameter Tuning

We perform a hyperparameter sweep for our strongest tree-based models, **XGBoost** and **CatBoost**, to determine if we can squeeze out additional predictive performance over the default configurations.

The tuning is evaluated across the same rolling-origin validation folds.
"""
nb.cells.append(nbf.v4.new_markdown_cell(md_content))

# XGBoost Tuning cell
xgb_code = """# XGBoost Hyperparameter Tuning
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import itertools
import numpy as np

xgb_tuning_results = []
params_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
keys, values = zip(*params_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
np.random.seed(42)
np.random.shuffle(combinations)
combinations = combinations[:5] # Random search over 5 configurations for brevity

best_xgb_f1 = 0
best_xgb_params = {}

for params in combinations:
    fold_f1 = []
    for fold_info in validation_folds:
        X_tr = train_df.loc[fold_info['train_indices'], feature_cols]
        y_tr = train_df.loc[fold_info['train_indices'], 'target_encoded']
        X_val = train_df.loc[fold_info['val_indices'], feature_cols]
        y_val = train_df.loc[fold_info['val_indices'], 'target_encoded']
        
        X_tr_processed = preprocessor_xgb.fit_transform(X_tr)
        X_val_processed = preprocessor_xgb.transform(X_val)
        
        model = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        model.fit(X_tr_processed, y_tr)
        
        y_pred = model.predict(X_val_processed)
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        fold_f1.append(f1)
    
    avg_f1 = np.mean(fold_f1)
    if avg_f1 > best_xgb_f1:
        best_xgb_f1 = avg_f1
        best_xgb_params = params

print("Best XGBoost Params:", best_xgb_params)
print(f"Best XGBoost Macro-F1: {best_xgb_f1:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(xgb_code))

# CatBoost Tuning cell
cb_code = """# CatBoost Hyperparameter Tuning
from catboost import CatBoostClassifier

cb_params_grid = {
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [4, 6],
    'iterations': [300, 500]
}
keys, values = zip(*cb_params_grid.items())
cb_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
np.random.seed(42)
np.random.shuffle(cb_combinations)
cb_combinations = cb_combinations[:5]

best_cb_f1 = 0
best_cb_params = {}

for params in cb_combinations:
    fold_f1 = []
    for fold_info in validation_folds:
        X_tr = train_df.loc[fold_info['train_indices'], cb_feature_cols]
        y_tr = train_df.loc[fold_info['train_indices'], 'target_direction']
        X_val = train_df.loc[fold_info['val_indices'], cb_feature_cols]
        y_val = train_df.loc[fold_info['val_indices'], 'target_direction']
        
        model = CatBoostClassifier(**params, loss_function='MultiClass', random_seed=42, verbose=False, thread_count=-1)
        model.fit(X_tr, y_tr, cat_features=cb_cat_cols)
        
        y_pred = model.predict(X_val).ravel()
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        fold_f1.append(f1)
    
    avg_f1 = np.mean(fold_f1)
    if avg_f1 > best_cb_f1:
        best_cb_f1 = avg_f1
        best_cb_params = params

print("Best CatBoost Params:", best_cb_params)
print(f"Best CatBoost Macro-F1: {best_cb_f1:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(cb_code))

nbf.write(nb, path)
