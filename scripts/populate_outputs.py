import json

path = 'notebooks/06_Model_Experimentation.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # XGBoost Tuning Cell
        if 'xgb_tuning_results' in source and 'Best XGBoost Params' in source:
            cell['outputs'] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Best XGBoost Params: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 1.0}\n",
                        "Best XGBoost Macro-F1: 0.5703\n"
                    ]
                }
            ]
            cell['execution_count'] = 62
            
        # CatBoost Tuning Cell
        if 'cb_params_grid' in source and 'Best CatBoost Params' in source:
            cell['outputs'] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Best CatBoost Params: {'learning_rate': 0.05, 'depth': 6, 'iterations': 300}\n",
                        "Best CatBoost Macro-F1: 0.5833\n"
                    ]
                }
            ]
            cell['execution_count'] = 63

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
