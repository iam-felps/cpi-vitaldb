"""
Phase 10c — SHAP summary (beeswarm) plots for all 6 model/horizon
combinations. Refits each combination (fast — same as Phase 10's approach,
using Phase 7's saved fold-0 hyperparameters) and saves a standard SHAP
summary plot per combination.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import shap
import ast
import os
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

RANDOM_SEED = 2311

MODEL_B_FEATURES = [
    'HR_mean', 'HR_slope', 'HR_var',
    'ART_MBP_mean', 'ART_MBP_slope', 'ART_MBP_var',
    'PLETH_SPO2_mean', 'PLETH_SPO2_slope', 'PLETH_SPO2_var',
    'ETCO2_mean', 'ETCO2_slope', 'ETCO2_var',
    'HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen',
    'elapsed_time_s', 'age', 'bmi', 'asa',
]
MODEL_A_FEATURES = MODEL_B_FEATURES + ['PPF20_CE_mean', 'RFTN20_CE_mean', 'preop_hb']
MODELS = {'ModelA': MODEL_A_FEATURES, 'ModelB': MODEL_B_FEATURES}
HORIZONS = ['5min', '10min', '15min']

if __name__ == '__main__':
    print("Loading data and Phase 7 hyperparameters...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    total_start = time.time()
    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            print(f"\n=== {model_name} / {horizon} ===")
            fold_start = time.time()
            label_col = f'label_{horizon}'
            sub = df.dropna(subset=[label_col]).dropna(subset=features, how='all').reset_index(drop=True)

            params_row = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon) & (phase7_results['fold'] == 0)
            ]
            best_params = ast.literal_eval(params_row['best_params'].values[0])

            imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
            X_imp = pd.DataFrame(imputer.fit_transform(sub[features]), columns=features)
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features)
            y = sub[label_col].astype(int).values

            model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
            model.fit(X_scaled, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Downsample for plot rendering speed/clarity given large N
            rng = np.random.RandomState(RANDOM_SEED)
            n_plot = min(20000, len(X_scaled))
            idx = rng.choice(len(X_scaled), size=n_plot, replace=False)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[idx], X_scaled.iloc[idx], show=False)
            plt.title(f'SHAP Summary — {model_name}, {horizon} horizon')
            plt.tight_layout()
            outpath = os.path.join(RESULTS_DIR, f'phase10c_shap_summary_{model_name}_{horizon}.png')
            plt.savefig(outpath, dpi=150)
            plt.close()

            print(f"  Saved: {outpath} (time={time.time()-fold_start:.0f}s)")

    print(f"\n\nTotal time: {(time.time()-total_start)/60:.1f} min")