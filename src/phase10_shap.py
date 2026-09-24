"""
Phase 10 — Interpretability (SHAP)

Fits one final model per model/horizon combination (using Phase 7's saved
best hyperparameters, fold-0 params as a representative choice) on the full
available dataset, computes SHAP values via TreeExplainer (fast, exact for
tree models — not the slow KernelExplainer), and runs the cross-horizon
consistency check (Spearman correlation + Jaccard index) within each model.

NOTE ON TOP-N: the protocol specifies top-50/top-100 Jaccard index, borrowed
from Qian et al.'s much larger (27+) feature space. This study's feature sets
are much smaller (18-21 features), so top-50/100 doesn't meaningfully apply —
adapted here to top-5 and top-10 (or all features when fewer exist), noted
as a deliberate, documented adaptation to feature-set size.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import shap
import ast
from scipy.stats import spearmanr
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


def mice_impute_fit(df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    arr = imputer.fit_transform(df[features])
    return pd.DataFrame(arr, columns=features)


if __name__ == '__main__':
    print("Loading data and Phase 7 hyperparameters...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    total_start = time.time()
    importance_by_model_horizon = {}
    shap_summary_rows = []

    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            print(f"\n=== {model_name} / {horizon} ===")
            label_col = f'label_{horizon}'
            sub = df.dropna(subset=[label_col]).dropna(subset=features, how='all').reset_index(drop=True)

            params_row = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon) & (phase7_results['fold'] == 0)
            ]
            best_params = ast.literal_eval(params_row['best_params'].values[0])

            X_imp = mice_impute_fit(sub, features)
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features)
            y = sub[label_col].astype(int).values

            fold_start = time.time()
            model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
            model.fit(X_scaled, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
            importance_by_model_horizon[(model_name, horizon)] = mean_abs_shap

            print(f"  Top 5 features: {mean_abs_shap.head(5).to_dict()}")
            print(f"  Time: {time.time()-fold_start:.0f}s")

            for feat, val in mean_abs_shap.items():
                shap_summary_rows.append({'model': model_name, 'horizon': horizon, 'feature': feat, 'mean_abs_shap': val})

    shap_df = pd.DataFrame(shap_summary_rows)
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'phase10_shap_importance.csv'), index=False)

    print("\n\n=== Cross-Horizon SHAP Consistency Check ===")
    consistency_results = []
    for model_name in MODELS:
        horizon_pairs = [('5min', '10min'), ('10min', '15min'), ('5min', '15min')]
        for h1, h2 in horizon_pairs:
            imp1 = importance_by_model_horizon[(model_name, h1)]
            imp2 = importance_by_model_horizon[(model_name, h2)]
            common_features = imp1.index.tolist()

            rho, p = spearmanr(imp1[common_features], imp2[common_features])

            n_features = len(common_features)
            for top_n in [5, 10]:
                top_n_actual = min(top_n, n_features)
                top1 = set(imp1.head(top_n_actual).index)
                top2 = set(imp2.head(top_n_actual).index)
                jaccard = len(top1 & top2) / len(top1 | top2)
                print(f"  {model_name}: {h1} vs {h2}, top-{top_n_actual}: Jaccard={jaccard:.3f}")
                consistency_results.append({'model': model_name, 'horizon_1': h1, 'horizon_2': h2,
                                             'spearman_rho': rho, 'spearman_p': p,
                                             'top_n': top_n_actual, 'jaccard': jaccard})
            print(f"  {model_name}: {h1} vs {h2}, full ranking Spearman rho={rho:.3f} (p={p:.4f})")

    pd.DataFrame(consistency_results).to_csv(os.path.join(RESULTS_DIR, 'phase10_shap_consistency.csv'), index=False)

    print(f"\n\nTotal time: {(time.time()-total_start)/60:.1f} min")
    