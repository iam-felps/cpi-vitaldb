"""
Phase 9 — Hypothesis Test 1 (core): DeLong's test, Model A vs. Model B, per horizon.

Both models are evaluated on the SAME rows (intersection of rows retained by
each model's own dropna) and the SAME GroupKFold splits, so per-row predictions
are directly paired for DeLong's correlated-ROC test.

Reuses Phase 7's saved hyperparameters per model/horizon/fold (pragmatic
simplification: fold index alignment assumes the intersection subset is very
close to each model's original per-model subset, which holds here since only
a small fraction of rows differ between Model A's and Model B's dropna --
documented as a simplification, not treated as perfectly rigorous).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
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
N_OUTER_FOLDS = 5

MODEL_B_FEATURES = [
    'HR_mean', 'HR_slope', 'HR_var',
    'ART_MBP_mean', 'ART_MBP_slope', 'ART_MBP_var',
    'PLETH_SPO2_mean', 'PLETH_SPO2_slope', 'PLETH_SPO2_var',
    'ETCO2_mean', 'ETCO2_slope', 'ETCO2_var',
    'HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen',
    'elapsed_time_s', 'age', 'bmi', 'asa',
]
MODEL_A_FEATURES = MODEL_B_FEATURES + ['PPF20_CE_mean', 'RFTN20_CE_mean', 'preop_hb']
HORIZONS = ['5min', '10min', '15min']


def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    return pd.DataFrame(train_arr, columns=features), pd.DataFrame(test_arr, columns=features)


def fit_predict(X_train_raw, y_train, X_test_raw, features, best_params):
    X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, features)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

    sm = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
    model.fit(X_train_res, y_train_res)
    return model.predict_proba(X_test_scaled)[:, 1]


# --- DeLong's test implementation (Sun & Xu 2014, fast O(N log N) version) ---
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, preds_a, preds_b):
    """Returns (auc_a, auc_b, z_statistic, p_value) for paired ROC comparison."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    m = int(y_sorted.sum())  # number of positives

    preds = np.vstack([np.asarray(preds_a)[order], np.asarray(preds_b)[order]])
    aucs, cov = _fastDeLong(preds, m)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan, np.nan
    z = diff / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


if __name__ == '__main__':
    print("Loading data and Phase 7 hyperparameters...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    all_results = []
    total_start = time.time()

    for horizon in HORIZONS:
        print(f"\n=== Horizon: {horizon} ===")
        label_col = f'label_{horizon}'
        sub = df.dropna(subset=[label_col])

        # Intersection of rows valid for BOTH models
        valid_a = sub.dropna(subset=MODEL_A_FEATURES, how='all').index
        valid_b = sub.dropna(subset=MODEL_B_FEATURES, how='all').index
        common_idx = valid_a.intersection(valid_b)
        sub = sub.loc[common_idx].reset_index(drop=True)
        print(f"  Rows in intersection: {len(sub)}")

        y = sub[label_col].astype(int).values
        groups = sub['caseid'].values

        params_a = phase7_results[(phase7_results['model'] == 'ModelA') & (phase7_results['horizon'] == horizon)] \
            .sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()
        params_b = phase7_results[(phase7_results['model'] == 'ModelB') & (phase7_results['horizon'] == horizon)] \
            .sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()

        outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
        all_y_test, all_preds_a, all_preds_b = [], [], []

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(sub, y, groups)):
            fold_start = time.time()
            X_train_a = sub.iloc[train_idx][MODEL_A_FEATURES].reset_index(drop=True)
            X_test_a = sub.iloc[test_idx][MODEL_A_FEATURES].reset_index(drop=True)
            X_train_b = sub.iloc[train_idx][MODEL_B_FEATURES].reset_index(drop=True)
            X_test_b = sub.iloc[test_idx][MODEL_B_FEATURES].reset_index(drop=True)
            y_train, y_test = y[train_idx], y[test_idx]

            preds_a = fit_predict(X_train_a, y_train, X_test_a, MODEL_A_FEATURES, params_a[fold_i])
            preds_b = fit_predict(X_train_b, y_train, X_test_b, MODEL_B_FEATURES, params_b[fold_i])

            all_y_test.append(y_test)
            all_preds_a.append(preds_a)
            all_preds_b.append(preds_b)
            print(f"  Fold {fold_i} done, time={time.time()-fold_start:.0f}s")

        y_test_full = np.concatenate(all_y_test)
        preds_a_full = np.concatenate(all_preds_a)
        preds_b_full = np.concatenate(all_preds_b)

        auc_a, auc_b, z, p = delong_roc_test(y_test_full, preds_a_full, preds_b_full)
        print(f"  DeLong's test: AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, z={z:.3f}, p={p:.6f}")

        all_results.append({
            'horizon': horizon, 'auc_model_a': auc_a, 'auc_model_b': auc_b,
            'z_statistic': z, 'p_value': p, 'n_rows': len(sub)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_delong_results.csv'), index=False)

    print(f"\n\n=== SUMMARY (total time {(time.time()-total_start)/60:.1f} min) ===")
    print(results_df.to_string(index=False))