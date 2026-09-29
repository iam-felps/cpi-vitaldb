"""
Phase 7 — Nested Grouped CV Pipeline (full MVP scope: Model A & B, all 3 horizons)

Same verified pipeline as the Model B / 5-min MVP (see decision log entry 28),
generalized to loop over both models and all three horizons. Reuses the exact
same nesting logic, leakage tests, and IterativeImputer-based MICE-equivalent.

Model A = Model B features + TCI effect-site concentrations + Hb
Model B = hemodynamics + nonlinear HR + clinical (age/bmi/asa)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
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
N_INNER_FOLDS = 3
HORIZONS = ['5min', '10min', '15min']

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

LGB_PARAM_GRID = [
    {'learning_rate': 0.05, 'num_leaves': 15, 'n_estimators': 100},
    {'learning_rate': 0.05, 'num_leaves': 31, 'n_estimators': 100},
    {'learning_rate': 0.1, 'num_leaves': 15, 'n_estimators': 100},
    {'learning_rate': 0.1, 'num_leaves': 31, 'n_estimators': 200},
]


def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    train_imputed = pd.DataFrame(train_arr, columns=features)
    test_imputed = pd.DataFrame(test_arr, columns=features)
    learned_signature = dict(zip(features, imputer.initial_imputer_.statistics_))
    return train_imputed, test_imputed, learned_signature


def run_one_combo(df, model_name, features, horizon):
    label_col = f'label_{horizon}'
    sub = df.dropna(subset=[label_col])
    sub = sub.dropna(subset=features, how='all')

    X = sub[features].copy()
    y = sub[label_col].astype(int).values
    groups = sub['caseid'].values

    outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
    fold_results = []
    imputer_signatures = []
    leakage_violations = []
    smote_calls = 0

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        fold_start = time.time()
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        if set(groups_train) & set(groups_test):
            leakage_violations.append(f"{model_name}/{horizon} outer fold {fold_i}: caseid overlap")

        X_train_imp, X_test_imp, sig = mice_impute(X_train_raw, X_test_raw, features)
        imputer_signatures.append(sig)

        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

        inner_cv = GroupKFold(n_splits=N_INNER_FOLDS)
        best_params, best_score = None, -np.inf

        for params in LGB_PARAM_GRID:
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_scaled, y_train, groups_train):
                Xi_train = X_train_scaled.iloc[inner_train_idx]
                Xi_val = X_train_scaled.iloc[inner_val_idx]
                yi_train = y_train[inner_train_idx]
                yi_val = y_train[inner_val_idx]

                gi_train = groups_train[inner_train_idx]
                gi_val = groups_train[inner_val_idx]
                if set(gi_train) & set(gi_val):
                    leakage_violations.append(f"{model_name}/{horizon} outer {fold_i} inner: caseid overlap")

                if yi_train.sum() < 2:
                    continue

                sm = SMOTE(random_state=RANDOM_SEED)
                Xi_train_res, yi_train_res = sm.fit_resample(Xi_train, yi_train)
                smote_calls += 1

                model = lgb.LGBMClassifier(**params, random_state=RANDOM_SEED, verbose=-1)
                model.fit(Xi_train_res, yi_train_res)
                preds = model.predict_proba(Xi_val)[:, 1]
                if len(np.unique(yi_val)) > 1:
                    inner_scores.append(roc_auc_score(yi_val, preds))

            avg_score = np.mean(inner_scores) if inner_scores else -np.inf
            if avg_score > best_score:
                best_score, best_params = avg_score, params

        sm_outer = SMOTE(random_state=RANDOM_SEED)
        X_train_res, y_train_res = sm_outer.fit_resample(X_train_scaled, y_train)
        smote_calls += 1

        final_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
        final_model.fit(X_train_res, y_train_res)
        test_preds = final_model.predict_proba(X_test_scaled)[:, 1]

        auroc = roc_auc_score(y_test, test_preds)
        auprc = average_precision_score(y_test, test_preds)
        fold_time = time.time() - fold_start
        print(f"  [{model_name}/{horizon}] Fold {fold_i}: AUROC={auroc:.4f} AUPRC={auprc:.4f} time={fold_time:.0f}s")

        fold_results.append({'model': model_name, 'horizon': horizon, 'fold': fold_i,
                              'auroc': auroc, 'auprc': auprc, 'best_params': str(best_params),
                              'n_train': len(train_idx), 'n_test': len(test_idx)})

    sigs_identical = all(s == imputer_signatures[0] for s in imputer_signatures)
    if sigs_identical:
        leakage_violations.append(f"{model_name}/{horizon}: all imputer signatures identical")

    return fold_results, leakage_violations, smote_calls


if __name__ == '__main__':
    print("Loading labeled feature table...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    print(f"Total rows: {len(df)}, unique cases: {df['caseid'].nunique()}")

    all_results = []
    all_violations = []
    total_start = time.time()

    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            print(f"\n=== Running {model_name} / {horizon} ===")
            results, violations, n_smote = run_one_combo(df, model_name, features, horizon)
            all_results.extend(results)
            all_violations.extend(violations)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'), index=False)

    print(f"\n\n=== FULL SUMMARY (total time {(time.time()-total_start)/60:.1f} min) ===")
    summary = results_df.groupby(['model', 'horizon']).agg(
        mean_auroc=('auroc', 'mean'), sd_auroc=('auroc', 'std'),
        mean_auprc=('auprc', 'mean'), sd_auprc=('auprc', 'std')
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\n=== Leakage tests across all 6 combinations ===")
    if all_violations:
        print(f"FAILED — {len(all_violations)} violations:")
        for v in all_violations:
            print(f"  {v}")
    else:
        print("PASSED — no violations across any model/horizon combination")