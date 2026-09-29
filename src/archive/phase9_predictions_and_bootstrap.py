"""
Phase 9 — Save per-row predictions (raw + calibrated) for all 6 model/horizon
combinations, then compute patient-level bootstrap 95% CIs for AUROC/AUPRC.

Saving raw predictions here means subsequent Phase 9 analyses (subgroups,
further sensitivity work) can reuse this file instead of refitting again.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
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
N_CALIB_FOLDS = 3
N_BOOTSTRAP = 1000

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


def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    return pd.DataFrame(train_arr, columns=features), pd.DataFrame(test_arr, columns=features)


def run_one_combo(df, model_name, features, horizon, saved_params_by_fold):
    label_col = f'label_{horizon}'
    sub = df.dropna(subset=[label_col])
    sub = sub.dropna(subset=features, how='all').reset_index(drop=True)

    X = sub[features].copy()
    y = sub[label_col].astype(int).values
    groups = sub['caseid'].values

    outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
    rows_out = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        fold_start = time.time()
        best_params = saved_params_by_fold[fold_i]

        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, features)
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

        sm_outer = SMOTE(random_state=RANDOM_SEED)
        X_train_res, y_train_res = sm_outer.fit_resample(X_train_scaled, y_train)
        final_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
        final_model.fit(X_train_res, y_train_res)
        raw_preds = final_model.predict_proba(X_test_scaled)[:, 1]

        # Nested calibration (same as Phase 8)
        calib_cv = GroupKFold(n_splits=N_CALIB_FOLDS)
        oof_preds, oof_true = [], []
        for c_train_idx, c_val_idx in calib_cv.split(X_train_scaled, y_train, groups_train):
            Xc_train, Xc_val = X_train_scaled.iloc[c_train_idx], X_train_scaled.iloc[c_val_idx]
            yc_train, yc_val = y_train[c_train_idx], y_train[c_val_idx]
            if yc_train.sum() < 2:
                continue
            sm_c = SMOTE(random_state=RANDOM_SEED)
            Xc_train_res, yc_train_res = sm_c.fit_resample(Xc_train, yc_train)
            m = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
            m.fit(Xc_train_res, yc_train_res)
            oof_preds.append(m.predict_proba(Xc_val)[:, 1])
            oof_true.append(yc_val)
        oof_preds = np.concatenate(oof_preds)
        oof_true = np.concatenate(oof_true)

        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        briers_iso, briers_platt = [], []
        from sklearn.metrics import brier_score_loss
        for tr_i, va_i in kf.split(oof_preds):
            if oof_true[tr_i].sum() < 2 or oof_true[va_i].sum() < 1:
                continue
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(oof_preds[tr_i], oof_true[tr_i])
            briers_iso.append(brier_score_loss(oof_true[va_i], iso.predict(oof_preds[va_i])))
            platt = LogisticRegression()
            platt.fit(oof_preds[tr_i].reshape(-1, 1), oof_true[tr_i])
            briers_platt.append(brier_score_loss(oof_true[va_i], platt.predict_proba(oof_preds[va_i].reshape(-1, 1))[:, 1]))

        mean_iso = np.mean(briers_iso) if briers_iso else np.inf
        mean_platt = np.mean(briers_platt) if briers_platt else np.inf
        if mean_iso <= mean_platt:
            calibrator = IsotonicRegression(out_of_bounds='clip').fit(oof_preds, oof_true)
            calibrated_preds = calibrator.predict(raw_preds)
        else:
            calibrator = LogisticRegression().fit(oof_preds.reshape(-1, 1), oof_true)
            calibrated_preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]

        for i in range(len(test_idx)):
            rows_out.append({
                'model': model_name, 'horizon': horizon, 'fold': fold_i,
                'caseid': groups_test[i], 'y_true': y_test[i],
                'raw_pred': raw_preds[i], 'calibrated_pred': calibrated_preds[i],
            })

        print(f"  [{model_name}/{horizon}] Fold {fold_i} done, time={time.time()-fold_start:.0f}s")

    return rows_out


def patient_level_bootstrap(preds_df, n_boot=1000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    cases = preds_df['caseid'].unique()
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        sampled_cases = rng.choice(cases, size=len(cases), replace=True)
        # Build resampled prediction set (rows from repeated patients repeated too)
        resampled = preds_df.set_index('caseid').loc[sampled_cases].reset_index()
        y = resampled['y_true'].values
        p = resampled['calibrated_pred'].values
        if len(np.unique(y)) < 2:
            continue
        aurocs.append(roc_auc_score(y, p))
        auprcs.append(average_precision_score(y, p))
    return {
        'auroc_mean': np.mean(aurocs), 'auroc_ci_lo': np.percentile(aurocs, 2.5), 'auroc_ci_hi': np.percentile(aurocs, 97.5),
        'auprc_mean': np.mean(auprcs), 'auprc_ci_lo': np.percentile(auprcs, 2.5), 'auprc_ci_hi': np.percentile(auprcs, 97.5),
    }


if __name__ == '__main__':
    print("Loading data and Phase 7 hyperparameters...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    all_predictions = []
    total_start = time.time()

    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            fold_params = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon)
            ].sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()

            print(f"\n=== {model_name} / {horizon} ===")
            rows = run_one_combo(df, model_name, features, horizon, fold_params)
            all_predictions.extend(rows)

    preds_df = pd.DataFrame(all_predictions)
    preds_df.to_parquet(os.path.join(RESULTS_DIR, 'phase9_all_predictions.parquet'))
    print(f"\nSaved {len(preds_df)} prediction rows to results/phase9_all_predictions.parquet")

    print(f"\nComputing {N_BOOTSTRAP}-iteration patient-level bootstrap CIs...")
    bootstrap_results = []
    for model_name in MODELS:
        for horizon in HORIZONS:
            subset = preds_df[(preds_df['model'] == model_name) & (preds_df['horizon'] == horizon)]
            boot = patient_level_bootstrap(subset, n_boot=N_BOOTSTRAP)
            boot['model'] = model_name
            boot['horizon'] = horizon
            bootstrap_results.append(boot)
            print(f"  {model_name}/{horizon}: AUROC {boot['auroc_mean']:.4f} [{boot['auroc_ci_lo']:.4f}, {boot['auroc_ci_hi']:.4f}], "
                  f"AUPRC {boot['auprc_mean']:.4f} [{boot['auprc_ci_lo']:.4f}, {boot['auprc_ci_hi']:.4f}]")

    boot_df = pd.DataFrame(bootstrap_results)
    boot_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_bootstrap_cis.csv'), index=False)

    print(f"\n\nTotal time: {(time.time()-total_start)/60:.1f} min")
    print(boot_df.to_string(index=False))