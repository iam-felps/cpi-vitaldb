"""
Phase 8 — Calibration (CPI project)

Reuses hyperparameters already selected in Phase 7 (results/phase7_all_results.csv)
to avoid re-running the expensive tuning grid. For each outer fold, re-derives
the split/impute/scale/SMOTE/fit steps identically (same seed, same params —
AUROC/AUPRC should match Phase 7 exactly, serving as a built-in sanity check),
then adds:

  Nested calibration selection: within the outer-training fold, a further
  3-fold GroupKFold split generates out-of-fold predictions from freshly-fit
  models (same hyperparameters). Isotonic Regression and Platt Scaling are
  each fit on a portion of these OOF predictions and evaluated (Brier score)
  on a held-out portion, repeated across splits. The method with the lower
  mean cross-validated Brier score is selected, refit on the FULL set of OOF
  calibration predictions, and applied to that fold's outer-test predictions.

  This never touches the outer-test set during calibrator fitting or selection.
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
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


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def run_one_combo(df, model_name, features, horizon, saved_params_by_fold):
    label_col = f'label_{horizon}'
    sub = df.dropna(subset=[label_col])
    sub = sub.dropna(subset=features, how='all')

    X = sub[features].copy()
    y = sub[label_col].astype(int).values
    groups = sub['caseid'].values

    outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
    results = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        fold_start = time.time()
        best_params = saved_params_by_fold[fold_i]

        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, features)
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

        # Final model (same as Phase 7 — sanity-check reproduction)
        sm_outer = SMOTE(random_state=RANDOM_SEED)
        X_train_res, y_train_res = sm_outer.fit_resample(X_train_scaled, y_train)
        final_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
        final_model.fit(X_train_res, y_train_res)
        test_preds_raw = final_model.predict_proba(X_test_scaled)[:, 1]

        auroc = roc_auc_score(y_test, test_preds_raw)
        auprc = average_precision_score(y_test, test_preds_raw)

        # --- Nested calibration selection (never touches outer-test) ---
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

        # Select method via a further KFold split over the OOF calibration predictions
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        briers_iso, briers_platt = [], []
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
        chosen_method = 'isotonic' if mean_iso <= mean_platt else 'platt'

        if chosen_method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip').fit(oof_preds, oof_true)
            calibrated_preds = calibrator.predict(test_preds_raw)
        else:
            calibrator = LogisticRegression().fit(oof_preds.reshape(-1, 1), oof_true)
            calibrated_preds = calibrator.predict_proba(test_preds_raw.reshape(-1, 1))[:, 1]

        brier = brier_score_loss(y_test, calibrated_preds)
        ece = compute_ece(y_test, calibrated_preds)

        fold_time = time.time() - fold_start
        print(f"  [{model_name}/{horizon}] Fold {fold_i}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"calib={chosen_method} Brier={brier:.4f} ECE={ece:.4f} time={fold_time:.0f}s")

        results.append({'model': model_name, 'horizon': horizon, 'fold': fold_i,
                         'auroc': auroc, 'auprc': auprc, 'calibration_method': chosen_method,
                         'brier': brier, 'ece': ece})

    return results


if __name__ == '__main__':
    print("Loading labeled feature table and Phase 7 hyperparameters...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    all_results = []
    total_start = time.time()

    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            fold_params = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon)
            ].sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()

            print(f"\n=== Calibrating {model_name} / {horizon} ===")
            results = run_one_combo(df, model_name, features, horizon, fold_params)
            all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase8_calibration_results.csv'), index=False)

    print(f"\n\n=== FULL SUMMARY (total time {(time.time()-total_start)/60:.1f} min) ===")
    summary = results_df.groupby(['model', 'horizon']).agg(
        mean_auroc=('auroc', 'mean'), mean_auprc=('auprc', 'mean'),
        mean_brier=('brier', 'mean'), mean_ece=('ece', 'mean'),
    ).reset_index()
    print(summary.to_string(index=False))

    print("\nCalibration method selected per fold:")
    print(results_df.groupby(['model', 'horizon'])['calibration_method'].apply(lambda x: x.value_counts().to_dict()))