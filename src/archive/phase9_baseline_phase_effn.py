"""
Phase 9 — Naive baseline comparator, phase-stratified sensitivity analysis,
and effective (clustering-adjusted) sample size.

NAIVE BASELINE NOTE: the protocol specifies "last observed value and trailing
slope." We do not have the literal last-second value stored separately from
the window (only window mean/slope, from Phase 5). The window MEAN is used
as a pragmatic proxy for "current level" here — a documented simplification,
not an exact match to "last observed value." Flagged explicitly, not silently
substituted.

PHASE-STRATIFICATION NOTE: operationalized as each window's own relative
position within its case (t / case_max_t), bucketed early(<0.4)/mid(0.4-0.8)/
late(>0.8), rather than the event's own relative onset time specifically —
a reasonable, documented adaptation given readily available data.

EFFECTIVE SAMPLE SIZE NOTE: implemented as a standard design-effect
calculation (Kish 1965 / Riley et al.'s general approach: n_eff = n / [1 +
(avg_cluster_size - 1) * ICC]), using an ANOVA-based intraclass correlation
estimate on y_true clustered by caseid — an approximation of Riley et al.'s
specific method, not a verified reproduction of their exact formula.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
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

# "Last value" proxy = window mean; "trailing slope" = window slope (both from Phase 5)
BASELINE_FEATURES = ['HR_mean', 'HR_slope', 'ART_MBP_mean', 'ART_MBP_slope',
                      'PLETH_SPO2_mean', 'PLETH_SPO2_slope', 'ETCO2_mean', 'ETCO2_slope']
HORIZONS = ['5min', '10min', '15min']


def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    return pd.DataFrame(train_arr, columns=features), pd.DataFrame(test_arr, columns=features)


# ============ 1. Naive baseline comparator ============
def run_naive_baseline(df):
    print("=== Naive Baseline Comparator (logistic regression, last-value+slope) ===")
    results = []
    all_preds = []
    for horizon in HORIZONS:
        label_col = f'label_{horizon}'
        sub = df.dropna(subset=[label_col]).dropna(subset=BASELINE_FEATURES, how='all').reset_index(drop=True)
        X = sub[BASELINE_FEATURES].copy()
        y = sub[label_col].astype(int).values
        groups = sub['caseid'].values

        outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
        fold_aurocs, fold_auprcs = [], []

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            X_train_raw = X.iloc[train_idx].reset_index(drop=True)
            X_test_raw = X.iloc[test_idx].reset_index(drop=True)
            y_train, y_test = y[train_idx], y[test_idx]
            groups_test = groups[test_idx]

            X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, BASELINE_FEATURES)
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=BASELINE_FEATURES)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=BASELINE_FEATURES)

            sm = SMOTE(random_state=RANDOM_SEED)
            X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            model.fit(X_train_res, y_train_res)
            preds = model.predict_proba(X_test_scaled)[:, 1]

            auroc = roc_auc_score(y_test, preds)
            auprc = average_precision_score(y_test, preds)
            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)

            for i in range(len(test_idx)):
                all_preds.append({'horizon': horizon, 'fold': fold_i, 'caseid': groups_test[i],
                                   'y_true': y_test[i], 'pred': preds[i]})

        print(f"  {horizon}: AUROC {np.mean(fold_aurocs):.4f} (SD {np.std(fold_aurocs):.4f}), "
              f"AUPRC {np.mean(fold_auprcs):.4f} (SD {np.std(fold_auprcs):.4f})")
        results.append({'horizon': horizon, 'mean_auroc': np.mean(fold_aurocs), 'sd_auroc': np.std(fold_aurocs),
                         'mean_auprc': np.mean(fold_auprcs), 'sd_auprc': np.std(fold_auprcs)})

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'phase9_naive_baseline_results.csv'), index=False)
    pd.DataFrame(all_preds).to_parquet(os.path.join(RESULTS_DIR, 'phase9_naive_baseline_predictions.parquet'))
    return pd.DataFrame(results)


# ============ 2. Phase-stratified sensitivity analysis ============
def run_phase_stratified():
    print("\n=== Phase-Stratified Sensitivity Analysis ===")
    preds = pd.read_parquet(os.path.join(RESULTS_DIR, 'phase9_all_predictions.parquet'))
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))

    # Need original 't' per row to compute relative position — merge back via caseid+row order.
    # phase9_all_predictions doesn't carry 't', so approximate using each case's max t as duration
    # and reconstruct relative position from the row's position within its case in the ORIGINAL
    # feature table order (same order used throughout — caseid groupby preserves original order).
    case_max_t = features.groupby('caseid')['t'].max().to_dict()

    # Re-derive t for each prediction row: rebuild the same per-case,-model,-horizon ordered t sequence
    # used during CV (dropna on label + features, grouped by caseid in the same way Phase 6/7 did).
    results = []
    for model in ['ModelA', 'ModelB']:
        for horizon in HORIZONS:
            label_col = f'label_{horizon}'
            labeled = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
            sub = labeled.dropna(subset=[label_col]).reset_index(drop=True)
            # sub's row order matches what was used to build model_preds (same dropna/sort as original scripts)
            subset_preds = preds[(preds['model'] == model) & (preds['horizon'] == horizon)].reset_index(drop=True)

            if len(sub) != len(subset_preds):
                print(f"  {model}/{horizon}: row count mismatch (sub={len(sub)}, preds={len(subset_preds)}) — "
                      f"cannot reliably reconstruct t; skipping phase stratification for this combo")
                continue

            t_vals = sub['t'].values
            case_ids = sub['caseid'].values
            rel_pos = np.array([t_vals[i] / case_max_t.get(case_ids[i], t_vals[i]) if case_max_t.get(case_ids[i], 0) > 0 else np.nan
                                 for i in range(len(sub))])
            phase = pd.cut(rel_pos, bins=[0, 0.4, 0.8, 1.01], labels=['early', 'mid', 'late'], include_lowest=True)

            subset_preds = subset_preds.copy()
            subset_preds['phase'] = phase[:len(subset_preds)] if len(phase) == len(subset_preds) else pd.Series([np.nan] * len(subset_preds))

            for ph in ['early', 'mid', 'late']:
                grp = subset_preds[subset_preds['phase'] == ph]
                if grp['y_true'].nunique() < 2 or len(grp) < 50:
                    continue
                auroc = roc_auc_score(grp['y_true'], grp['calibrated_pred'])
                n_pos = grp['y_true'].sum()
                print(f"  {model}/{horizon}, Phase {ph}: n={len(grp)}, positive={n_pos}, AUROC={auroc:.4f}")
                results.append({'model': model, 'horizon': horizon, 'phase': ph,
                                 'n_rows': len(grp), 'n_positive': int(n_pos), 'auroc': auroc})

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'phase9_phase_stratified.csv'), index=False)


# ============ 3. Effective (clustering-adjusted) sample size ============
def compute_icc_anova(y, groups):
    """One-way random-effects ANOVA ICC for a binary outcome clustered by patient."""
    df = pd.DataFrame({'y': y, 'group': groups})
    grand_mean = df['y'].mean()
    group_means = df.groupby('group')['y'].mean()
    group_sizes = df.groupby('group')['y'].size()
    k = len(group_sizes)
    n_total = len(df)

    ms_between = ((group_sizes * (group_means - grand_mean) ** 2).sum()) / (k - 1)
    ms_within = df.groupby('group')['y'].var(ddof=1).mul(group_sizes - 1).sum() / (n_total - k)

    n0 = (n_total - (group_sizes ** 2).sum() / n_total) / (k - 1)
    if ms_between + (n0 - 1) * ms_within == 0:
        return 0.0
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    return max(0.0, icc)  # clip negative ICC to 0


def run_effective_sample_size():
    print("\n=== Effective (Clustering-Adjusted) Sample Size ===")
    preds = pd.read_parquet(os.path.join(RESULTS_DIR, 'phase9_all_predictions.parquet'))
    results = []
    for model in ['ModelA', 'ModelB']:
        for horizon in HORIZONS:
            subset = preds[(preds['model'] == model) & (preds['horizon'] == horizon)]
            y = subset['y_true'].values
            groups = subset['caseid'].values
            n_total = len(y)
            n_clusters = len(np.unique(groups))
            avg_cluster_size = n_total / n_clusters

            icc = compute_icc_anova(y, groups)
            design_effect = 1 + (avg_cluster_size - 1) * icc
            n_eff = n_total / design_effect

            print(f"  {model}/{horizon}: n_total={n_total}, n_clusters={n_clusters}, "
                  f"avg_cluster_size={avg_cluster_size:.1f}, ICC={icc:.4f}, design_effect={design_effect:.2f}, "
                  f"n_effective={n_eff:.0f}")
            results.append({'model': model, 'horizon': horizon, 'n_total': n_total, 'n_clusters': n_clusters,
                             'avg_cluster_size': avg_cluster_size, 'icc': icc,
                             'design_effect': design_effect, 'n_effective': n_eff})

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'phase9_effective_sample_size.csv'), index=False)


if __name__ == '__main__':
    total_start = time.time()
    print("Loading labeled feature table...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))

    run_naive_baseline(df)
    run_phase_stratified()
    run_effective_sample_size()

    print(f"\n\nTotal time: {(time.time()-total_start)/60:.1f} min")