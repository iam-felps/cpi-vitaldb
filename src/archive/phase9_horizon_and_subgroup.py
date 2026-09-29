"""
Phase 9 — Hypothesis Test 2 (horizon-pair comparisons) and ASA subgroup
reporting. Reuses results/phase9_all_predictions.parquet — no refitting.

Horizon-pair DeLong comparisons are valid here because all three horizons'
label columns share the same non-null pattern (the event-in-progress
exclusion applies at time t, independent of horizon), so the same feature
rows and the same GroupKFold(seed) produce IDENTICAL row order across
horizons for a given model — verified explicitly below before trusting the
pairing, rather than assumed.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

HORIZON_PAIRS = [('5min', '10min'), ('10min', '15min'), ('5min', '15min')]
MODELS = ['ModelA', 'ModelB']


# --- DeLong's test (same implementation as phase9_delong_test.py) ---
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
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    m = int(y_sorted.sum())
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
    print("Loading saved predictions...")
    preds = pd.read_parquet(os.path.join(RESULTS_DIR, 'phase9_all_predictions.parquet'))

    # ============ Hypothesis Test 2: horizon-pair comparisons ============
    print("\n=== Hypothesis Test 2: Horizon-pair DeLong comparisons ===")
    horizon_results = []
    for model in MODELS:
        for h1, h2 in HORIZON_PAIRS:
            p1 = preds[(preds['model'] == model) & (preds['horizon'] == h1)].reset_index(drop=True)
            p2 = preds[(preds['model'] == model) & (preds['horizon'] == h2)].reset_index(drop=True)

            # Verify identical row alignment before trusting the pairing
            if len(p1) != len(p2) or not (p1['caseid'].values == p2['caseid'].values).all():
                print(f"  WARNING: {model} {h1} vs {h2} — row alignment mismatch, skipping (needs re-derivation)")
                continue

            auc1, auc2, z, p = delong_roc_test(p1['y_true'].values, p1['raw_pred'].values, p2['raw_pred'].values)
            print(f"  {model}: {h1} (AUC={auc1:.4f}) vs {h2} (AUC={auc2:.4f}), z={z:.3f}, p={p:.6f}")
            horizon_results.append({'model': model, 'horizon_1': h1, 'horizon_2': h2,
                                     'auc_1': auc1, 'auc_2': auc2, 'z': z, 'p_value': p})

    pd.DataFrame(horizon_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_horizon_comparisons.csv'), index=False)

    # ============ ASA subgroup reporting (descriptive, no formal test per protocol) ============
    print("\n=== ASA Subgroup Performance (descriptive, no formal significance test per protocol) ===")
    labeled = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    asa_map = labeled[['caseid', 'asa']].drop_duplicates('caseid')
    preds_with_asa = preds.merge(asa_map, on='caseid', how='left')

    # Pool ASA IV-V per protocol (small cell sizes)
    preds_with_asa['asa_group'] = preds_with_asa['asa'].apply(
        lambda x: 'IV-V' if pd.notna(x) and x >= 4 else (str(int(x)) if pd.notna(x) else 'missing')
    )

    subgroup_results = []
    for model in MODELS:
        for horizon in ['5min', '10min', '15min']:
            subset = preds_with_asa[(preds_with_asa['model'] == model) & (preds_with_asa['horizon'] == horizon)]
            for asa_group in sorted(subset['asa_group'].unique()):
                grp = subset[subset['asa_group'] == asa_group]
                if grp['y_true'].nunique() < 2 or len(grp) < 50:
                    continue
                auroc = roc_auc_score(grp['y_true'], grp['calibrated_pred'])
                n_pos = grp['y_true'].sum()
                subgroup_results.append({'model': model, 'horizon': horizon, 'asa_group': asa_group,
                                          'n_rows': len(grp), 'n_positive': int(n_pos), 'auroc': auroc})
                print(f"  {model}/{horizon}, ASA {asa_group}: n={len(grp)}, positive={n_pos}, AUROC={auroc:.4f}")

    pd.DataFrame(subgroup_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_asa_subgroups.csv'), index=False)
    print("\nDone. Saved horizon comparisons and ASA subgroup results.")