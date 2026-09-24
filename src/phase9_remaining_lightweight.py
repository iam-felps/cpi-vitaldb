"""
Phase 9 — remaining lightweight analyses, all reusing results/phase9_all_predictions.parquet
(no refitting): age subgroup reporting, AUPRC paired-bootstrap for Hypothesis
Tests 1 (Model A vs B) and 2 (horizon pairs), and Shapiro-Wilk normality
checks on the bootstrap metric distributions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import shapiro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

RANDOM_SEED = 2311
N_BOOTSTRAP = 1000
HORIZON_PAIRS = [('5min', '10min'), ('10min', '15min'), ('5min', '15min')]


def bootstrap_metric_diff(df1, df2, metric_fn, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Paired patient-level bootstrap on a given metric (e.g. AUPRC), for two
    prediction sets that share identical row structure (same caseid sequence)."""
    rng = np.random.RandomState(seed)
    cases = df1['caseid'].values
    unique_cases = np.unique(cases)
    case_to_idx = {c: np.where(cases == c)[0] for c in unique_cases}

    diffs, vals1, vals2 = [], [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        idx = np.concatenate([case_to_idx[c] for c in sampled])
        y1, p1 = df1['y_true'].values[idx], df1['calibrated_pred'].values[idx]
        y2, p2 = df2['y_true'].values[idx], df2['calibrated_pred'].values[idx]
        if len(np.unique(y1)) < 2 or len(np.unique(y2)) < 2:
            continue
        m1, m2 = metric_fn(y1, p1), metric_fn(y2, p2)
        vals1.append(m1)
        vals2.append(m2)
        diffs.append(m1 - m2)
    diffs = np.array(diffs)
    return {
        'mean_diff': diffs.mean(), 'ci_lo': np.percentile(diffs, 2.5), 'ci_hi': np.percentile(diffs, 97.5),
        'p_value_approx': 2 * min((diffs > 0).mean(), (diffs < 0).mean()),
        'bootstrap_vals_1': np.array(vals1), 'bootstrap_vals_2': np.array(vals2),
    }


if __name__ == '__main__':
    print("Loading saved predictions...")
    preds = pd.read_parquet(os.path.join(RESULTS_DIR, 'phase9_all_predictions.parquet'))

    # ============ AUPRC paired-bootstrap: Hypothesis Test 1 (Model A vs B) ============
    print("\n=== Hypothesis Test 1 (AUPRC): Model A vs Model B, paired bootstrap ===")
    test1_results = []
    for horizon in ['5min', '10min', '15min']:
        dfA = preds[(preds['model'] == 'ModelA') & (preds['horizon'] == horizon)].reset_index(drop=True)
        dfB = preds[(preds['model'] == 'ModelB') & (preds['horizon'] == horizon)].reset_index(drop=True)
        if len(dfA) != len(dfB) or not (dfA['caseid'].values == dfB['caseid'].values).all():
            print(f"  {horizon}: WARNING row mismatch, skipping")
            continue
        res = bootstrap_metric_diff(dfA, dfB, average_precision_score)
        print(f"  {horizon}: AUPRC_A-AUPRC_B mean_diff={res['mean_diff']:.4f} "
              f"95% CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] approx_p={res['p_value_approx']:.4f}")
        test1_results.append({'horizon': horizon, 'mean_diff': res['mean_diff'],
                               'ci_lo': res['ci_lo'], 'ci_hi': res['ci_hi'], 'p_value_approx': res['p_value_approx']})
    pd.DataFrame(test1_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_test1_auprc_bootstrap.csv'), index=False)

    # ============ AUPRC paired-bootstrap: Hypothesis Test 2 (horizon pairs) ============
    print("\n=== Hypothesis Test 2 (AUPRC): horizon-pair comparisons, paired bootstrap ===")
    test2_results = []
    for model in ['ModelA', 'ModelB']:
        for h1, h2 in HORIZON_PAIRS:
            df1 = preds[(preds['model'] == model) & (preds['horizon'] == h1)].reset_index(drop=True)
            df2 = preds[(preds['model'] == model) & (preds['horizon'] == h2)].reset_index(drop=True)
            if len(df1) != len(df2) or not (df1['caseid'].values == df2['caseid'].values).all():
                print(f"  {model} {h1} vs {h2}: WARNING row mismatch, skipping")
                continue
            res = bootstrap_metric_diff(df1, df2, average_precision_score)
            print(f"  {model}: {h1} vs {h2}: mean_diff={res['mean_diff']:.4f} "
                  f"95% CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] approx_p={res['p_value_approx']:.4f}")
            test2_results.append({'model': model, 'horizon_1': h1, 'horizon_2': h2,
                                   'mean_diff': res['mean_diff'], 'ci_lo': res['ci_lo'],
                                   'ci_hi': res['ci_hi'], 'p_value_approx': res['p_value_approx']})
    pd.DataFrame(test2_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_test2_auprc_bootstrap.csv'), index=False)

    # ============ Shapiro-Wilk normality check on bootstrap AUROC/AUPRC distributions ============
    print("\n=== Shapiro-Wilk normality check (per model/horizon, AUROC & AUPRC bootstrap distributions) ===")
    shapiro_results = []
    rng = np.random.RandomState(RANDOM_SEED)
    for model in ['ModelA', 'ModelB']:
        for horizon in ['5min', '10min', '15min']:
            subset = preds[(preds['model'] == model) & (preds['horizon'] == horizon)].reset_index(drop=True)
            cases = subset['caseid'].values
            unique_cases = np.unique(cases)
            case_to_idx = {c: np.where(cases == c)[0] for c in unique_cases}

            aurocs, auprcs = [], []
            for _ in range(N_BOOTSTRAP):
                sampled = rng.choice(unique_cases, size=len(unique_cases), replace=True)
                idx = np.concatenate([case_to_idx[c] for c in sampled])
                y, p = subset['y_true'].values[idx], subset['calibrated_pred'].values[idx]
                if len(np.unique(y)) < 2:
                    continue
                aurocs.append(roc_auc_score(y, p))
                auprcs.append(average_precision_score(y, p))

            stat_a, p_a = shapiro(aurocs) if len(aurocs) >= 3 else (np.nan, np.nan)
            stat_p, p_p = shapiro(auprcs) if len(auprcs) >= 3 else (np.nan, np.nan)
            print(f"  {model}/{horizon}: AUROC Shapiro p={p_a:.4f} ({'normal' if p_a > 0.05 else 'NOT normal'}), "
                  f"AUPRC Shapiro p={p_p:.4f} ({'normal' if p_p > 0.05 else 'NOT normal'})")
            shapiro_results.append({'model': model, 'horizon': horizon,
                                     'auroc_shapiro_p': p_a, 'auroc_normal': p_a > 0.05,
                                     'auprc_shapiro_p': p_p, 'auprc_normal': p_p > 0.05})
    pd.DataFrame(shapiro_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_shapiro_normality.csv'), index=False)

    # ============ Age subgroup reporting ============
    print("\n=== Age Subgroup Performance (descriptive) ===")
    labeled = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    age_map = labeled[['caseid', 'age']].drop_duplicates('caseid')
    preds_with_age = preds.merge(age_map, on='caseid', how='left')

    # Age bins: <18 (unlikely present), 18-40, 40-65, 65-80, 80+ — standard perioperative age strata
    bins = [0, 40, 65, 80, 150]
    labels = ['<40', '40-65', '65-80', '80+']
    preds_with_age['age_group'] = pd.cut(preds_with_age['age'], bins=bins, labels=labels)

    age_results = []
    for model in ['ModelA', 'ModelB']:
        for horizon in ['5min', '10min', '15min']:
            subset = preds_with_age[(preds_with_age['model'] == model) & (preds_with_age['horizon'] == horizon)]
            for age_group in labels:
                grp = subset[subset['age_group'] == age_group]
                if grp['y_true'].nunique() < 2 or len(grp) < 50:
                    continue
                auroc = roc_auc_score(grp['y_true'], grp['calibrated_pred'])
                n_pos = grp['y_true'].sum()
                age_results.append({'model': model, 'horizon': horizon, 'age_group': age_group,
                                     'n_rows': len(grp), 'n_positive': int(n_pos), 'auroc': auroc})
                print(f"  {model}/{horizon}, Age {age_group}: n={len(grp)}, positive={n_pos}, AUROC={auroc:.4f}")

    pd.DataFrame(age_results).to_csv(os.path.join(RESULTS_DIR, 'phase9_age_subgroups.csv'), index=False)
    print("\nDone. All results saved to results/.")