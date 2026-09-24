"""
Phase 6c / Phase 9 — Matched-sampling secondary design (protocol Section 5.4)

Rationale: the primary design uses ALL valid windows as negatives, which
includes many windows that are only a few minutes away from the next BIS>=60
event (i.e., "easy" negatives are mixed with "hard" negatives that are close
in time to a positive). The matched-sampling secondary design instead builds
a much smaller, matched negative set per case (<=2 windows/case, each >=20
min / 1200s from the start OR end of any event in that case, chosen to be
maximally temporally separated) and pairs it with the SAME positive windows
used in the primary design. This is a class-separation sensitivity check:
if performance holds up (or improves) on this cleaner matched set, it argues
the primary-design result isn't just exploiting proximity-to-event artifacts.

Outputs:
  results/phase9_matched_class_separation.csv
      per horizon: mean difference (positive - negative) in HR_mean,
      ART_MBP_mean, HR_var, ART_MBP_slope, plus n_pos/n_neg.
  results/phase9_matched_sampling_results.csv
      per model/horizon/fold: AUROC/AUPRC on the matched dataset, using
      GroupKFold + IterativeImputer + RobustScaler + SMOTE + LightGBM,
      with hyperparameters reused from Phase 7 (most frequent best_params
      across that combo's 5 outer folds — avoids re-tuning on a much
      smaller matched dataset where inner-fold tuning would be noisy).
"""

import pandas as pd
import numpy as np
import ast
from collections import Counter
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
MIN_EVENT_DISTANCE_S = 1200  # 20 minutes
MIN_NEG_SEPARATION_S = 600   # require the 2 picks per case to be >=10 min apart

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

DIAGNOSTIC_COLS = ['HR_mean', 'ART_MBP_mean', 'HR_var', 'ART_MBP_slope']


# ---------------------------------------------------------------------------
# Negative-set construction
# ---------------------------------------------------------------------------

def dist_to_nearest_event_boundary(t, case_events):
    """Minimum distance (seconds) from t to the nearest event start or end.
    Returns +inf if the case has no events at all."""
    if not case_events:
        return np.inf
    dists = []
    for (start, end) in case_events:
        dists.append(abs(t - start))
        dists.append(abs(t - end))
    return min(dists)


def build_negative_set(case_df, case_events, t_col='t',
                        min_event_distance_s=MIN_EVENT_DISTANCE_S,
                        min_neg_separation_s=MIN_NEG_SEPARATION_S,
                        max_negs=2):
    """Select up to `max_negs` rows from case_df that are:
      - at least `min_event_distance_s` from the start AND end of every
        event in case_events
      - maximally separated from each other (farthest-point greedy pick)
      - at least `min_neg_separation_s` apart from each other

    Returns a list of row-index labels (into case_df.index) chosen as
    negatives for this case. Assumes case_df already excludes
    event-in-progress rows.
    """
    if len(case_df) == 0:
        return []

    t_vals = case_df[t_col].values
    dists = np.array([dist_to_nearest_event_boundary(t, case_events) for t in t_vals])

    eligible_mask = dists >= min_event_distance_s
    eligible_idx = np.where(eligible_mask)[0]
    if len(eligible_idx) == 0:
        return []

    # Pick #1: eligible candidate with the largest distance to any event
    # boundary (the "safest"/most separated point in the case).
    order = eligible_idx[np.argsort(-dists[eligible_idx])]
    first = order[0]
    picked = [first]
    picked_t = [t_vals[first]]

    if max_negs >= 2 and len(order) > 1:
        # Pick #2: farthest-point sampling — among the OTHER eligible
        # candidates, choose the one that is temporally farthest from pick
        # #1. This is what actually maximizes mutual separation (picking by
        # "largest remaining distance-to-event" instead, as a first cut did,
        # tends to cluster both picks near the same end of the case when the
        # event sits closer to one end than the other — caught by a
        # synthetic test before this script was ever run on real data).
        remaining = order[1:]
        remaining_t = t_vals[remaining]
        sep_from_first = np.abs(remaining_t - picked_t[0])
        best_second_pos = np.argmax(sep_from_first)
        if sep_from_first[best_second_pos] >= min_neg_separation_s:
            picked.append(remaining[best_second_pos])

    return list(case_df.index[picked])


def build_all_negatives(labeled_df, events_by_case):
    """Apply build_negative_set per case, over the full labeled table
    (using rows already free of event-in-progress, i.e. phase6 output)."""
    neg_row_labels = []
    for cid, case_df in labeled_df.groupby('caseid'):
        case_events = events_by_case.get(cid, [])
        neg_row_labels.extend(build_negative_set(case_df, case_events))
    return labeled_df.loc[neg_row_labels].copy()


# ---------------------------------------------------------------------------
# Hyperparameter reuse from Phase 7
# ---------------------------------------------------------------------------

def most_common_params(phase7_df, model_name, horizon):
    sub = phase7_df[(phase7_df['model'] == model_name) & (phase7_df['horizon'] == horizon)]
    parsed = [ast.literal_eval(p) for p in sub['best_params']]
    as_tuples = [tuple(sorted(p.items())) for p in parsed]
    counts = Counter(as_tuples)
    most_common_tuple, _ = counts.most_common(1)[0]
    return dict(most_common_tuple)


# ---------------------------------------------------------------------------
# CV pipeline (same structure as Phase 7, but no inner-loop tuning — reuses
# the fixed hyperparameters selected above)
# ---------------------------------------------------------------------------

def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    train_imputed = pd.DataFrame(train_arr, columns=features)
    test_imputed = pd.DataFrame(test_arr, columns=features)
    return train_imputed, test_imputed


def run_one_combo(matched_df, model_name, features, horizon, params):
    label_col = f'label_{horizon}'
    sub = matched_df.dropna(subset=[label_col]).reset_index(drop=True)

    X = sub[features].copy()
    y = sub[label_col].astype(int).values
    groups = sub['caseid'].values

    n_case_groups = len(np.unique(groups))
    n_folds = min(N_OUTER_FOLDS, n_case_groups)
    outer_cv = GroupKFold(n_splits=n_folds)

    fold_results = []
    leakage_violations = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        fold_start = time.time()
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        if set(groups_train) & set(groups_test):
            leakage_violations.append(f"{model_name}/{horizon} fold {fold_i}: caseid overlap")

        if y_train.sum() < 2 or len(np.unique(y_test)) < 2:
            print(f"  [{model_name}/{horizon}] Fold {fold_i}: skipped (insufficient class balance)")
            continue

        X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, features)

        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

        sm = SMOTE(random_state=RANDOM_SEED)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

        model = lgb.LGBMClassifier(**params, random_state=RANDOM_SEED, verbose=-1)
        model.fit(X_train_res, y_train_res)
        test_preds = model.predict_proba(X_test_scaled)[:, 1]

        auroc = roc_auc_score(y_test, test_preds)
        auprc = average_precision_score(y_test, test_preds)
        fold_time = time.time() - fold_start
        print(f"  [{model_name}/{horizon}] Fold {fold_i}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"n_train={len(train_idx)} n_test={len(test_idx)} time={fold_time:.0f}s")

        fold_results.append({
            'model': model_name, 'horizon': horizon, 'fold': fold_i,
            'auroc': auroc, 'auprc': auprc, 'params_used': str(params),
            'n_train': len(train_idx), 'n_test': len(test_idx),
        })

    return fold_results, leakage_violations


if __name__ == '__main__':
    print("Loading base feature table, clinical-merged labels, events, cohort, and Phase 7 hyperparameters...")
    labeled = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    events = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi_primary.csv'))
    final_cohort = set(pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist())
    phase7_df = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    events = events[events['caseid'].isin(final_cohort)].copy()
    events['start'] = pd.to_timedelta(events['start']).dt.total_seconds().astype(int)
    events['end'] = pd.to_timedelta(events['end']).dt.total_seconds().astype(int)
    events_by_case = events.groupby('caseid')[['start', 'end']].apply(
        lambda g: list(zip(g['start'], g['end']))
    ).to_dict()

    print(f"Labeled table: {len(labeled)} rows, {labeled['caseid'].nunique()} cases")

    print("\nBuilding matched negative set (<=2/case, >=20min from any event, max-separated)...")
    neg_start = time.time()
    negatives = build_all_negatives(labeled, events_by_case)
    print(f"Matched negatives built: {len(negatives)} rows across {negatives['caseid'].nunique()} cases "
          f"(time={time.time()-neg_start:.0f}s)")

    # ---- Class-separation diagnostic (Section 7) + assembly per horizon ----
    class_sep_rows = []
    matched_by_horizon = {}
    for horizon in HORIZONS:
        label_col = f'label_{horizon}'
        positives = labeled[labeled[label_col] == 1].copy()
        matched = pd.concat([positives, negatives], ignore_index=False)
        # Force the negative rows' label to 0 for this horizon (by
        # construction they are >=20min from any event, which exceeds the
        # longest horizon of 15min, so this should already hold — this just
        # makes the matched-design label explicit and independent of the
        # primary per-horizon label value).
        matched.loc[negatives.index, label_col] = 0
        matched_by_horizon[horizon] = matched

        pos_means = positives[DIAGNOSTIC_COLS].mean()
        neg_means = negatives[DIAGNOSTIC_COLS].mean()
        diff = pos_means - neg_means
        class_sep_rows.append({
            'horizon': horizon,
            'n_pos': len(positives), 'n_neg': len(negatives),
            'HR_mean_diff': diff['HR_mean'],
            'ART_MBP_mean_diff': diff['ART_MBP_mean'],
            'HR_var_diff': diff['HR_var'],
            'ART_MBP_slope_diff': diff['ART_MBP_slope'],
        })
        print(f"\n{horizon}: n_pos={len(positives)}, n_neg={len(negatives)}")
        print(f"  HR_mean diff (pos-neg)={diff['HR_mean']:.3f}, "
              f"ART_MBP_mean diff={diff['ART_MBP_mean']:.3f}, "
              f"HR_var diff={diff['HR_var']:.3f}, "
              f"ART_MBP_slope diff={diff['ART_MBP_slope']:.5f}")

    class_sep_df = pd.DataFrame(class_sep_rows)
    class_sep_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_matched_class_separation.csv'), index=False)
    print(f"\nSaved class-separation diagnostic to results/phase9_matched_class_separation.csv")

    # ---- Refit all 6 combos on matched data, reusing Phase 7 hyperparameters ----
    all_results = []
    all_violations = []
    total_start = time.time()

    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            params = most_common_params(phase7_df, model_name, horizon)
            print(f"\n=== {model_name} / {horizon} (reusing params: {params}) ===")
            matched = matched_by_horizon[horizon]
            results, violations = run_one_combo(matched, model_name, features, horizon, params)
            all_results.extend(results)
            all_violations.extend(violations)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_matched_sampling_results.csv'), index=False)

    print(f"\n\n=== MATCHED-SAMPLING SUMMARY (total time {(time.time()-total_start)/60:.1f} min) ===")
    summary = results_df.groupby(['model', 'horizon']).agg(
        mean_auroc=('auroc', 'mean'), sd_auroc=('auroc', 'std'),
        mean_auprc=('auprc', 'mean'), sd_auprc=('auprc', 'std'),
        n_folds=('fold', 'count'),
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\n=== Leakage tests ===")
    if all_violations:
        print(f"FAILED — {len(all_violations)} violations:")
        for v in all_violations:
            print(f"  {v}")
    else:
        print("PASSED — no caseid overlap across any fold, any combination")