"""
Phase 9 — Overnight run: Hypothesis Tests 3 (artifact-filtering) and 4
(SQI-threshold), all 6 model/horizon combinations.

Test 3 compares the primary SQI<50-filtered target labels against three
stored comparators: EMG>40dB, EMG>35dB (Kim et al.), fully unfiltered.
Test 4 compares primary SQI<50 against a stricter SQI<15 threshold (Huang
et al.'s more conservative precedent) — computed fresh here, since it was
not built during Phase 4.

Reuses Phase 7's saved hyperparameters (no re-tuning). All 5 target-label
variants are relabeled onto the SAME feature rows, restricted to the
INTERSECTION of rows valid across all 5 (since each variant's event-in-progress
exclusion differs, as different artifact-filtering choices produce different
event sets) — ensures every DeLong comparison is a genuine paired test.

NOTE ON SCOPE: Hypothesis Tests 5 (LOCF threshold) and 6 (window-missingness
threshold) are NOT included in this run. Both require rebuilding earlier
pipeline stages (LOCF cascades through interim data + full feature
re-engineering; window-missingness requires per-window missingness %, never
stored) rather than reusing existing artifacts — building and testing those
correctly needs a dedicated session, not an unattended overnight queue.
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
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

RANDOM_SEED = 2311
N_OUTER_FOLDS = 5
LOCF_MAX_GAP_S = 15
MIN_EVENT_DURATION_S = 180

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
HORIZONS_S = {'5min': 300, '10min': 600, '15min': 900}


# ============ Step 1: build the SQI<15 event variant (Test 4's comparator) ============
def locf_bounded(series, max_gap):
    is_na = series.isna()
    group = (~is_na).cumsum()
    run_lengths = is_na.groupby(group).transform('sum')
    fillable = is_na & (run_lengths <= max_gap)
    filled = series.ffill()
    result = series.copy()
    result[fillable] = filled[fillable]
    return result


def load_raw_track_1s(raw_df, tname, target_index):
    series = raw_df[raw_df['tname'] == tname].set_index('dt')['nval']
    if len(series) == 0:
        return pd.Series(np.nan, index=target_index)
    t0 = series.index.min()
    series.index = pd.to_timedelta(series.index - t0, unit='s')
    series = series.resample('1s').mean()
    return series.reindex(target_index)


def detect_events(bis_series, threshold=60, min_duration=180):
    valid_high = bis_series.notna() & (bis_series >= threshold)
    group = (~valid_high).cumsum()
    events = []
    for _, idx in valid_high[valid_high].groupby(group).groups.items():
        duration = len(idx)
        if duration >= min_duration:
            events.append({'start': idx[0], 'end': idx[-1], 'duration_s': duration})
    return events


def build_sqi15_events(cohort_ids):
    print("Building SQI<15 event variant (Hypothesis Test 4 comparator)...")
    all_events = []
    for i, cid in enumerate(cohort_ids):
        interim_path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
        raw_path = os.path.join(RAW_DIR, f"{cid}.parquet")
        try:
            interim = pd.read_parquet(interim_path)
            raw = pd.read_parquet(raw_path)
        except Exception:
            continue
        target_index = interim.index
        bis_raw = load_raw_track_1s(raw, 'BIS', target_index)
        sqi_raw = load_raw_track_1s(raw, 'SQI', target_index)
        bis_base = bis_raw.where(bis_raw != 0, np.nan)
        bis_sqi15 = bis_base.where(sqi_raw >= 15, np.nan)
        bis_sqi15 = locf_bounded(bis_sqi15, LOCF_MAX_GAP_S)

        events = detect_events(bis_sqi15, min_duration=MIN_EVENT_DURATION_S)
        for e in events:
            e['caseid'] = cid
            all_events.append(e)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(cohort_ids)}] events so far: {len(all_events)}")
    events_df = pd.DataFrame(all_events)
    events_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi15.csv'), index=False)
    print(f"SQI<15 events built: {len(events_df)} events across {events_df['caseid'].nunique()} cases")
    return events_df


# ============ Step 2: relabel features against a given event set ============
def label_with_events(features_df, events_df, final_cohort):
    events_df = events_df[events_df['caseid'].isin(final_cohort)].copy()
    events_df['start'] = pd.to_timedelta(events_df['start']).dt.total_seconds().astype(int)
    events_df['end'] = pd.to_timedelta(events_df['end']).dt.total_seconds().astype(int)
    events_by_case = events_df.groupby('caseid')[['start', 'end']].apply(
        lambda g: list(zip(g['start'], g['end']))
    ).to_dict()

    labeled_parts = []
    for cid, group in features_df.groupby('caseid'):
        group = group.copy()
        case_events = events_by_case.get(cid, [])
        t_vals = group['t'].values
        labels = {h: np.zeros(len(t_vals), dtype=object) for h in HORIZONS_S}
        in_progress = np.zeros(len(t_vals), dtype=bool)
        for i, t in enumerate(t_vals):
            for (start, end) in case_events:
                if start <= t <= end:
                    in_progress[i] = True
                    break
            if in_progress[i]:
                for h in HORIZONS_S:
                    labels[h][i] = np.nan
                continue
            for h, horizon_s in HORIZONS_S.items():
                window_end = t + horizon_s
                positive = any(t < start <= window_end for (start, end) in case_events)
                labels[h][i] = 1 if positive else 0
        for h in HORIZONS_S:
            group[f'label_{h}'] = labels[h]
        group['event_in_progress'] = in_progress
        labeled_parts.append(group)

    labeled = pd.concat(labeled_parts, ignore_index=True)
    return labeled[~labeled['event_in_progress']].copy()


# ============ Step 3: refit with saved Phase 7 params, get pooled predictions ============
def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    return pd.DataFrame(train_arr, columns=features), pd.DataFrame(test_arr, columns=features)


def refit_get_predictions(sub, features, label_col, best_params_by_fold):
    X = sub[features].copy()
    y = sub[label_col].astype(int).values
    groups = sub['caseid'].values

    outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
    all_y, all_preds, all_caseids = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        best_params = best_params_by_fold[fold_i]
        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, features)
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

        sm = SMOTE(random_state=RANDOM_SEED)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
        model.fit(X_train_res, y_train_res)
        preds = model.predict_proba(X_test_scaled)[:, 1]

        all_y.append(y_test)
        all_preds.append(preds)
        all_caseids.append(groups_test)

    return np.concatenate(all_y), np.concatenate(all_preds), np.concatenate(all_caseids)


# --- DeLong's test ---
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
    m = int(y_true[order].sum())
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
    total_start = time.time()
    print("Loading base feature table, clinical data, and cohort...")
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))
    final_cohort = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist()
    clinical = pd.read_csv("https://api.vitaldb.net/cases")[['caseid', 'age', 'bmi', 'asa', 'preop_hb']]
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    # --- Build SQI<15 event variant ---
    sqi15_events = build_sqi15_events(final_cohort)

    # --- Load the four existing event variants ---
    event_files = {
        'sqi_primary': 'phase4_events_sqi_primary.csv',
        'emg40': 'phase4_events_emg40.csv',
        'emg35': 'phase4_events_emg35.csv',
        'unfiltered': 'phase4_events_unfiltered.csv',
        'sqi15': 'phase4_events_sqi15.csv',
    }
    event_dfs = {name: pd.read_csv(os.path.join(RESULTS_DIR, fname)) for name, fname in event_files.items()}

    # --- Relabel features against each variant, merge clinical ---
    print("\nRelabeling features against all 5 target-label variants...")
    labeled_variants = {}
    for name, ev_df in event_dfs.items():
        labeled = label_with_events(features, ev_df, final_cohort)
        labeled = labeled.merge(clinical, on='caseid', how='left')
        labeled_variants[name] = labeled
        print(f"  {name}: {len(labeled)} labeled rows")

    # --- Restrict every variant to the intersection of retained rows (by row position key) ---
    # Use (caseid, t) as the row key, since event-in-progress exclusion differs by variant
    print("\nComputing intersection of rows across all 5 variants...")
    key_sets = []
    for name, ldf in labeled_variants.items():
        keys = set(zip(ldf['caseid'], ldf['t']))
        key_sets.append(keys)
    common_keys = set.intersection(*key_sets)
    print(f"Common rows across all 5 variants: {len(common_keys)}")

    for name in labeled_variants:
        ldf = labeled_variants[name]
        keys = pd.MultiIndex.from_arrays([ldf['caseid'], ldf['t']])
        mask = keys.isin(pd.MultiIndex.from_tuples(list(common_keys)))
        labeled_variants[name] = ldf[mask].sort_values(['caseid', 't']).reset_index(drop=True)

    # --- Refit and compare, per model/horizon ---
    all_results = []
    for model_name, feat_list in MODELS.items():
        for horizon in HORIZONS_S:
            print(f"\n=== {model_name} / {horizon} ===")
            label_col = f'label_{horizon}'
            fold_params = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon)
            ].sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()

            variant_preds = {}
            for name, ldf in labeled_variants.items():
                sub = ldf.dropna(subset=[label_col]).dropna(subset=feat_list, how='all').reset_index(drop=True)
                fold_start = time.time()
                y, preds, caseids = refit_get_predictions(sub, feat_list, label_col, fold_params)
                variant_preds[name] = (y, preds, caseids)
                auc = roc_auc_score(y, preds)
                print(f"  {name}: AUROC={auc:.4f} (n={len(y)}, time={time.time()-fold_start:.0f}s)")

            # Compare primary (sqi_primary) against each comparator
            y_primary, preds_primary, _ = variant_preds['sqi_primary']
            for comparator in ['emg40', 'emg35', 'unfiltered', 'sqi15']:
                y_comp, preds_comp, _ = variant_preds[comparator]
                if len(y_primary) == len(y_comp) and np.array_equal(y_primary, y_comp):
                    auc1, auc2, z, p = delong_roc_test(y_primary, preds_primary, preds_comp)
                    print(f"    sqi_primary (AUC={auc1:.4f}) vs {comparator} (AUC={auc2:.4f}): z={z:.3f}, p={p:.6f}")
                    all_results.append({'model': model_name, 'horizon': horizon,
                                         'comparator': comparator, 'auc_primary': auc1,
                                         'auc_comparator': auc2, 'z': z, 'p_value': p})
                else:
                    print(f"    WARNING: {comparator} row mismatch with primary — skipping DeLong for this pair")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_artifact_sqi_sensitivity.csv'), index=False)

    print(f"\n\n=== DONE. Total time: {(time.time()-total_start)/60:.1f} min ===")
    print(results_df.to_string(index=False))