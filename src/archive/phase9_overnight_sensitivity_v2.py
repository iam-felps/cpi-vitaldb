"""
Phase 9 — Hypothesis Tests 3 (artifact-filtering) and 4 (SQI-threshold),
CORRECTED VERSION.

Fix from the first attempt: DeLong's test requires shared ground truth
between the two things being compared. Here, each artifact-filtering variant
has its OWN label definition (a window's "does an event start soon" answer
genuinely differs by which event set defines "event") — even on identical
rows, y_true legitimately differs across variants. DeLong's test therefore
does not apply. Per protocol, the correct tool for this case is a PAIRED
BOOTSTRAP DIFFERENCE test: resample patients once per iteration, compute
AUROC_primary - AUROC_comparator on that same resample (using each variant's
own labels/predictions), repeat, report the CI of the difference.

Row order is verified identical across all 5 variants for a given
model/horizon (same intersection, same sort, same GroupKFold groups array ->
identical deterministic fold assignment), so positional pairing is valid.

All raw predictions are saved to disk this time (results/phase9_sensitivity_predictions.parquet)
so this never needs to be re-run for further analysis.
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
HORIZONS_S = {'5min': 300, '10min': 600, '15min': 900}


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
    events_path = os.path.join(RESULTS_DIR, 'phase4_events_sqi15.csv')
    if os.path.exists(events_path):
        print("SQI<15 events already built, loading from disk...")
        return pd.read_csv(events_path)

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
    events_df.to_csv(events_path, index=False)
    print(f"SQI<15 events built: {len(events_df)} events across {events_df['caseid'].nunique()} cases")
    return events_df


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


def paired_bootstrap_diff(caseids, y_primary, preds_primary, y_comp, preds_comp, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Patient-level paired bootstrap: same resampled patients applied to both
    variants' own labels/predictions each iteration."""
    rng = np.random.RandomState(seed)
    unique_cases = np.unique(caseids)

    # Precompute case -> row-position index once (fast lookup for all iterations)
    case_to_idx = {}
    for c in unique_cases:
        case_to_idx[c] = np.where(caseids == c)[0]

    diffs = []
    for _ in range(n_boot):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        idx = np.concatenate([case_to_idx[c] for c in sampled_cases])

        yp, pp = y_primary[idx], preds_primary[idx]
        yc, pc = y_comp[idx], preds_comp[idx]
        if len(np.unique(yp)) < 2 or len(np.unique(yc)) < 2:
            continue
        auroc_p = roc_auc_score(yp, pp)
        auroc_c = roc_auc_score(yc, pc)
        diffs.append(auroc_p - auroc_c)

    diffs = np.array(diffs)
    return {
        'mean_diff': diffs.mean(),
        'ci_lo': np.percentile(diffs, 2.5),
        'ci_hi': np.percentile(diffs, 97.5),
        'p_value_approx': 2 * min((diffs > 0).mean(), (diffs < 0).mean()),  # proportion-based two-sided approx
    }


if __name__ == '__main__':
    total_start = time.time()
    print("Loading base feature table, clinical data, and cohort...")
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))
    final_cohort = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist()
    clinical = pd.read_csv("https://api.vitaldb.net/cases")[['caseid', 'age', 'bmi', 'asa', 'preop_hb']]
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    sqi15_events = build_sqi15_events(final_cohort)

    event_files = {
        'sqi_primary': 'phase4_events_sqi_primary.csv',
        'emg40': 'phase4_events_emg40.csv',
        'emg35': 'phase4_events_emg35.csv',
        'unfiltered': 'phase4_events_unfiltered.csv',
        'sqi15': 'phase4_events_sqi15.csv',
    }
    event_dfs = {name: pd.read_csv(os.path.join(RESULTS_DIR, fname)) for name, fname in event_files.items()}

    print("\nRelabeling features against all 5 target-label variants...")
    labeled_variants = {}
    for name, ev_df in event_dfs.items():
        labeled = label_with_events(features, ev_df, final_cohort)
        labeled = labeled.merge(clinical, on='caseid', how='left')
        labeled_variants[name] = labeled
        print(f"  {name}: {len(labeled)} labeled rows")

    print("\nComputing intersection of rows across all 5 variants...")
    key_sets = [set(zip(ldf['caseid'], ldf['t'])) for ldf in labeled_variants.values()]
    common_keys = set.intersection(*key_sets)
    print(f"Common rows across all 5 variants: {len(common_keys)}")

    common_keys_mi = pd.MultiIndex.from_tuples(list(common_keys))
    for name in labeled_variants:
        ldf = labeled_variants[name]
        keys = pd.MultiIndex.from_arrays([ldf['caseid'], ldf['t']])
        mask = keys.isin(common_keys_mi)
        labeled_variants[name] = ldf[mask].sort_values(['caseid', 't']).reset_index(drop=True)

    all_predictions = []
    all_bootstrap_results = []

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

                # Save every prediction row for reuse
                for i in range(len(y)):
                    all_predictions.append({
                        'model': model_name, 'horizon': horizon, 'variant': name,
                        'caseid': caseids[i], 'y_true': y[i], 'pred': preds[i],
                    })

            # Paired bootstrap: primary vs each comparator (verified same row/caseid structure)
            y_primary, preds_primary, caseids_primary = variant_preds['sqi_primary']
            for comparator in ['emg40', 'emg35', 'unfiltered', 'sqi15']:
                y_comp, preds_comp, caseids_comp = variant_preds[comparator]
                if not np.array_equal(caseids_primary, caseids_comp):
                    print(f"    WARNING: {comparator} caseid sequence mismatch — skipping bootstrap for this pair")
                    continue
                result = paired_bootstrap_diff(caseids_primary, y_primary, preds_primary, y_comp, preds_comp)
                print(f"    sqi_primary vs {comparator}: mean AUROC diff={result['mean_diff']:.4f} "
                      f"95% CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}] approx_p={result['p_value_approx']:.4f}")
                result.update({'model': model_name, 'horizon': horizon, 'comparator': comparator})
                all_bootstrap_results.append(result)

    preds_df = pd.DataFrame(all_predictions)
    preds_df.to_parquet(os.path.join(RESULTS_DIR, 'phase9_sensitivity_predictions.parquet'))
    print(f"\nSaved {len(preds_df)} prediction rows to results/phase9_sensitivity_predictions.parquet")

    results_df = pd.DataFrame(all_bootstrap_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'phase9_artifact_sqi_sensitivity_bootstrap.csv'), index=False)

    print(f"\n\n=== DONE. Total time: {(time.time()-total_start)/60:.1f} min ===")
    print(results_df.to_string(index=False))