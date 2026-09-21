"""
Phase 5 — Feature Engineering (CPI project), parallelized across cases.

Same feature definitions as before (see decision log entry 23):
  Rolling mean/slope/var for HR, MAP, SpO2, ETCO2 (trailing 5-min windows, 60s stride)
  DFA alpha1 (4-16s), alpha2 (16-64s), sample entropy (m=2, r=0.2*SD) on HR
  Elapsed time since induction
  TCI effect-site concentrations (Model A only)

Parallelized via multiprocessing.Pool across cases (each case is independent —
this is the dominant cost, since DFA/sample entropy are inherently slow per-window).
"""

import pandas as pd
import numpy as np
import nolds
import os
import time
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_S = 300
STRIDE_S = 60
MIN_WINDOW_FRAC = 0.5

DFA_ALPHA1_NVALS = np.unique(np.logspace(np.log10(4), np.log10(16), 8).astype(int))
DFA_ALPHA2_NVALS = np.unique(np.logspace(np.log10(16), np.log10(64), 8).astype(int))

CORE_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2']

_X = np.arange(WINDOW_S)


def fast_rolling_stats(series, window_s, stride_s):
    n = len(series)
    idx_ends = list(range(window_s, n, stride_s))
    means = np.full(len(idx_ends), np.nan)
    variances = np.full(len(idx_ends), np.nan)
    slopes = np.full(len(idx_ends), np.nan)

    arr = series.values
    for j, end in enumerate(idx_ends):
        window = arr[end - window_s:end]
        valid_mask = ~np.isnan(window)
        n_valid = valid_mask.sum()
        if n_valid < window_s * MIN_WINDOW_FRAC:
            continue
        y = window[valid_mask]
        x = _X[valid_mask]
        means[j] = y.mean()
        variances[j] = y.var()
        if n_valid >= 2:
            x_mean = x.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                slopes[j] = np.sum((x - x_mean) * (y - y.mean())) / denom
    return idx_ends, means, slopes, variances


def nonlinear_hr_features(hr_series, window_s, stride_s):
    n = len(hr_series)
    idx_ends = list(range(window_s, n, stride_s))
    a1s = np.full(len(idx_ends), np.nan)
    a2s = np.full(len(idx_ends), np.nan)
    sampens = np.full(len(idx_ends), np.nan)

    arr = hr_series.values
    for j, end in enumerate(idx_ends):
        window = arr[end - window_s:end]
        valid = window[~np.isnan(window)]
        if len(valid) < window_s * MIN_WINDOW_FRAC:
            continue
        try:
            a1s[j] = nolds.dfa(valid, nvals=DFA_ALPHA1_NVALS, order=1)
        except Exception:
            pass
        try:
            a2s[j] = nolds.dfa(valid, nvals=DFA_ALPHA2_NVALS, order=1)
        except Exception:
            pass
        try:
            sd = np.std(valid)
            if sd > 0:
                sampens[j] = nolds.sampen(valid, emb_dim=2, tolerance=0.2 * sd)
        except Exception:
            pass
    return idx_ends, a1s, a2s, sampens


def build_case_features(cid):
    """Worker function — runs in a separate process per case."""
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        return None, (cid, str(e))

    n = len(wide)
    if n < WINDOW_S:
        return None, None

    idx_ends, hr_mean, hr_slope, hr_var = fast_rolling_stats(wide['HR'], WINDOW_S, STRIDE_S)
    feat = pd.DataFrame(index=idx_ends)
    feat.index.name = 't'
    feat['HR_mean'], feat['HR_slope'], feat['HR_var'] = hr_mean, hr_slope, hr_var

    for track in ['ART_MBP', 'PLETH_SPO2', 'ETCO2']:
        _, m, s, v = fast_rolling_stats(wide[track], WINDOW_S, STRIDE_S)
        feat[f'{track}_mean'], feat[f'{track}_slope'], feat[f'{track}_var'] = m, s, v

    _, a1, a2, se = nonlinear_hr_features(wide['HR'], WINDOW_S, STRIDE_S)
    feat['HR_dfa_alpha1'], feat['HR_dfa_alpha2'], feat['HR_sampen'] = a1, a2, se

    feat['elapsed_time_s'] = feat.index.values

    for track in ['PPF20_CE', 'RFTN20_CE']:
        if track in wide.columns:
            _, m, _, _ = fast_rolling_stats(wide[track], WINDOW_S, STRIDE_S)
            feat[f'{track}_mean'] = m
        else:
            feat[f'{track}_mean'] = np.nan

    feat['caseid'] = cid
    return feat.reset_index(), None


if __name__ == '__main__':
    cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist()
    n_workers = min(cpu_count(), 20)
    print(f"Building features for {len(cohort_ids)} cases using {n_workers} parallel workers...")

    start = time.time()
    all_features = []
    failed = []

    with Pool(n_workers) as pool:
        for i, (feat, err) in enumerate(pool.imap(build_case_features, cohort_ids, chunksize=4)):
            if err is not None:
                failed.append(err)
            elif feat is not None:
                all_features.append(feat)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta_min = (len(cohort_ids) - (i + 1)) / rate / 60
                print(f"[{i+1}/{len(cohort_ids)}] elapsed={elapsed:.0f}s ETA={eta_min:.0f}min")

    full_table = pd.concat(all_features, ignore_index=True)
    full_table.to_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))

    print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
    print(f"Total feature rows: {len(full_table)}")
    print(f"Columns: {full_table.columns.tolist()}")
    if failed:
        print(f"First few failures: {failed[:5]}")