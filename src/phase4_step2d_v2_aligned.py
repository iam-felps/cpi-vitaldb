"""
Phase 4 step 2d (v2) — rebuild all BIS filter variants from raw, ON THE SAME
TIME GRID AS PHASE 4 STEP 1.

Bug fixed (see decision log): v1's load_raw_track_1s() re-zeroed each track on
its OWN first timestamp, while the interim grid (step 1) is zeroed on the
earliest sample across all step-1 tracks. Any track starting later than the
grid (BIS in ~4% of cases, EMG in ~77%) was shifted earlier on the grid by
that offset. v2 zeroes every track on the common grid start (grid_t0).

Built-in verification BEFORE writing anything: EMG is rebuilt from raw with the
v2 loader and must reproduce the step-1 interim EMG column exactly (values AND
missingness pattern). EMG starts after the grid start in most cases, so this
check fails for the v1 loader and passes only for a correctly aligned one. If the match rate is below
99.9% on the check sample, the script stops without touching data/interim/.

Outputs (per case, overwriting only these columns in data/interim/):
  BIS                 primary, SQI>=50 filtered, LOCF<=15s
  BIS_emg40_filtered  EMG<=40 dB, LOCF<=15s
  BIS_emg35_filtered  EMG<=35 dB, LOCF<=15s
  BIS_sqi15_filtered  SQI>=15 (Hypothesis Test 4 comparator), LOCF<=15s
  BIS_unfiltered      BIS=0 nulled only, LOCF<=15s
  SQI                 raw SQI on the grid (diagnostics)
Run on the Phase 3 cohort (3,066) — step 4 case exclusion must be re-run after.
"""

import pandas as pd
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

LOCF_MAX_GAP_S = 15
SQI_THRESHOLD = 50
SQI_STRICT_THRESHOLD = 15
EMG_THRESHOLDS = {'emg40': 40, 'emg35': 35}
N_VERIFY_CASES = 50
MIN_HR_MATCH_RATE = 0.999

# Must be identical to phase4_step1's ALL_TRACKS: defines the common grid start
STEP1_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2', 'BIS', 'EMG', 'PPF20_CE', 'RFTN20_CE']


def grid_t0(raw_df):
    """Common start time of the step-1 grid: earliest non-null sample across step-1 tracks."""
    g = raw_df[raw_df['tname'].isin(STEP1_TRACKS) & raw_df['nval'].notna()]
    return g['dt'].min()


def load_raw_track_1s(raw_df, tname, target_index, t0):
    """Load one raw track on the COMMON grid: second k holds the mean of samples with
    t0 + k <= dt < t0 + k + 1 — the same binning step 1 produced (pivot, zero on t0,
    resample('1s') starting at 0). Binning is done explicitly on integer seconds from t0:
    pandas' resample on a TimedeltaIndex anchors bins on the track's FIRST timestamp, which
    silently misaligns (and, after reindex, blanks) any track not starting exactly on t0."""
    sub = raw_df[(raw_df['tname'] == tname) & raw_df['nval'].notna()]
    if len(sub) == 0:
        return pd.Series(np.nan, index=target_index)
    sec = np.floor(sub['dt'].values - t0).astype(np.int64)
    series = pd.Series(sub['nval'].values).groupby(sec).mean()
    series.index = pd.to_timedelta(series.index, unit='s')
    return series.reindex(target_index)


def locf_bounded(series, max_gap):
    is_na = series.isna()
    group = (~is_na).cumsum()
    run_lengths = is_na.groupby(group).transform('sum')
    fillable = is_na & (run_lengths <= max_gap)
    filled = series.ffill()
    result = series.copy()
    result[fillable] = filled[fillable]
    return result


def verify_alignment(cohort_ids):
    """Rebuild EMG from raw with the v2 loader and require it to reproduce the interim EMG
    column EXACTLY (values and NaN pattern). EMG is the right reference: step 1 placed it on
    the common grid, no later step modifies it, and it starts after the grid start in most
    cases — so the v1 loader fails this check while a correct loader passes it."""
    rng = np.random.RandomState(2311)
    sample = rng.choice(cohort_ids, size=min(N_VERIFY_CASES, len(cohort_ids)), replace=False)
    n_compared, n_match = 0, 0
    for cid in sample:
        raw = pd.read_parquet(os.path.join(RAW_DIR, f"{cid}.parquet"))
        interim = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"))
        emg = load_raw_track_1s(raw, 'EMG', interim.index, grid_t0(raw))
        ref = interim['EMG']
        either = emg.notna() | ref.notna()
        both = emg.notna() & ref.notna()
        n_compared += int(either.sum())
        n_match += int(np.isclose(emg[both], ref[both]).sum())
    rate = n_match / n_compared if n_compared else 0.0
    print(f"Alignment check (EMG rebuilt vs step-1 interim, {len(sample)} cases): "
          f"{n_match}/{n_compared} grid-seconds identical ({rate*100:.3f}%)")
    return rate


if __name__ == '__main__':
    cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
    print(f"Cohort: {len(cohort_ids)} cases")

    rate = verify_alignment(cohort_ids)
    if rate < MIN_HR_MATCH_RATE:
        print(f"STOP: match rate below {MIN_HR_MATCH_RATE*100:.1f}% — loader does not reproduce the step-1 grid. "
              "Nothing was written. Paste this output.")
        sys.exit(1)

    failed = []
    summary = {k: 0 for k in ['sqi50', 'sqi15', 'emg40', 'emg35', 'unfiltered']}
    summary_valid = 0

    for i, cid in enumerate(cohort_ids):
        interim_path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
        try:
            interim = pd.read_parquet(interim_path)
            raw = pd.read_parquet(os.path.join(RAW_DIR, f"{cid}.parquet"))
        except Exception as e:
            failed.append((cid, str(e)))
            continue

        idx = interim.index
        t0 = grid_t0(raw)
        bis_raw = load_raw_track_1s(raw, 'BIS', idx, t0)
        emg_raw = load_raw_track_1s(raw, 'EMG', idx, t0)
        sqi_raw = load_raw_track_1s(raw, 'SQI', idx, t0)

        bis_base = bis_raw.where(bis_raw != 0, np.nan)
        summary_valid += int(bis_base.notna().sum())

        variants = {
            'sqi50': bis_base.where(sqi_raw >= SQI_THRESHOLD, np.nan),
            'sqi15': bis_base.where(sqi_raw >= SQI_STRICT_THRESHOLD, np.nan),
            'emg40': bis_base.where(emg_raw <= EMG_THRESHOLDS['emg40'], np.nan),
            'emg35': bis_base.where(emg_raw <= EMG_THRESHOLDS['emg35'], np.nan),
            'unfiltered': bis_base,
        }
        variants = {k: locf_bounded(v, LOCF_MAX_GAP_S) for k, v in variants.items()}
        for k, v in variants.items():
            summary[k] += int(v.notna().sum())

        interim['BIS'] = variants['sqi50']
        interim['BIS_sqi15_filtered'] = variants['sqi15']
        interim['BIS_emg40_filtered'] = variants['emg40']
        interim['BIS_emg35_filtered'] = variants['emg35']
        interim['BIS_unfiltered'] = variants['unfiltered']
        interim['SQI'] = sqi_raw
        interim = interim.drop(columns=[c for c in ['BIS_emg_unfiltered'] if c in interim.columns])
        interim.to_parquet(interim_path)

        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(cohort_ids)}] done")

    print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
    print(f"Total valid (non-BIS=0) BIS samples: {summary_valid}")
    for k, v in summary.items():
        print(f"  {k}: {v} non-null after LOCF ({v/summary_valid*100:.1f}% of base)")
    if failed:
        print(f"First few failures: {failed[:5]}")