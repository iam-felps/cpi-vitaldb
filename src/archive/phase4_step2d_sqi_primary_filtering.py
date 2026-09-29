import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

LOCF_MAX_GAP_S = 15
EMG_THRESHOLDS = {'emg40': 40, 'emg35': 35}
SQI_THRESHOLD = 50  # primary, per Phase 4 validation

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
    """Load one raw track, resample to 1s, align to target_index."""
    series = raw_df[raw_df['tname'] == tname].set_index('dt')['nval']
    if len(series) == 0:
        return pd.Series(np.nan, index=target_index)
    t0 = series.index.min()
    series.index = pd.to_timedelta(series.index - t0, unit='s')
    series = series.resample('1s').mean()
    return series.reindex(target_index)

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Rebuilding BIS filter variants for {len(cohort_ids)} cases...")

failed = []
summary = {'sqi50': 0, 'emg40': 0, 'emg35': 0, 'unfiltered': 0}
summary_valid = 0

for i, cid in enumerate(cohort_ids):
    interim_path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    raw_path = os.path.join(RAW_DIR, f"{cid}.parquet")
    try:
        interim = pd.read_parquet(interim_path)
        raw = pd.read_parquet(raw_path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    target_index = interim.index

    bis_raw = load_raw_track_1s(raw, 'BIS', target_index)
    emg_raw = load_raw_track_1s(raw, 'EMG', target_index)
    sqi_raw = load_raw_track_1s(raw, 'SQI', target_index)

    # BIS=0 sensor-off artifact nulled in every variant
    bis_base = bis_raw.where(bis_raw != 0, np.nan)
    summary_valid += bis_base.notna().sum()

    # Variant 1: SQI-filtered (PRIMARY)
    bis_sqi = bis_base.where(sqi_raw >= SQI_THRESHOLD, np.nan)
    bis_sqi = locf_bounded(bis_sqi, LOCF_MAX_GAP_S)
    summary['sqi50'] += bis_sqi.notna().sum()

    # Variant 2 & 3: EMG-filtered (40dB primary-original, 35dB Kim et al.)
    bis_emg40 = bis_base.where(emg_raw <= EMG_THRESHOLDS['emg40'], np.nan)
    bis_emg40 = locf_bounded(bis_emg40, LOCF_MAX_GAP_S)
    summary['emg40'] += bis_emg40.notna().sum()

    bis_emg35 = bis_base.where(emg_raw <= EMG_THRESHOLDS['emg35'], np.nan)
    bis_emg35 = locf_bounded(bis_emg35, LOCF_MAX_GAP_S)
    summary['emg35'] += bis_emg35.notna().sum()

    # Variant 4: fully unfiltered (BIS=0 nulled only)
    bis_unfiltered = locf_bounded(bis_base, LOCF_MAX_GAP_S)
    summary['unfiltered'] += bis_unfiltered.notna().sum()

    # Write all variants; 'BIS' becomes the primary (SQI-filtered) column
    interim['BIS'] = bis_sqi
    interim['BIS_emg40_filtered'] = bis_emg40
    interim['BIS_emg35_filtered'] = bis_emg35
    interim['BIS_unfiltered'] = bis_unfiltered
    interim['SQI'] = sqi_raw
    if 'BIS_emg_unfiltered' in interim.columns:
        interim = interim.drop(columns=['BIS_emg_unfiltered'])  # superseded by BIS_unfiltered

    interim.to_parquet(interim_path)

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
print(f"Total valid (non-BIS=0) BIS samples: {summary_valid}")
for k, v in summary.items():
    print(f"  {k}: {v} non-null after LOCF ({v/summary_valid*100:.1f}% of base)")
if failed:
    print(f"First few failures: {failed[:5]}")