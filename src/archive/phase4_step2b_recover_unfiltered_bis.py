import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

LOCF_MAX_GAP_S = 15

def locf_bounded(series, max_gap):
    is_na = series.isna()
    group = (~is_na).cumsum()
    run_lengths = is_na.groupby(group).transform('sum')
    fillable = is_na & (run_lengths <= max_gap)
    filled = series.ffill()
    result = series.copy()
    result[fillable] = filled[fillable]
    return result

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Reconstructing EMG-unfiltered BIS for {len(cohort_ids)} cases...")

failed = []
for i, cid in enumerate(cohort_ids):
    raw_path = os.path.join(RAW_DIR, f"{cid}.parquet")
    interim_path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        raw = pd.read_parquet(raw_path)
        interim = pd.read_parquet(interim_path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    bis_raw = raw[raw['tname'] == 'BIS'].set_index('dt')['nval']
    if len(bis_raw) == 0:
        continue

    t0 = bis_raw.index.min()
    bis_raw.index = pd.to_timedelta(bis_raw.index - t0, unit='s')
    bis_raw = bis_raw.resample('1s').mean()

    # Only BIS=0 (sensor-off) nulled — NOT EMG-filtered
    bis_raw = bis_raw.where(bis_raw != 0, np.nan)

    # Align to interim's index (same grid, same case start reference)
    bis_raw = bis_raw.reindex(interim.index)

    # Apply the same LOCF rule for consistency with the filtered version
    bis_unfiltered = locf_bounded(bis_raw, LOCF_MAX_GAP_S)

    interim['BIS_emg_unfiltered'] = bis_unfiltered
    interim.to_parquet(interim_path)

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
if failed:
    print(f"First few failures: {failed[:5]}")