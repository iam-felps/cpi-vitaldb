import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(INTERIM_DIR, exist_ok=True)

PLAUSIBILITY_BOUNDS = {
    'HR': (30, 200),
    'ART_MBP': (30, 160),
    'PLETH_SPO2': (50, 100),
    'ETCO2': (10, 80),
}

ALL_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2', 'BIS', 'EMG', 'PPF20_CE', 'RFTN20_CE']

def load_case_wide_1s(caseid):
    """Load a case's raw parquet, resample to a proper 1-second grid."""
    path = os.path.join(RAW_DIR, f"{caseid}.parquet")
    df = pd.read_parquet(path)
    df = df[df['tname'].isin(ALL_TRACKS)]
    if df.empty:
        return None

    wide = df.pivot_table(index='dt', columns='tname', values='nval', aggfunc='first')
    wide = wide.sort_index()

    t0 = wide.index.min()
    wide.index = pd.to_timedelta(wide.index - t0, unit='s')
    wide = wide.resample('1s').mean()

    for col in ALL_TRACKS:
        if col not in wide.columns:
            wide[col] = np.nan

    return wide[ALL_TRACKS]

def apply_plausibility_filter(wide):
    flagged_counts = {}
    for track, (low, high) in PLAUSIBILITY_BOUNDS.items():
        if track in wide.columns:
            mask = (wide[track] < low) | (wide[track] > high)
            flagged_counts[track] = int(mask.sum())
            wide.loc[mask, track] = np.nan
    return wide, flagged_counts

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Processing {len(cohort_ids)} cases...")

total_flagged = {track: 0 for track in PLAUSIBILITY_BOUNDS}
failed = []

for i, cid in enumerate(cohort_ids):
    try:
        wide = load_case_wide_1s(cid)
    except Exception as e:
        failed.append((cid, str(e)))
        continue
    if wide is None:
        failed.append((cid, "empty after track filter"))
        continue

    wide, flagged = apply_plausibility_filter(wide)
    for k, v in flagged.items():
        total_flagged[k] += v

    outpath = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    wide.to_parquet(outpath)

    if (i + 1) % 200 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. Processed {len(cohort_ids) - len(failed)} cases, {len(failed)} failed.")
print("Values flagged as implausible and set to NaN, by track:")
for k, v in total_flagged.items():
    print(f"  {k}: {v}")
if failed:
    print(f"First few failures: {failed[:5]}")