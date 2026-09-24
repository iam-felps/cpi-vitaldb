"""
Diagnostic: how far is each BIS/SQI/EMG track shifted relative to the interim grid?

Phase 4 step 1 builds the interim 1-second grid from ONE common start time
(the earliest sample across all tracks). Phase 4 step 2d (and the SQI<15
builder in the Phase 9 sensitivity scripts) re-loads BIS, SQI and EMG with
load_raw_track_1s(), which re-zeroes each track on ITS OWN first timestamp.
Any track that starts later than the earliest track is therefore shifted
earlier on the grid by exactly that offset.

This script is read-only. It measures that offset per case, per track.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Same track list Phase 4 step 1 used to define the common grid start
STEP1_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2', 'BIS', 'EMG', 'PPF20_CE', 'RFTN20_CE']
CHECK_TRACKS = ['BIS', 'SQI', 'EMG']

cohort = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
final_2665 = set(pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'])
print(f"Checking {len(cohort)} cases...")

rows = []
for i, cid in enumerate(cohort):
    path = os.path.join(RAW_DIR, f"{cid}.parquet")
    if not os.path.exists(path):
        continue
    raw = pd.read_parquet(path)
    grid = raw[raw['tname'].isin(STEP1_TRACKS) & raw['nval'].notna()]
    if grid.empty:
        continue
    grid_t0 = grid['dt'].min()
    rec = {'caseid': cid, 'in_final_2665': cid in final_2665}
    for tr in CHECK_TRACKS:
        s = raw[raw['tname'] == tr]
        # load_raw_track_1s uses the minimum dt of ALL rows for that track (NaN rows included)
        rec[f'{tr}_offset_s'] = (s['dt'].min() - grid_t0) if len(s) else np.nan
    rows.append(rec)
    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort)}]")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, 'diag_track_alignment_offsets.csv'), index=False)

for tr in CHECK_TRACKS:
    col = f'{tr}_offset_s'
    x = df[col].dropna()
    print(f"\n=== {tr}: offset of track start vs grid start (seconds) ===")
    print(x.describe().round(1))
    for thr in [1, 60, 300, 900]:
        print(f"  |offset| > {thr:>4}s: {(x.abs() > thr).sum()} cases ({(x.abs() > thr).mean()*100:.1f}%)")

d = df.dropna(subset=['BIS_offset_s', 'SQI_offset_s'])
print(f"\nBIS vs SQI start mismatch > 1s: {((d['BIS_offset_s'] - d['SQI_offset_s']).abs() > 1).sum()} cases")
print("\nSaved per-case offsets to results/diag_track_alignment_offsets.csv")