import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

EMG_THRESHOLD_DB = 40  # empirically chosen — Phase 9 sensitivity analysis target

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Sanitizing BIS for {len(cohort_ids)} cases...")

total_bis0 = 0
total_emg_flagged = 0
total_bis_valid_before = 0
failed = []

for i, cid in enumerate(cohort_ids):
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    if 'BIS' not in wide.columns:
        continue

    total_bis_valid_before += wide['BIS'].notna().sum()

    # BIS=0 is a known sensor-off/dropout artifact (confirmed in Phase 2 sanity check) — not real physiology
    bis0_mask = wide['BIS'] == 0
    total_bis0 += int(bis0_mask.sum())
    wide.loc[bis0_mask, 'BIS'] = np.nan

    # Flag BIS values with concurrent EMG > 40 dB as artifact-contaminated — set to NaN
    if 'EMG' in wide.columns:
        emg_mask = wide['EMG'] > EMG_THRESHOLD_DB
        total_emg_flagged += int((emg_mask & wide['BIS'].notna()).sum())
        wide.loc[emg_mask, 'BIS'] = np.nan

    wide.to_parquet(path)  # overwrite interim file with sanitized BIS

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
print(f"Total valid BIS samples before sanitization: {total_bis_valid_before}")
print(f"BIS=0 (sensor-off artifact) samples nulled: {total_bis0} ({total_bis0/total_bis_valid_before*100:.2f}%)")
print(f"BIS samples nulled due to EMG > {EMG_THRESHOLD_DB} dB: {total_emg_flagged} ({total_emg_flagged/total_bis_valid_before*100:.2f}%)")
if failed:
    print(f"First few failures: {failed[:5]}")