"""
Re-verify the Phase 4 headline finding on the ALIGNED interim data (full cohort):
what fraction of genuine BIS>=60 time does each artifact filter remove?

Reference = BIS_unfiltered (BIS=0 nulled only). For every grid-second where
BIS_unfiltered >= 60, check whether each filtered variant nulled it.
Also reports the share of ALL valid BIS seconds each filter removes, so the
two numbers can be compared (a filter that removes little overall but a lot of
BIS>=60 time is targeting the outcome, not artifact). Read-only.
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

VARIANTS = {'SQI<50 (primary)': 'BIS', 'SQI<15': 'BIS_sqi15_filtered',
            'EMG>40': 'BIS_emg40_filtered', 'EMG>35': 'BIS_emg35_filtered'}

cohort = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
n_valid = n_high = 0
removed_all = {k: 0 for k in VARIANTS}
removed_high = {k: 0 for k in VARIANTS}

for i, cid in enumerate(cohort):
    w = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"),
                        columns=['BIS_unfiltered'] + list(VARIANTS.values()))
    valid = w['BIS_unfiltered'].notna()
    high = valid & (w['BIS_unfiltered'] >= 60)
    n_valid += int(valid.sum())
    n_high += int(high.sum())
    for name, col in VARIANTS.items():
        gone = w[col].isna()
        removed_all[name] += int((valid & gone).sum())
        removed_high[name] += int((high & gone).sum())
    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort)}]")

print(f"\nValid BIS seconds: {n_valid:,}   of which BIS>=60: {n_high:,} ({n_high/n_valid*100:.2f}%)")
print(f"{'Filter':<18}{'% of all valid BIS removed':>28}{'% of BIS>=60 time removed':>28}")
rows = []
for name in VARIANTS:
    a = removed_all[name] / n_valid * 100
    h = removed_high[name] / n_high * 100
    print(f"{name:<18}{a:>27.2f}%{h:>27.2f}%")
    rows.append({'filter': name, 'pct_all_valid_removed': a, 'pct_bis60_removed': h})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'phase4_filter_overlap_aligned.csv'), index=False)
print("\nSaved to results/phase4_filter_overlap_aligned.csv")