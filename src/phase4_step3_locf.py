import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

LOCF_MAX_GAP_S = 15  # primary cutoff; 10s/30s alternates tested in Phase 9
LOCF_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2', 'BIS']  # intraoperative tracks only
# EMG and TCI tracks (PPF20_CE, RFTN20_CE) are not LOCF-imputed here — EMG is an
# artifact-detection signal, not a feature; TCI tracks are drug-infusion-derived
# and behave differently from physiological monitoring signals.

def locf_bounded(series, max_gap):
    """Forward-fill only runs of consecutive NaN with length <= max_gap.
    Longer gaps are left entirely as NaN (not partially filled)."""
    is_na = series.isna()
    # Identify consecutive NaN run groups
    group = (~is_na).cumsum()
    run_lengths = is_na.groupby(group).transform('sum')
    fillable = is_na & (run_lengths <= max_gap)

    filled = series.ffill()
    result = series.copy()
    result[fillable] = filled[fillable]
    return result

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Applying LOCF (max gap {LOCF_MAX_GAP_S}s) to {len(cohort_ids)} cases...")

total_before = {t: 0 for t in LOCF_TRACKS}
total_filled = {t: 0 for t in LOCF_TRACKS}
failed = []

for i, cid in enumerate(cohort_ids):
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    for track in LOCF_TRACKS:
        if track not in wide.columns:
            continue
        na_before = wide[track].isna().sum()
        total_before[track] += na_before

        wide[track] = locf_bounded(wide[track], LOCF_MAX_GAP_S)

        na_after = wide[track].isna().sum()
        total_filled[track] += (na_before - na_after)

    wide.to_parquet(path)  # overwrite interim file, cumulative with steps 1-2

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
print(f"LOCF fill summary (gaps <= {LOCF_MAX_GAP_S}s only):")
for t in LOCF_TRACKS:
    pct = total_filled[t] / total_before[t] * 100 if total_before[t] > 0 else 0
    print(f"  {t}: {total_filled[t]} of {total_before[t]} missing samples filled ({pct:.1f}%)")
if failed:
    print(f"First few failures: {failed[:5]}")