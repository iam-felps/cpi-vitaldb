import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

MIN_EVENT_DURATION_S = 180  # 3 minutes

def detect_events(bis_series, threshold=60, min_duration=180):
    """Return list of (start_idx, end_idx, duration_s) for maximal runs where
    bis_series is non-missing and >= threshold, duration >= min_duration."""
    valid_high = bis_series.notna() & (bis_series >= threshold)
    group = (~valid_high).cumsum()
    events = []
    for _, idx in valid_high[valid_high].groupby(group).groups.items():
        duration = len(idx)
        if duration >= min_duration:
            events.append({
                'start': idx[0],
                'end': idx[-1],
                'duration_s': duration
            })
    return events

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Detecting BIS>=60 events (>=180s) for {len(cohort_ids)} cases...")

all_events_primary = []
all_events_unfiltered = []
failed = []

for i, cid in enumerate(cohort_ids):
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    if 'BIS' in wide.columns:
        events = detect_events(wide['BIS'], min_duration=MIN_EVENT_DURATION_S)
        for e in events:
            e['caseid'] = cid
            all_events_primary.append(e)

    if 'BIS_emg_unfiltered' in wide.columns:
        events_uf = detect_events(wide['BIS_emg_unfiltered'], min_duration=MIN_EVENT_DURATION_S)
        for e in events_uf:
            e['caseid'] = cid
            all_events_unfiltered.append(e)

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

primary_df = pd.DataFrame(all_events_primary)
unfiltered_df = pd.DataFrame(all_events_unfiltered)

primary_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_events_primary_emgfiltered.csv'), index=False)
unfiltered_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_events_emgunfiltered.csv'), index=False)

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
print(f"Primary (EMG-filtered) events detected: {len(primary_df)}")
print(f"  Cases with >=1 event: {primary_df['caseid'].nunique() if len(primary_df) > 0 else 0} / {len(cohort_ids)}")
print(f"EMG-unfiltered events detected: {len(unfiltered_df)}")
print(f"  Cases with >=1 event: {unfiltered_df['caseid'].nunique() if len(unfiltered_df) > 0 else 0} / {len(cohort_ids)}")
if failed:
    print(f"First few failures: {failed[:5]}")

