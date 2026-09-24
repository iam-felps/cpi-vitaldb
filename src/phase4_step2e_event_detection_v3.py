import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

MIN_EVENT_DURATION_S = 180

# All four variants — 'BIS' is primary (SQI-filtered)
VARIANTS = {
    'sqi_primary': 'BIS',
    'sqi15': 'BIS_sqi15_filtered',
    'emg40': 'BIS_emg40_filtered',
    'emg35': 'BIS_emg35_filtered',
    'unfiltered': 'BIS_unfiltered',
}

def detect_events(bis_series, threshold=60, min_duration=180):
    valid_high = bis_series.notna() & (bis_series >= threshold)
    group = (~valid_high).cumsum()
    events = []
    for _, idx in valid_high[valid_high].groupby(group).groups.items():
        duration = len(idx)
        if duration >= min_duration:
            events.append({'start': idx[0], 'end': idx[-1], 'duration_s': duration})
    return events

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Detecting BIS>=60 events (>=180s) across 4 variants for {len(cohort_ids)} cases...")

all_events = {k: [] for k in VARIANTS}
failed = []

for i, cid in enumerate(cohort_ids):
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    for variant_name, col in VARIANTS.items():
        if col not in wide.columns:
            continue
        events = detect_events(wide[col], min_duration=MIN_EVENT_DURATION_S)
        for e in events:
            e['caseid'] = cid
            all_events[variant_name].append(e)

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

print(f"\nDone. {len(cohort_ids) - len(failed)} cases processed, {len(failed)} failed.")
for variant_name in VARIANTS:
    df = pd.DataFrame(all_events[variant_name])
    outpath = os.path.join(RESULTS_DIR, f'phase4_events_{variant_name}.csv')
    df.to_csv(outpath, index=False)
    n_cases_with_event = df['caseid'].nunique() if len(df) > 0 else 0
    print(f"{variant_name}: {len(df)} events, {n_cases_with_event}/{len(cohort_ids)} cases with >=1 event")

if failed:
    print(f"First few failures: {failed[:5]}")