"""
Phase 6 — Labeling (CPI project)

For each feature row (window ending at time t), label positive/negative for
each horizon (5/10/15 min) based on whether a sustained BIS>=60 event (from
Phase 4 step 2e, SQI-primary variant) begins within that horizon after t.

Event-in-progress rule (per proposal Section 2): a window is excluded if the
patient is already within an active (not-yet-sustained OR sustained) BIS>=60
episode at time t itself — preserves the forecasting framing.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')

HORIZONS_S = {'5min': 300, '10min': 600, '15min': 900}

print("Loading features and events...")
features = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))
events = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi_primary.csv'))

final_cohort = set(pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist())
events = events[events['caseid'].isin(final_cohort)].copy()
events['start'] = pd.to_timedelta(events['start']).dt.total_seconds().astype(int)
events['end'] = pd.to_timedelta(events['end']).dt.total_seconds().astype(int)
print(f"Events after aligning to final cohort: {len(events)} across {events['caseid'].nunique()} cases")

events_by_case = events.groupby('caseid')[['start', 'end']].apply(
    lambda g: list(zip(g['start'], g['end']))
).to_dict()

def label_case(case_df, cid):
    case_events = events_by_case.get(cid, [])
    t_vals = case_df['t'].values
    labels = {h: np.zeros(len(t_vals), dtype=object) for h in HORIZONS_S}
    in_progress = np.zeros(len(t_vals), dtype=bool)

    for i, t in enumerate(t_vals):
        for (start, end) in case_events:
            if start <= t <= end:
                in_progress[i] = True
                break
        if in_progress[i]:
            for h in HORIZONS_S:
                labels[h][i] = np.nan
            continue

        for h, horizon_s in HORIZONS_S.items():
            window_end = t + horizon_s
            positive = any(t < start <= window_end for (start, end) in case_events)
            labels[h][i] = 1 if positive else 0

    for h in HORIZONS_S:
        case_df[f'label_{h}'] = labels[h]
    case_df['event_in_progress'] = in_progress
    return case_df

print("Labeling all cases...")
labeled_parts = []
for cid, group in features.groupby('caseid'):
    labeled_parts.append(label_case(group.copy(), cid))

labeled = pd.concat(labeled_parts, ignore_index=True)

n_before = len(labeled)
labeled_clean = labeled[~labeled['event_in_progress']].copy()
n_excluded_inprogress = n_before - len(labeled_clean)

labeled_clean.to_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))

print(f"\nDone. Total rows: {n_before}")
print(f"Excluded (event-in-progress at t): {n_excluded_inprogress}")
print(f"Remaining labeled rows: {len(labeled_clean)}")
for h in HORIZONS_S:
    col = f'label_{h}'
    vc = labeled_clean[col].value_counts(dropna=False)
    pos = vc.get(1, 0)
    neg = vc.get(0, 0)
    print(f"\n{h} horizon: positive={pos} ({pos/(pos+neg)*100:.2f}%), negative={neg}")