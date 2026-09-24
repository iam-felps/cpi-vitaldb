"""
Phase 6 (v2) — labeling for all five BIS label variants, with the corrected rules.

Inputs:  data/processed/phase5_features_v2.parquet (row keys caseid, t; case timing)
         results/phase4_events_{variant}.csv       (from step 2e after the entry-43 fix)
         data/interim/{caseid}.parquet              (BIS variant columns, for the rules below)
Output:  data/processed/phase6_labels_v2.parquet    (long format: caseid, t, variant, labels)
         results/phase6_label_attrition_v2.csv      (rows removed by each rule)

Rules (decision log entry 45), applied per variant, per window ending at t:
  R1 Analysis period (Mand, 2026-09-24): only windows with anestart <= t and events whose
     onset falls within [anestart, aneend]. BIS>=60 before induction or after anesthesia
     end is not an inadequate-depth event.
  R2 Active episode (protocol Section 2): window excluded if t lies inside a sustained
     event [start, end] OR the variant's BIS at second t is >= 60 (an episode already
     in progress, even if it never reaches 3 min).
  R3 Censoring, per horizon h: label is missing (not negative) if t + h > aneend, or if
     the variant's BIS is present for < MIN_HORIZON_BIS_COVERAGE of seconds (t, t+h].
     Applied to both classes. An unobservable horizon is not evidence of "no event".
  Positive for horizon h: a sustained event (>=180 s) of that variant starts in (t, t+h].
Cases with missing anestart/aneend in the clinical table are excluded and counted.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import PROCESSED_DIR, RESULTS_DIR, INTERIM_DIR, HORIZONS_S

MIN_HORIZON_BIS_COVERAGE = 0.5   # proposed default, see entry 45 — printed impact below
VARIANTS = {
    'sqi_primary': 'BIS',
    'sqi15': 'BIS_sqi15_filtered',
    'emg40': 'BIS_emg40_filtered',
    'emg35': 'BIS_emg35_filtered',
    'unfiltered': 'BIS_unfiltered',
}


def load_events(variant):
    ev = pd.read_csv(os.path.join(RESULTS_DIR, f'phase4_events_{variant}.csv'))
    ev['start'] = pd.to_timedelta(ev['start']).dt.total_seconds().astype(int)
    ev['end'] = pd.to_timedelta(ev['end']).dt.total_seconds().astype(int)
    return {cid: g[['start', 'end']].to_numpy() for cid, g in ev.groupby('caseid')}


def label_case(t, bis, events, ane_start, ane_end):
    """t: window end times (grid s); bis: variant BIS on the 1 s grid; events: (k,2) array."""
    n = len(bis)
    out = {'in_period': t >= ane_start}
    t_safe = np.clip(t, 0, n - 1)
    bis_now_high = (t < n) & (np.nan_to_num(bis[t_safe], nan=0.0) >= 60)
    in_event = np.zeros(len(t), dtype=bool)
    for s, e in events:
        in_event |= (t >= s) & (t <= e)
    out['in_progress'] = in_event | bis_now_high

    present = np.r_[0, np.cumsum(~np.isnan(bis))]          # present[k] = #non-null in [0, k)
    starts = events[:, 0] if len(events) else np.array([], dtype=int)
    starts = starts[(starts >= ane_start) & (starts <= ane_end)]
    for h, hs in HORIZONS_S.items():
        end_h = t + hs
        observable_end = end_h <= min(ane_end, n - 1)
        lo, hi = np.clip(t + 1, 0, n), np.clip(end_h + 1, 0, n)
        coverage = (present[hi] - present[lo]) / hs
        observable = observable_end & (coverage >= MIN_HORIZON_BIS_COVERAGE)
        positive = np.array([np.any((starts > ti) & (starts <= ti + hs)) for ti in t]) if len(starts) \
            else np.zeros(len(t), dtype=bool)
        lab = positive.astype(float)
        lab[~observable] = np.nan
        out[f'label_{h}'] = lab
        out[f'censored_{h}'] = ~observable
    return out


if __name__ == '__main__':
    feat = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'),
                           columns=['caseid', 't', 't_case_s', 'elapsed_time_s', 'aneend_case_s'])
    feat['ane_start_grid'] = feat['t'] - feat['elapsed_time_s']
    feat['ane_end_grid'] = feat['t'] - feat['t_case_s'] + feat['aneend_case_s']
    no_timing = feat.groupby('caseid')[['ane_start_grid', 'ane_end_grid']].first().isna().any(axis=1)
    print(f"Feature rows: {len(feat)}, cases: {feat['caseid'].nunique()}; "
          f"cases excluded for missing anestart/aneend: {int(no_timing.sum())}")
    feat = feat[~feat['caseid'].isin(no_timing[no_timing].index)]

    events = {v: load_events(v) for v in VARIANTS}
    parts, attrition = [], []

    for i, (cid, g) in enumerate(feat.groupby('caseid')):
        wide = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"), columns=list(VARIANTS.values()))
        t = g['t'].to_numpy()
        a0, a1 = g['ane_start_grid'].iloc[0], g['ane_end_grid'].iloc[0]
        for v, col in VARIANTS.items():
            ev = events[v].get(cid, np.empty((0, 2), dtype=int))
            lab = label_case(t, wide[col].to_numpy(dtype=float), ev, a0, a1)
            df = pd.DataFrame({'caseid': cid, 't': t, 'variant': v, **lab})
            parts.append(df)
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{feat['caseid'].nunique()}]")

    labels = pd.concat(parts, ignore_index=True)

    for v in VARIANTS:
        d = labels[labels['variant'] == v]
        n0 = len(d)
        d1 = d[d['in_period']]
        d2 = d1[~d1['in_progress']]
        row = {'variant': v, 'rows_total': n0, 'removed_outside_anesthesia': n0 - len(d1),
               'removed_active_episode': len(d1) - len(d2), 'rows_eligible': len(d2)}
        for h in HORIZONS_S:
            cens = int(d2[f'censored_{h}'].sum())
            pos = int((d2[f'label_{h}'] == 1).sum())
            neg = int((d2[f'label_{h}'] == 0).sum())
            row.update({f'{h}_censored': cens, f'{h}_pos': pos, f'{h}_neg': neg,
                        f'{h}_pos_rate_pct': round(pos / (pos + neg) * 100, 2) if pos + neg else np.nan,
                        f'{h}_cases_with_pos': int(d2.loc[d2[f'label_{h}'] == 1, 'caseid'].nunique())})
        attrition.append(row)

    labels = labels[labels['in_period'] & ~labels['in_progress']].drop(columns=['in_period', 'in_progress'])
    labels.to_parquet(os.path.join(PROCESSED_DIR, 'phase6_labels_v2.parquet'))
    att = pd.DataFrame(attrition)
    att.to_csv(os.path.join(RESULTS_DIR, 'phase6_label_attrition_v2.csv'), index=False)
    pd.set_option('display.width', 250)
    print("\n=== Label attrition and class balance (per variant) ===")
    print(att.T.to_string())
    print(f"\nSaved labels ({len(labels)} rows, long format) and attrition table.")