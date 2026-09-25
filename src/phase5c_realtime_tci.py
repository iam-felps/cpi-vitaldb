"""
Phase 5c — make the TCI features REAL-TIME (removes a lookahead introduced in Phase 5 v2).
Decision log entry 47.

Problem: Phase 5 v2 set tiva_propofol = 1 for the whole case if propofol TCI data appeared
at ANY time, and left PPF20_CE_mean / RFTN20_CE_mean as NaN (TIVA cases) vs 0 (non-TIVA
cases) before the pump started. Both encode the case's future.

Real-time definitions (at window end t, using only data before t):
  tiva_propofol   1 if any Orchestra/PPF20_CE sample exists before t, else 0
  PPF20_CE_mean   mean of CE samples in the window; 0 if no propofol CE sample has
                  occurred yet in the case (no drug on board); NaN only if the infusion had
                  started but the window has no sample (true missing data)
  RFTN20_CE_mean  same rule for remifentanil
The case-level flag is kept as 'tiva_case' for description only (Table 1), never a model feature.
Edits data/processed/phase5_features_v2.parquet in place; idempotent.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import INTERIM_DIR, PROCESSED_DIR, WINDOW_S

PATH = os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet')


def realtime_ce(arr, ends):
    finite = np.isfinite(arr)
    first = int(np.argmax(finite)) if finite.any() else None
    mean = np.full(len(ends), np.nan)
    started = np.zeros(len(ends), dtype=int)
    for j, e in enumerate(ends):
        if first is None or first >= e:
            mean[j] = 0.0
            continue
        started[j] = 1
        w = arr[e - WINDOW_S:e]
        w = w[np.isfinite(w)]
        if len(w):
            mean[j] = w.mean()
    return mean, started


if __name__ == '__main__':
    table = pd.read_parquet(PATH)
    old_flag = table['tiva_propofol'].copy()
    old_ppf, old_rft = table['PPF20_CE_mean'].copy(), table['RFTN20_CE_mean'].copy()
    table['tiva_case'] = table.groupby('caseid')['tiva_propofol'].transform('max') if 'tiva_case' not in table \
        else table['tiva_case']

    for i, (cid, idx) in enumerate(table.groupby('caseid').groups.items()):
        wide = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"), columns=['PPF20_CE', 'RFTN20_CE'])
        ends = table.loc[idx, 't'].to_numpy()
        ppf_mean, ppf_started = realtime_ce(wide['PPF20_CE'].to_numpy(float), ends)
        rft_mean, _ = realtime_ce(wide['RFTN20_CE'].to_numpy(float), ends)
        table.loc[idx, 'PPF20_CE_mean'] = ppf_mean
        table.loc[idx, 'RFTN20_CE_mean'] = rft_mean
        table.loc[idx, 'tiva_propofol'] = ppf_started
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{table['caseid'].nunique()}]")

    table.to_parquet(PATH)

    tiva_cases = table.groupby('caseid')['tiva_case'].first()
    changed_flag = (old_flag != table['tiva_propofol'])
    print(f"\nTIVA cases (case-level, description only): {int(tiva_cases.sum())} / {len(tiva_cases)}")
    print(f"Windows whose tiva_propofol changed 1 -> 0 (before the pump started): {int(changed_flag.sum())} "
          f"({changed_flag.mean()*100:.2f}%) in {table.loc[changed_flag, 'caseid'].nunique()} cases")
    for name, old in (('PPF20_CE_mean', old_ppf), ('RFTN20_CE_mean', old_rft)):
        new = table[name]
        print(f"{name}: NaN -> 0 in {int((old.isna() & (new == 0)).sum())} windows; "
              f"now NaN in {int(new.isna().sum())} windows ({new.isna().mean()*100:.2f}%)")
    start_rel = []
    for cid, g in table[table['tiva_case'] == 1].groupby('caseid'):
        on = g.loc[g['tiva_propofol'] == 1, 'elapsed_time_s']
        if len(on):
            start_rel.append(on.min() / 60)
    s = pd.Series(start_rel)
    print(f"First window with propofol TCI, minutes after anestart (TIVA cases): median {s.median():.1f}, "
          f"IQR {s.quantile(.25):.1f}-{s.quantile(.75):.1f}, >15 min in {(s > 15).sum()} cases")
    print(f"Saved (in place): {PATH}")