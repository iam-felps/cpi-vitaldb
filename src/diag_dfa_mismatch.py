"""
Diagnose why cached v1 DFA/SampEn values differ from recomputation (read-only).
For the 3 spot-check cases it reports, per feature:
  - rows compared, NaN-pattern disagreements, max / median absolute difference
  - whether recomputing TWICE with the current code gives identical values
    (tests run-to-run determinism: nolds' default DFA fit is RANSAC, which is random)
  - the same with fit_exp='poly' (ordinary least squares, deterministic)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import nolds

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import INTERIM_DIR, PROCESSED_DIR, WINDOW_S, STRIDE_S, FEATURE_MIN_WINDOW_FRAC

CASES = [4245, 2261, 3888]
N1 = np.unique(np.logspace(np.log10(4), np.log10(16), 8).astype(int))
N2 = np.unique(np.logspace(np.log10(16), np.log10(64), 8).astype(int))
COLS = ['HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen']


def compute(hr, fit):
    ends = list(range(WINDOW_S, len(hr), STRIDE_S))
    out = {c: np.full(len(ends), np.nan) for c in COLS}
    for j, e in enumerate(ends):
        v = hr[e - WINDOW_S:e]
        v = v[~np.isnan(v)]
        if len(v) < WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
            continue
        for col, nv in (('HR_dfa_alpha1', N1), ('HR_dfa_alpha2', N2)):
            try:
                out[col][j] = nolds.dfa(v, nvals=nv, order=1, fit_exp=fit)
            except Exception:
                pass
        try:
            sd = np.std(v)
            if sd > 0:
                out['HR_sampen'][j] = nolds.sampen(v, emb_dim=2, tolerance=0.2 * sd)
        except Exception:
            pass
    return pd.DataFrame(out, index=ends)


def compare(a, b, label):
    for c in COLS:
        x, y = a[c], b[c].reindex(a.index)
        nan_mis = int((x.isna() != y.isna()).sum())
        both = x.notna() & y.notna()
        d = (x[both] - y[both]).abs()
        n_diff = int((d > 1e-9).sum())
        print(f"    {label:<34}{c:<15} rows={both.sum():>4}  NaN-pattern diffs={nan_mis:>3}  "
              f"values differing={n_diff:>4}  max|diff|={d.max() if len(d) else 0:.4g}  "
              f"median|diff|={d.median() if len(d) else 0:.4g}")


print(f"nolds version: {getattr(nolds, '__version__', 'unknown')}")
cache = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'),
                        columns=['caseid', 't'] + COLS)
for cid in CASES:
    hr = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"), columns=['HR'])['HR'].to_numpy(float)
    old = cache[cache['caseid'] == cid].set_index('t')[COLS]
    r1 = compute(hr, 'RANSAC')
    r2 = compute(hr, 'RANSAC')
    p1 = compute(hr, 'poly')
    print(f"\n=== case {cid}: cached rows {len(old)}, recomputed rows {len(r1)} ===")
    compare(old, r1, 'cached v1 vs recompute (RANSAC)')
    compare(r1, r2, 'recompute RANSAC run1 vs run2')
    compare(old, p1, 'cached v1 vs recompute (poly)')