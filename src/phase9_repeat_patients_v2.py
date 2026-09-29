"""
Phase 9 (v2) — repeat-patient sensitivity analysis. Entry 58.

Outer folds were assigned per CASE; 60 patients (VitalDB subjectid) contribute >1 case, and
some have cases in different outer folds, so a model may have been trained on another
surgery of the same patient it is tested on. This analysis re-evaluates the SAVED out-of-fold
predictions (no retraining) after removing every test case whose patient has any other case
in a different outer fold (i.e., in that fold's training set).

Per model (A, B, baseline) and horizon: AUROC and AUPRC with 95% bootstrap CIs on all
windows and on the restricted windows, and Model A minus Model B on the restricted windows
(paired). Bootstrap resamples PATIENTS (subjectid), so remaining repeat patients whose cases
share a fold are one cluster.
Output: results/phase9_v2/repeat_patient_sensitivity.csv
Usage: python src/phase9_repeat_patients_v2.py [--n-boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, HORIZONS_S, RANDOM_SEED, CLINICAL_SNAPSHOT
from phase9_tests_subgroups_v2 import FastMetrics, boot_weights, boot_p

P7 = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
TAG = 'sqi_primary_w70'


def ci(y, p, clusters, n_boot):
    fm = FastMetrics(y, p)
    au, ap = fm.compute()
    codes, uniq = pd.factorize(clusters)
    rng = np.random.RandomState(RANDOM_SEED)
    a, b = [], []
    for _ in range(n_boot):
        x, z = fm.compute(boot_weights(codes, len(uniq), rng))
        if not np.isnan(x):
            a.append(x); b.append(z)
    return au, np.percentile(a, 2.5), np.percentile(a, 97.5), ap, np.percentile(b, 2.5), np.percentile(b, 97.5)


def paired_diff(y, p1, p2, clusters, n_boot):
    f1, f2 = FastMetrics(y, p1), FastMetrics(y, p2)
    codes, uniq = pd.factorize(clusters)
    rng = np.random.RandomState(RANDOM_SEED)
    d = []
    for _ in range(n_boot):
        w = boot_weights(codes, len(uniq), rng)
        a1, _ = f1.compute(w)
        a2, _ = f2.compute(w)
        if not (np.isnan(a1) or np.isnan(a2)):
            d.append(a1 - a2)
    return f1.compute()[0] - f2.compute()[0], np.percentile(d, 2.5), np.percentile(d, 97.5), boot_p(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    subj = pd.read_csv(CLINICAL_SNAPSHOT, usecols=['caseid', 'subjectid']).set_index('caseid')['subjectid']
    folds = pd.read_csv(os.path.join(P7, 'outer_folds.csv')).set_index('caseid')['fold']
    cf = pd.DataFrame({'fold': folds, 'subjectid': subj.reindex(folds.index)})
    n_folds_per_subj = cf.groupby('subjectid')['fold'].nunique()
    split_subj = set(n_folds_per_subj[n_folds_per_subj > 1].index)
    drop_cases = set(cf.index[cf['subjectid'].isin(split_subj)])
    print(f"Patients with cases in more than one outer fold: {len(split_subj)}; "
          f"their cases removed from evaluation: {len(drop_cases)} of {len(cf)}")

    rows = []
    for h in HORIZONS_S:
        preds = {m: pd.read_parquet(os.path.join(P7, f'{TAG}_{m}_{h}_predictions.parquet'),
                                    columns=['caseid', 't', 'y_true', 'calibrated_pred']).sort_values(['caseid', 't']).reset_index(drop=True)
                 for m in ('A', 'B', 'baseline')}
        base = preds['A']
        keep = ~base['caseid'].isin(drop_cases).to_numpy()
        cl_all = base['caseid'].map(subj).to_numpy()
        print(f"\n[{h}] windows {len(base):,} -> {int(keep.sum()):,} after removal "
              f"({int(base['y_true'].sum()):,} -> {int(base.loc[keep, 'y_true'].sum()):,} positive)")
        for m, d in preds.items():
            assert (d['caseid'].to_numpy() == base['caseid'].to_numpy()).all()
            y, p = d['y_true'].to_numpy().astype(int), d['calibrated_pred'].to_numpy()
            full = ci(y, p, cl_all, args.n_boot)
            rest = ci(y[keep], p[keep], cl_all[keep], args.n_boot)
            rows.append({'horizon': h, 'model': m, 'n_windows_all': len(y), 'n_windows_restricted': int(keep.sum()),
                         'auroc_all': full[0], 'auroc_all_lo': full[1], 'auroc_all_hi': full[2],
                         'auroc_restricted': rest[0], 'auroc_restricted_lo': rest[1], 'auroc_restricted_hi': rest[2],
                         'auroc_change': rest[0] - full[0],
                         'auprc_all': full[3], 'auprc_restricted': rest[3], 'auprc_restricted_lo': rest[4], 'auprc_restricted_hi': rest[5]})
            print(f"  {m:8s} AUROC all {full[0]:.3f} [{full[1]:.3f}-{full[2]:.3f}] | restricted {rest[0]:.3f} "
                  f"[{rest[1]:.3f}-{rest[2]:.3f}] | change {rest[0]-full[0]:+.4f} | AUPRC {full[3]:.3f} -> {rest[3]:.3f}", flush=True)
        a, b = preds['A'], preds['B']
        d, lo, hi, pv = paired_diff(a['y_true'].to_numpy()[keep].astype(int), a['calibrated_pred'].to_numpy()[keep],
                                    b['calibrated_pred'].to_numpy()[keep], cl_all[keep], args.n_boot)
        rows.append({'horizon': h, 'model': 'A minus B (restricted)', 'auroc_restricted': d,
                     'auroc_restricted_lo': lo, 'auroc_restricted_hi': hi, 'p': pv})
        print(f"  A - B (restricted): {d:+.3f} [{lo:+.3f}, {hi:+.3f}] p={pv:.3g}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'repeat_patient_sensitivity.csv'), index=False)
    print(f"\nSaved {os.path.join(OUT_DIR, 'repeat_patient_sensitivity.csv')}")


if __name__ == '__main__':
    main()