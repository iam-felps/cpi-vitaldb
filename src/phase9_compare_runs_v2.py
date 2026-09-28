"""
Phase 9 (v2) — paired comparisons BETWEEN Phase 7 runs (sensitivity analyses and reference
models), patient-level paired bootstrap on the windows the two runs share. Entry 52.

Each comparison: run 1 (model m1, tag t1) vs run 2 (model m2, tag t2), per horizon.
Windows are matched on (caseid, t); each run keeps its OWN labels (label definitions differ
across artifact-filtering variants) and its own calibrated predictions. Reported: n common
windows/cases, AUROC and AUPRC of each run on the common windows, difference, 95% CI, p.

Built-in comparison sets (--sets, default all):
  test3   primary vs emg40 / emg35 / unfiltered            (Models A, B)
  test4   primary vs sqi15                                  (Models A, B)
  test6   primary (>=70% complete) vs >=90% and >=50%       (Models A, B)
  refs    Model A vs anesthetic course (time + drugs); Model B vs timing only;
          anesthesia and surgery periods
Missing runs are skipped with a message. Output: results/phase9_v2/run_comparisons.csv

Usage: python src/phase9_compare_runs_v2.py [--sets test3 test4 test6 refs] [--n-boot 1000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, HORIZONS_S
from phase9_tests_subgroups_v2 import FastMetrics, paired_bootstrap, boot_p

IN_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
P = 'sqi_primary_w70'
COURSE = 'only-elapsed_time_s-PPF20_CE_mean-RFTN20_CE_mean-tiva_propofol'
TIMING = 'only-elapsed_time_s'


def comparison_sets():
    s = {'test3': [], 'test4': [], 'test6': [], 'refs': []}
    for m in ('A', 'B'):
        for v in ('emg40', 'emg35', 'unfiltered'):
            s['test3'].append((f'Test 3: {v}', m, P, m, f'{v}_w70'))
        s['test4'].append(('Test 4: SQI<15', m, P, m, 'sqi15_w70'))
        for w in ('90', '50'):
            s['test6'].append((f'Test 6: window >={w}%', m, P, m, f'sqi_primary_w{w}'))
    for per in ('', '_surgery'):
        lab = 'surgery' if per else 'anesthesia'
        s['refs'].append((f'Model A vs anesthetic course ({lab})', 'A', P + per, 'A', f'{P}{per}_{COURSE}'))
        s['refs'].append((f'Model B vs timing only ({lab})', 'B', P + per, 'A', f'{P}{per}_{TIMING}'))
    return s


def load(tag, model, h):
    p = os.path.join(IN_DIR, f'{tag}_{model}_{h}_predictions.parquet')
    return pd.read_parquet(p, columns=['caseid', 't', 'y_true', 'calibrated_pred']) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['test3', 'test4', 'test6', 'refs'])
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    sets = comparison_sets()
    rows = []
    for name in args.sets:
        print(f"\n=== {name} ===", flush=True)
        for label, m1, t1, m2, t2 in sets[name]:
            for h in HORIZONS_S:
                a, b = load(t1, m1, h), load(t2, m2, h)
                if a is None or b is None:
                    print(f"  {label} {m1} {h}: run not found — skipped ({t1 if a is None else t2})")
                    continue
                j = a.merge(b, on=['caseid', 't'], suffixes=('_1', '_2'))
                y1, y2 = j['y_true_1'].to_numpy().astype(int), j['y_true_2'].to_numpy().astype(int)
                p1, p2 = j['calibrated_pred_1'].to_numpy(), j['calibrated_pred_2'].to_numpy()
                if len(j) == 0 or y1.min() == y1.max() or y2.min() == y2.max():
                    print(f"  {label} | {m1} {h}: no events (or no windows) in common — skipped")
                    continue
                t0 = time.time()
                au1, ap1 = FastMetrics(y1, p1).compute()
                au2, ap2 = FastMetrics(y2, p2).compute()
                d = paired_bootstrap(j['caseid'].to_numpy(), y1, p1, y2, p2, args.n_boot)
                if len(d['auroc']) < 0.9 * args.n_boot:
                    print(f"  {label} | {m1} {h}: too few valid bootstrap draws — skipped")
                    continue
                r = {'set': name, 'comparison': label, 'model_1': m1, 'tag_1': t1, 'model_2': m2, 'tag_2': t2,
                     'horizon': h, 'n_windows_common': len(j), 'n_cases_common': j['caseid'].nunique(),
                     'share_of_run1_windows': len(j) / len(a), 'labels_identical': bool((y1 == y2).all()),
                     'auroc_1': au1, 'auroc_2': au2, 'auroc_diff': au1 - au2,
                     'auroc_diff_ci_lo': np.percentile(d['auroc'], 2.5), 'auroc_diff_ci_hi': np.percentile(d['auroc'], 97.5),
                     'auroc_p': boot_p(d['auroc']),
                     'auprc_1': ap1, 'auprc_2': ap2, 'auprc_diff': ap1 - ap2,
                     'auprc_diff_ci_lo': np.percentile(d['auprc'], 2.5), 'auprc_diff_ci_hi': np.percentile(d['auprc'], 97.5),
                     'auprc_p': boot_p(d['auprc'])}
                rows.append(r)
                print(f"  {label} | {m1} {h}: {au1:.3f} vs {au2:.3f}, dAUROC {r['auroc_diff']:+.4f} "
                      f"[{r['auroc_diff_ci_lo']:+.4f}, {r['auroc_diff_ci_hi']:+.4f}] p={r['auroc_p']:.3g} | "
                      f"common windows {len(j)} ({r['share_of_run1_windows']*100:.0f}% of run 1) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'run_comparisons.csv'), index=False)
    print(f"\nSaved {len(rows)} comparisons to {os.path.join(OUT_DIR, 'run_comparisons.csv')}")


if __name__ == '__main__':
    main()