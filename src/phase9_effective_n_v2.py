"""
Phase 9 (v2) — effective (clustering-adjusted) sample size. Supersedes the effective-N part
of entry 38. Entry 55.

Windows from the same case are strongly correlated, so the raw window count overstates the
information in the data. Two estimates, per horizon, on the primary analysis windows:

1. Outcome design effect (Kish 1965; Eldridge et al. 2006 for unequal cluster sizes).
   ICC of the window label within cases by one-way ANOVA (n0 correction for unequal sizes);
     DE_equal   = 1 + (m_bar - 1) * ICC
     DE_unequal = 1 + ((1 + CV^2) * m_bar - 1) * ICC        (CV = SD/mean of cluster sizes)
   n_eff = N / DE; effective positives = positives / DE; effective positives per candidate
   feature (the quantity Riley et al.'s criteria are built on) for Model A and Model B.
2. Empirical design effect of the reported metric: variance of AUROC under the
   patient-level bootstrap divided by its variance under a naive window-level bootstrap
   (windows treated as independent), for Models A and B. N / DE_AUROC is the number of
   independent windows that would give the same precision.
Output: results/phase9_v2/effective_sample_size.csv
Usage: python src/phase9_effective_n_v2.py [--n-boot 500]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, HORIZONS_S, RANDOM_SEED, MODEL_A_FEATURES, MODEL_B_FEATURES
from phase9_tests_subgroups_v2 import FastMetrics, boot_weights

IN_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
TAG = 'sqi_primary_w70'


def anova_icc(y, codes, k):
    N = len(y)
    m = np.bincount(codes, minlength=k).astype(float)
    s = np.bincount(codes, weights=y, minlength=k)
    means = s / m
    grand = y.mean()
    ssb = np.sum(m * (means - grand) ** 2)
    ssw = np.sum((y - means[codes]) ** 2)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(m ** 2) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return max(icc, 0.0), m


def auroc_boot_var(y, p, codes, k, n_boot, level):
    fm = FastMetrics(y, p)
    rng = np.random.RandomState(RANDOM_SEED)
    N = len(y)
    vals = []
    for _ in range(n_boot):
        w = boot_weights(codes, k, rng) if level == 'patient' else np.bincount(rng.randint(0, N, N), minlength=N).astype(float)
        a, _ = fm.compute(w)
        if not np.isnan(a):
            vals.append(a)
    return np.var(vals, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=500)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n_feat = {'A': len(MODEL_A_FEATURES), 'B': len(MODEL_B_FEATURES)}
    rows = []
    for h in HORIZONS_S:
        preds = {m: pd.read_parquet(os.path.join(IN_DIR, f'{TAG}_{m}_{h}_predictions.parquet'),
                                    columns=['caseid', 't', 'y_true', 'calibrated_pred']).sort_values(['caseid', 't'])
                 for m in ('A', 'B')}
        d = preds['A']
        y = d['y_true'].to_numpy(float)
        codes, uniq = pd.factorize(d['caseid'])
        k, N, pos = len(uniq), len(y), int(y.sum())
        icc, m = anova_icc(y, codes, k)
        m_bar, cv = m.mean(), m.std() / m.mean()
        de_eq = 1 + (m_bar - 1) * icc
        de_uneq = 1 + ((1 + cv ** 2) * m_bar - 1) * icc
        r = {'horizon': h, 'n_windows': N, 'n_cases': k, 'n_pos_windows': pos,
             'n_cases_with_pos': int(d.groupby('caseid')['y_true'].max().sum()),
             'windows_per_case_mean': m_bar, 'windows_per_case_cv': cv, 'icc_label': icc,
             'de_equal': de_eq, 'de_unequal': de_uneq,
             'n_eff_windows': N / de_uneq, 'n_eff_pos': pos / de_uneq,
             'eff_pos_per_feature_A': pos / de_uneq / n_feat['A'], 'eff_pos_per_feature_B': pos / de_uneq / n_feat['B']}
        msg = (f"[{h}] N={N} windows, {k} cases ({m_bar:.0f} windows/case, CV {cv:.2f}), {pos} positive windows "
               f"in {r['n_cases_with_pos']} cases | ICC {icc:.3f} | DE {de_eq:.1f} (equal) / {de_uneq:.1f} (unequal) | "
               f"n_eff {N/de_uneq:,.0f} windows, {pos/de_uneq:,.0f} positives "
               f"({r['eff_pos_per_feature_A']:.0f}/feature A, {r['eff_pos_per_feature_B']:.0f}/feature B)")
        print(msg, flush=True)
        for mk in ('A', 'B'):
            dm = preds[mk]
            assert (dm['caseid'].to_numpy() == d['caseid'].to_numpy()).all() and (dm['t'].to_numpy() == d['t'].to_numpy()).all()
            p = dm['calibrated_pred'].to_numpy()
            v_pat = auroc_boot_var(y, p, codes, k, args.n_boot, 'patient')
            v_win = auroc_boot_var(y, p, codes, k, args.n_boot, 'window')
            de_auc = v_pat / v_win
            r.update({f'auroc_se_patient_{mk}': np.sqrt(v_pat), f'auroc_se_window_{mk}': np.sqrt(v_win),
                      f'de_auroc_{mk}': de_auc, f'n_eff_auroc_{mk}': N / de_auc})
            print(f"    Model {mk}: AUROC SE patient-level {np.sqrt(v_pat):.4f} vs naive window-level "
                  f"{np.sqrt(v_win):.4f} -> design effect {de_auc:.1f}, n_eff {N/de_auc:,.0f} windows", flush=True)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'effective_sample_size.csv'), index=False)
    print(f"\nSaved {os.path.join(OUT_DIR, 'effective_sample_size.csv')}")


if __name__ == '__main__':
    main()