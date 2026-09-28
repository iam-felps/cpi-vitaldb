"""
Phase 9 (v2) — matched negative-sampling secondary analysis (protocol Section 5.4), scored
with the SAVED primary out-of-fold predictions (no refitting). Supersedes entry 42. Entry 53.

Design
  Negatives  one shared set for all horizons and models. Candidate windows: in the primary
             predictions at all three horizons, negative at 15 min (hence at 5 and 10 min),
             and >= 20 min (1,200 s) from the start AND end of every detected sustained
             BIS>=60 event in the case (all events, including those outside anesthesia).
             Per case, up to 2 windows drawn AT RANDOM (seeded), >= 10 min apart. Random
             choice replaces entry 42's farthest-point rule, which for event-free cases always
             picked the first and last window (induction / emergence) and biased elapsed time.
  Positives  all positive windows of the primary design at each horizon.
  Scores     each window's calibrated out-of-fold prediction from the primary run (the SAME
             trained models; protocol 5.4 "re-evaluating the same trained models").
Outputs (results/phase9_v2/): matched_sampling_results.csv (AUROC, AUPRC, patient-level
bootstrap CIs; A vs B paired difference), matched_class_separation.csv (positive vs
negative means and standardized mean differences), matched_negatives.csv.
Usage: python src/phase9_matched_sampling_v2.py [--n-boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, PROCESSED_DIR, HORIZONS_S, RANDOM_SEED
from phase9_tests_subgroups_v2 import FastMetrics, paired_bootstrap, boot_p, boot_weights

IN_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
TAG = 'sqi_primary_w70'
MIN_EVENT_DIST_S, MIN_SEP_S, MAX_NEG = 1200, 600, 2
DIAG = ['elapsed_time_s', 'HR_mean', 'HR_var', 'ART_MBP_mean', 'ART_MBP_slope', 'ETCO2_mean',
        'PPF20_CE_mean', 'RFTN20_CE_mean']


def load(model, h):
    return pd.read_parquet(os.path.join(IN_DIR, f'{TAG}_{model}_{h}_predictions.parquet'),
                           columns=['caseid', 't', 'y_true', 'calibrated_pred'])


def build_negatives():
    keys = None
    for h in HORIZONS_S:
        k = load('A', h)[['caseid', 't', 'y_true']].rename(columns={'y_true': f'y_{h}'})
        keys = k if keys is None else keys.merge(k, on=['caseid', 't'])
    cand = keys[keys['y_15min'] == 0].copy()
    ev = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi_primary.csv'))
    ev['s'] = pd.to_timedelta(ev['start']).dt.total_seconds()
    ev['e'] = pd.to_timedelta(ev['end']).dt.total_seconds()
    ev_by = {c: g[['s', 'e']].to_numpy() for c, g in ev.groupby('caseid')}
    rng = np.random.RandomState(RANDOM_SEED)
    picks = []
    for cid, g in cand.groupby('caseid'):
        t = g['t'].to_numpy()
        ok = np.ones(len(t), bool)
        for s, e in ev_by.get(cid, []):
            ok &= (t < s - MIN_EVENT_DIST_S) | (t > e + MIN_EVENT_DIST_S)
        pool = list(t[ok])
        chosen = []
        while pool and len(chosen) < MAX_NEG:
            x = pool.pop(rng.randint(len(pool)))
            chosen.append(x)
            pool = [p for p in pool if abs(p - x) >= MIN_SEP_S]
        picks += [(cid, x) for x in chosen]
    return pd.DataFrame(picks, columns=['caseid', 't'])


def auc_ci(y, p, cases, n_boot):
    fm = FastMetrics(y, p)
    au, ap = fm.compute()
    rng = np.random.RandomState(RANDOM_SEED)
    codes, uniq = pd.factorize(cases)
    a_s, p_s = [], []
    for _ in range(n_boot):
        a1, p1 = fm.compute(boot_weights(codes, len(uniq), rng))
        if not np.isnan(a1):
            a_s.append(a1); p_s.append(p1)
    return au, np.percentile(a_s, 2.5), np.percentile(a_s, 97.5), ap, np.percentile(p_s, 2.5), np.percentile(p_s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    neg = build_negatives()
    neg.to_csv(os.path.join(OUT_DIR, 'matched_negatives.csv'), index=False)
    print(f"Matched negatives: {len(neg)} windows in {neg['caseid'].nunique()} cases "
          f"(>= {MIN_EVENT_DIST_S//60} min from any event, <= {MAX_NEG}/case, random, >= {MIN_SEP_S//60} min apart)")
    feats = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'), columns=['caseid', 't'] + DIAG)

    res, sep = [], []
    for h in HORIZONS_S:
        sets = {}
        for m in ('A', 'B', 'baseline'):
            d = load(m, h)
            pos = d[d['y_true'] == 1]
            ng = d.merge(neg, on=['caseid', 't'])
            sets[m] = pd.concat([pos, ng], ignore_index=True).sort_values(['caseid', 't']).reset_index(drop=True)
        s = sets['A']
        print(f"\n[{h}] matched set: {int(s['y_true'].sum())} positive windows ({s.loc[s.y_true == 1, 'caseid'].nunique()} cases), "
              f"{int((s['y_true'] == 0).sum())} negative windows")
        for m, d in sets.items():
            au, alo, ahi, apr, plo, phi = auc_ci(d['y_true'].to_numpy(), d['calibrated_pred'].to_numpy(), d['caseid'].to_numpy(), args.n_boot)
            res.append({'horizon': h, 'model': m, 'n_pos': int(d['y_true'].sum()), 'n_neg': int((d['y_true'] == 0).sum()),
                        'auroc': au, 'auroc_ci_lo': alo, 'auroc_ci_hi': ahi, 'auprc': apr, 'auprc_ci_lo': plo, 'auprc_ci_hi': phi})
            print(f"  {m}: AUROC {au:.3f} [{alo:.3f}-{ahi:.3f}]  AUPRC {apr:.3f}")
        a, b = sets['A'], sets['B']
        assert a[['caseid', 't']].equals(b[['caseid', 't']])
        d = paired_bootstrap(a['caseid'].to_numpy(), a['y_true'].to_numpy(), a['calibrated_pred'].to_numpy(),
                             b['y_true'].to_numpy(), b['calibrated_pred'].to_numpy(), args.n_boot)
        res.append({'horizon': h, 'model': 'A minus B', 'auroc': res[-3]['auroc'] - res[-2]['auroc'],
                    'auroc_ci_lo': np.percentile(d['auroc'], 2.5), 'auroc_ci_hi': np.percentile(d['auroc'], 97.5),
                    'auroc_p': boot_p(d['auroc'])})
        print(f"  A - B: dAUROC {res[-1]['auroc']:+.3f} [{res[-1]['auroc_ci_lo']:+.3f}, {res[-1]['auroc_ci_hi']:+.3f}] p={res[-1]['auroc_p']:.3g}")
        f = s[['caseid', 't', 'y_true']].merge(feats, on=['caseid', 't'], how='left')
        for c in DIAG:
            p_, n_ = f.loc[f.y_true == 1, c], f.loc[f.y_true == 0, c]
            sd = np.sqrt((p_.var() + n_.var()) / 2)
            sep.append({'horizon': h, 'feature': c, 'mean_pos': p_.mean(), 'mean_neg': n_.mean(),
                        'diff': p_.mean() - n_.mean(), 'smd': (p_.mean() - n_.mean()) / sd if sd > 0 else np.nan})
        top = sorted([r for r in sep if r['horizon'] == h], key=lambda r: -abs(r['smd']))[:3]
        print("  largest class separation (SMD): " + ", ".join(f"{r['feature']} {r['smd']:+.2f}" for r in top))

    pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, 'matched_sampling_results.csv'), index=False)
    pd.DataFrame(sep).to_csv(os.path.join(OUT_DIR, 'matched_class_separation.csv'), index=False)
    print(f"\nSaved matched_sampling_results.csv, matched_class_separation.csv, matched_negatives.csv to {OUT_DIR}")


if __name__ == '__main__':
    main()