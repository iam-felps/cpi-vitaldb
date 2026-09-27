"""
Phase 9 (v2) — hypothesis tests, subgroups and phase stratification, from the saved
Phase 7 v2 predictions (no refitting). Decision log entry 49.

All inference is at the PATIENT level (protocol 5.7): patients are resampled with
replacement; every window of a drawn patient comes along. Discrimination metrics use the
calibrated predictions, as in entry 48.

Part 1  Hypothesis tests (paired bootstrap, --n-boot iterations, default 1000)
  Test 1  per horizon: Model A vs Model B (primary confirmatory comparison), A vs naive
          baseline, B vs naive baseline. Same windows, same patients -> paired.
  Test 2  per model (A, B): 5 vs 10, 10 vs 15, 5 vs 15 min. Each horizon keeps its own
          labels; restricted to windows eligible at BOTH horizons (censoring differs).
  Reported: each AUROC/AUPRC, difference, 95% CI of the difference, two-sided bootstrap p
  (2 x smaller tail share, floored at 1/n_boot, so p<0.001 is shown as 0.001).
  Replaces DeLong on pooled windows (entries 31, 33), which ignored within-patient correlation.

Part 2  Subgroups (descriptive, no significance tests, protocol Section 7)
  ASA I, II, III, IV-V (pooled), missing; age <40, 40-65, 65-80, >=80.
  Per subgroup: cases, windows, positives, AUROC and AUPRC with bootstrap CIs, Brier,
  observed/expected ratio (mean outcome / mean predicted risk), calibration slope.

Part 3  Phase-stratified analysis (protocol 5.5): relative time within anesthesia,
  (time - anestart) / (aneend - anestart): early < 0.4, mid 0.4-0.8, late > 0.8.
  Window level: each window by its own relative time -> AUROC, AUPRC, sensitivity at the
  cross-fitted Youden thresholds (entry 48). Event level: each event by its ONSET's
  relative time -> share of events caught (>=1 alarm in the horizon before onset).

Speed: bootstrap uses FastMetrics (one sort, weighted O(n) passes).

Usage: python src/phase9_tests_subgroups_v2.py [--parts 1 2 3] [--n-boot 1000] [--n-boot-sub 500]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, PROCESSED_DIR, HORIZONS_S, RANDOM_SEED

IN_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
HORIZON_PAIRS = [('5min', '10min'), ('10min', '15min'), ('5min', '15min')]


def load_preds(tag, model, h):
    p = os.path.join(IN_DIR, f'{tag}_{model}_{h}_predictions.parquet')
    return pd.read_parquet(p).sort_values(['caseid', 't']).reset_index(drop=True)


def case_index(cases):
    order = np.argsort(cases, kind='stable')
    uniq, starts = np.unique(cases[order], return_index=True)
    ends = np.r_[starts[1:], len(cases)]
    return uniq, {c: order[s:e] for c, s, e in zip(uniq, starts, ends)}


class FastMetrics:
    """AUROC / average precision under per-row weights, after ONE sort.
    Resampling patients with replacement == weighting each window by how many times its
    patient was drawn, so each bootstrap iteration is an O(n) pass (no re-sorting).
    Ties are handled exactly as sklearn does (verified in the test harness)."""

    def __init__(self, y, p):
        self.order = np.argsort(-p, kind='mergesort')
        ps = p[self.order]
        self.ys = y[self.order].astype(float)
        self.ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]   # last index of each tie group

    def compute(self, w=None):
        wt = np.ones(len(self.ys)) if w is None else w[self.order]
        tp = np.cumsum(wt * self.ys)[self.ends]
        fp = np.cumsum(wt * (1 - self.ys))[self.ends]
        P, N = tp[-1], fp[-1]
        if P == 0 or N == 0:
            return np.nan, np.nan
        tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / N]
        auroc = np.trapz(tpr, fpr) if hasattr(np, 'trapz') else np.trapezoid(tpr, fpr)
        prec = tp / np.maximum(tp + fp, 1e-300)
        ap = np.sum(np.diff(np.r_[0, tp / P]) * prec)
        return auroc, ap


def boot_weights(case_codes, n_cases, rng):
    counts = np.bincount(rng.randint(0, n_cases, n_cases), minlength=n_cases)
    return counts[case_codes].astype(float)


def boot_p(d):
    d = np.asarray(d)
    return min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d)))


# ------------------------------------------------------------------ part 1
def paired_bootstrap(cases, y1, p1, y2, p2, n_boot):
    rng = np.random.RandomState(RANDOM_SEED)
    codes, uniq = pd.factorize(cases)
    f1, f2 = FastMetrics(y1, p1), FastMetrics(y2, p2)
    out = {'auroc': [], 'auprc': []}
    for _ in range(n_boot):
        w = boot_weights(codes, len(uniq), rng)
        a1, ap1 = f1.compute(w)
        a2, ap2 = f2.compute(w)
        if np.isnan(a1) or np.isnan(a2):
            continue
        out['auroc'].append(a1 - a2)
        out['auprc'].append(ap1 - ap2)
    return out


def part1(tag, n_boot):
    rows = []
    for h in HORIZONS_S:
        P = {m: load_preds(tag, m, h) for m in ('A', 'B', 'baseline')}
        for m in ('B', 'baseline'):
            assert P['A'][['caseid', 't']].equals(P[m][['caseid', 't']]), f"row mismatch A vs {m} at {h}"
        for m1, m2 in (('A', 'B'), ('A', 'baseline'), ('B', 'baseline')):
            y = P[m1]['y_true'].to_numpy()
            p1, p2 = P[m1]['calibrated_pred'].to_numpy(), P[m2]['calibrated_pred'].to_numpy()
            t0 = time.time()
            d = paired_bootstrap(P[m1]['caseid'].to_numpy(), y, p1, y, p2, n_boot)
            rows.append(summ('Test 1', f'{m1} vs {m2}', h, y, p1, y, p2, d))
            print(f"  Test 1 {h} {m1} vs {m2}: dAUROC {rows[-1]['auroc_diff']:+.4f} "
                  f"[{rows[-1]['auroc_diff_ci_lo']:+.4f}, {rows[-1]['auroc_diff_ci_hi']:+.4f}] p={rows[-1]['auroc_p']:.3g} | "
                  f"dAUPRC {rows[-1]['auprc_diff']:+.4f} p={rows[-1]['auprc_p']:.3g} ({time.time()-t0:.0f}s)", flush=True)
    for m in ('A', 'B'):
        for h1, h2 in HORIZON_PAIRS:
            a, b = load_preds(tag, m, h1), load_preds(tag, m, h2)
            j = a.merge(b, on=['caseid', 't'], suffixes=('_1', '_2'))
            y1, y2 = j['y_true_1'].to_numpy(), j['y_true_2'].to_numpy()
            p1, p2 = j['calibrated_pred_1'].to_numpy(), j['calibrated_pred_2'].to_numpy()
            t0 = time.time()
            d = paired_bootstrap(j['caseid'].to_numpy(), y1, p1, y2, p2, n_boot)
            rows.append(summ('Test 2', f'Model {m}', f'{h1} vs {h2}', y1, p1, y2, p2, d))
            rows[-1]['n_windows_common'] = len(j)
            print(f"  Test 2 Model {m} {h1} vs {h2}: dAUROC {rows[-1]['auroc_diff']:+.4f} "
                  f"[{rows[-1]['auroc_diff_ci_lo']:+.4f}, {rows[-1]['auroc_diff_ci_hi']:+.4f}] p={rows[-1]['auroc_p']:.3g} | "
                  f"dAUPRC {rows[-1]['auprc_diff']:+.4f} p={rows[-1]['auprc_p']:.3g} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'{tag}_paired_tests.csv'), index=False)


def summ(test, comparison, horizon, y1, p1, y2, p2, d):
    r = {'test': test, 'comparison': comparison, 'horizon': horizon, 'n_boot_valid': len(d['auroc'])}
    for met, fn in (('auroc', roc_auc_score), ('auprc', average_precision_score)):
        v1, v2 = fn(y1, p1), fn(y2, p2)
        r.update({f'{met}_1': v1, f'{met}_2': v2, f'{met}_diff': v1 - v2,
                  f'{met}_diff_ci_lo': np.percentile(d[met], 2.5), f'{met}_diff_ci_hi': np.percentile(d[met], 97.5),
                  f'{met}_p': boot_p(d[met])})
    return r


# ------------------------------------------------------------------ part 2
def calib_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0][0])


def group_metrics(g, n_boot, rng_seed):
    y, p = g['y_true'].to_numpy(), g['calibrated_pred'].to_numpy()
    r = {'n_cases': g['caseid'].nunique(), 'n_windows': len(g), 'n_positive': int(y.sum())}
    if y.min() == y.max() or y.sum() < 10:
        r['note'] = 'too few positives for discrimination metrics'
        return r
    r.update({'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p),
              'brier': brier_score_loss(y, p), 'obs_exp_ratio': y.mean() / p.mean(),
              'calibration_slope': calib_slope(y, p)})
    rng = np.random.RandomState(rng_seed)
    codes, uniq = pd.factorize(g['caseid'])
    fm = FastMetrics(y, p)
    au, ap = [], []
    for _ in range(n_boot):
        a1, p1 = fm.compute(boot_weights(codes, len(uniq), rng))
        if not np.isnan(a1):
            au.append(a1)
            ap.append(p1)
    r.update({'auroc_ci_lo': np.percentile(au, 2.5), 'auroc_ci_hi': np.percentile(au, 97.5),
              'auprc_ci_lo': np.percentile(ap, 2.5), 'auprc_ci_hi': np.percentile(ap, 97.5)})
    return r


def part2(tag, n_boot, feats):
    clin = feats.groupby('caseid')[['asa', 'age']].first()
    asa_grp = clin['asa'].map(lambda a: 'missing' if pd.isna(a) else ('IV-V' if a >= 4 else {1: 'I', 2: 'II', 3: 'III'}[int(a)]))
    age_grp = pd.cut(clin['age'], [-np.inf, 40, 65, 80, np.inf], right=False, labels=['<40', '40-65', '65-80', '>=80'])
    rows = []
    for m in ('A', 'B', 'baseline'):
        for h in HORIZONS_S:
            P = load_preds(tag, m, h)
            for var, grp, order in (('ASA', asa_grp, ['I', 'II', 'III', 'IV-V', 'missing']),
                                    ('age', age_grp.astype(str), ['<40', '40-65', '65-80', '>=80'])):
                P['_g'] = P['caseid'].map(grp)
                for level in order:
                    g = P[P['_g'] == level]
                    if len(g) == 0:
                        continue
                    r = {'model': m, 'horizon': h, 'variable': var, 'group': level}
                    r.update(group_metrics(g, n_boot, RANDOM_SEED))
                    rows.append(r)
                    if 'auroc' in r:
                        print(f"  {m}/{h} {var} {level}: cases={r['n_cases']} AUROC {r['auroc']:.3f} "
                              f"[{r['auroc_ci_lo']:.3f}-{r['auroc_ci_hi']:.3f}] O/E {r['obs_exp_ratio']:.2f} "
                              f"slope {r['calibration_slope']:.2f}", flush=True)
                    else:
                        print(f"  {m}/{h} {var} {level}: cases={r['n_cases']} — {r['note']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'{tag}_subgroups.csv'), index=False)


# ------------------------------------------------------------------ part 3
def part3(tag, n_boot, feats):
    thr = pd.read_csv(os.path.join(OUT_DIR, f'{tag}_youden_thresholds.csv'))
    timing = feats[['caseid', 't', 't_case_s', 'elapsed_time_s', 'aneend_case_s']].copy()
    timing['anestart_case_s'] = timing['t_case_s'] - timing['elapsed_time_s']
    timing['offset'] = timing['t_case_s'] - timing['t']              # grid s -> case s
    per_case = timing.groupby('caseid')[['anestart_case_s', 'aneend_case_s', 'offset']].first()
    per_case['dur'] = per_case['aneend_case_s'] - per_case['anestart_case_s']
    ev = pd.read_csv(os.path.join(RESULTS_DIR, f"phase4_events_{tag.rsplit('_w', 1)[0]}.csv"))
    ev['onset'] = pd.to_timedelta(ev['start']).dt.total_seconds()
    ev = ev.join(per_case, on='caseid', how='inner')
    ev['rel'] = (ev['onset'] + ev['offset'] - ev['anestart_case_s']) / ev['dur']
    ev = ev[(ev['rel'] >= 0) & (ev['rel'] <= 1)]
    phase = lambda r: pd.cut(r, [-np.inf, 0.4, 0.8, np.inf], labels=['early', 'mid', 'late'])
    ev['phase'] = phase(ev['rel'])
    print(f"  Events within anesthesia by onset phase: {ev['phase'].value_counts().reindex(['early','mid','late']).to_dict()}")
    rows = []
    for m in ('A', 'B', 'baseline'):
        for h, hs in HORIZONS_S.items():
            P = load_preds(tag, m, h).merge(timing[['caseid', 't', 'elapsed_time_s']], on=['caseid', 't'])
            P['rel'] = P['elapsed_time_s'] / P['caseid'].map(per_case['dur'])
            P['phase'] = phase(P['rel'])
            t_map = thr[(thr['model'] == m) & (thr['horizon'] == h)].set_index('fold')['youden_threshold']
            P['alarm'] = (P['calibrated_pred'] >= P['fold'].map(t_map)).astype(int)
            # event level: alarm in (onset - h, onset) among that case's windows
            e = ev[['caseid', 'onset', 'phase']]
            j = P[['caseid', 't', 'alarm']].merge(e, on='caseid')
            j = j[(j['t'] < j['onset']) & (j['t'] >= j['onset'] - hs)]
            caught = j.groupby(['caseid', 'onset', 'phase'], observed=True)['alarm'].max().reset_index()
            for ph in ['early', 'mid', 'late']:
                g = P[P['phase'] == ph]
                r = {'model': m, 'horizon': h, 'phase': ph}
                r.update(group_metrics(g, n_boot, RANDOM_SEED))
                pos = g['y_true'] == 1
                r['window_sensitivity'] = g.loc[pos, 'alarm'].mean() if pos.any() else np.nan
                c = caught[caught['phase'] == ph]
                r['n_events'] = len(c)
                r['event_sensitivity'] = c['alarm'].mean() if len(c) else np.nan
                rows.append(r)
                print(f"  {m}/{h} {ph}: windows={r['n_windows']} AUROC {r.get('auroc', np.nan):.3f} "
                      f"window sens {r['window_sensitivity']:.3f} | events={r['n_events']} "
                      f"caught {r['event_sensitivity']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'{tag}_phase_stratified.csv'), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='sqi_primary_w70')
    ap.add_argument('--parts', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--n-boot-sub', type=int, default=500)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'),
                            columns=['caseid', 't', 'asa', 'age', 't_case_s', 'elapsed_time_s', 'aneend_case_s'])
    t0 = time.time()
    if 1 in args.parts:
        print("=== Part 1: paired hypothesis tests ===")
        part1(args.tag, args.n_boot)
    if 2 in args.parts:
        print("\n=== Part 2: subgroups (descriptive) ===")
        part2(args.tag, args.n_boot_sub, feats)
    if 3 in args.parts:
        print("\n=== Part 3: phase-stratified ===")
        part3(args.tag, args.n_boot_sub, feats)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Outputs in {OUT_DIR}: "
          f"{args.tag}_paired_tests.csv, _subgroups.csv, _phase_stratified.csv")


if __name__ == '__main__':
    main()