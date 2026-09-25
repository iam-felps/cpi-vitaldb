"""
Phase 9 (v2) — diagnostic, calibration and clinical-utility metrics from the saved
Phase 7 v2 predictions (no refitting). Decision log entry 48.

Operating point: Youden index (max sensitivity + specificity - 1; Mand, 2026-09-24),
CROSS-FITTED: the threshold applied to outer fold k is chosen on the pooled calibrated
predictions of the other four folds, so no patient influences their own threshold.
Thresholds use calibrated predictions (a common probability scale across folds).

Per model (A, B, baseline) x horizon, with 1,000-iteration patient-level bootstrap 95% CIs
(patients resampled with replacement; thresholds held fixed at their cross-fitted values):
  Discrimination  AUROC, AUPRC (pooled calibrated predictions)
  Threshold       sensitivity, specificity, PPV, NPV, F1, balanced accuracy, LR+, LR-,
                  confusion-matrix counts (point estimate)
  Clinical        alarms per hour of eligible anesthesia time (windows are 60 s apart);
                  event-level sensitivity = share of BIS>=60 events with >=1 alarm in the
                  horizon before onset; lead time = minutes from first alarm to onset
                  (median, IQR; detected events only)
  Calibration     Brier, ECE (10 equal-width bins), calibration intercept and slope
                  (logistic recalibration on logit(p)); reliability diagram by deciles of
                  predicted risk (data + figure)
  Utility         decision curve (window-level net benefit vs treat-all / treat-none,
                  thresholds 0.5%-40%) + figure
Also reports the per-fold mean of raw-score AUROC (each fold's own model) as a check.

Usage: python src/phase9_diagnostic_metrics_v2.py [--tag sqi_primary_w70] [--n-boot 1000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, HORIZONS_S, RANDOM_SEED, STRIDE_S

IN_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
MODELS = ['A', 'B', 'baseline']
DCA_THRESHOLDS = np.round(np.arange(0.005, 0.4001, 0.005), 3)


# ------------------------------------------------------------------ thresholds
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best = np.flatnonzero(j == j.max())
    return float(thr[best[len(best) // 2]])     # middle of tied optima (isotonic plateaus)


def crossfit_thresholds(df):
    out = {}
    for k in sorted(df['fold'].unique()):
        other = df[df['fold'] != k]
        out[k] = youden_threshold(other['y_true'].to_numpy(), other['calibrated_pred'].to_numpy())
    return out


# ------------------------------------------------------------------ events
def load_event_onsets(variant):
    ev = pd.read_csv(os.path.join(RESULTS_DIR, f'phase4_events_{variant}.csv'))
    ev['start'] = pd.to_timedelta(ev['start']).dt.total_seconds().astype(int)
    return {c: np.sort(g['start'].to_numpy()) for c, g in ev.groupby('caseid')}


def attach_event_ids(df, onsets, h_s):
    """For positive windows: onset of the event they precede (first onset in (t, t+h])."""
    ev_onset = np.full(len(df), np.nan)
    pos = np.flatnonzero(df['y_true'].to_numpy() == 1)
    cids, ts = df['caseid'].to_numpy(), df['t'].to_numpy()
    for i in pos:
        s = onsets.get(cids[i])
        if s is None:
            continue
        j = np.searchsorted(s, ts[i], side='right')
        if j < len(s) and s[j] <= ts[i] + h_s:
            ev_onset[i] = s[j]
    return ev_onset


# ------------------------------------------------------------------ metrics
def calib_int_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope_m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    slope = float(slope_m.coef_[0][0])
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model), via Newton steps
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp.ravel())))
        g, hss = np.sum(y - mu), np.sum(mu * (1 - mu))
        a += g / hss
        if abs(g / hss) < 1e-10:
            break
    return a, slope


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def threshold_metrics(y, alarm):
    tp = int(((alarm == 1) & (y == 1)).sum()); fn = int(((alarm == 0) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum()); tn = int(((alarm == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if ppv and sens and not np.isnan(ppv) else np.nan
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'sensitivity': sens, 'specificity': spec,
            'ppv': ppv, 'npv': npv, 'f1': f1, 'balanced_accuracy': (sens + spec) / 2,
            'lr_pos': sens / (1 - spec) if spec < 1 else np.inf,
            'lr_neg': (1 - sens) / spec if spec > 0 else np.nan,
            'alarms_per_hour': alarm.mean() * 3600 / STRIDE_S}


def event_metrics(ev_key, alarm):
    """ev_key: unique id per (case, event) for positive windows, NaN otherwise."""
    m = ~np.isnan(ev_key)
    if not m.any():
        return {'n_events': 0, 'event_sensitivity': np.nan}
    detected = pd.Series(alarm[m]).groupby(ev_key[m]).max()
    return {'n_events': int(len(detected)), 'event_sensitivity': float(detected.mean())}


def all_metrics(y, p, alarm, ev_key, t):
    out = {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p),
           'brier': brier_score_loss(y, p), 'ece': ece(y, p), 'prevalence': y.mean()}
    out.update(threshold_metrics(y, alarm))
    out.update(event_metrics(ev_key, alarm))
    return out


BOOT_METRICS = ['auroc', 'auprc', 'brier', 'ece', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1',
                'balanced_accuracy', 'lr_pos', 'lr_neg', 'alarms_per_hour', 'event_sensitivity',
                'lead_time_median_min']


# ------------------------------------------------------------------ figures
def reliability_table(y, p, bins=10):
    """Deciles of predicted risk (equal-count groups): informative for rare outcomes,
    where equal-width bins would put almost all windows in the lowest bin."""
    grp = pd.qcut(pd.Series(p).rank(method='first'), bins, labels=False)
    d = pd.DataFrame({'g': grp, 'y': y, 'p': p}).groupby('g')
    return pd.DataFrame({'bin': d.size().index, 'n': d.size().values,
                         'mean_pred': d['p'].mean().values, 'obs_rate': d['y'].mean().values})


def decision_curve(y, p):
    n, prev = len(y), y.mean()
    rows = []
    for th in DCA_THRESHOLDS:
        a = p >= th
        tp, fp = ((a) & (y == 1)).sum(), ((a) & (y == 0)).sum()
        w = th / (1 - th)
        rows.append({'threshold': th, 'net_benefit_model': tp / n - fp / n * w,
                     'net_benefit_treat_all': prev - (1 - prev) * w, 'net_benefit_treat_none': 0.0})
    return pd.DataFrame(rows)


def plot_figures(rel, dca, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for h in HORIZONS_S:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for mdl in MODELS:
            r = rel[(rel['model'] == mdl) & (rel['horizon'] == h)]
            if len(r):
                ax[0].plot(r['mean_pred'], r['obs_rate'], marker='o', label=f'Model {mdl}' if mdl != 'baseline' else 'Naive baseline')
            d = dca[(dca['model'] == mdl) & (dca['horizon'] == h)]
            if len(d):
                ax[1].plot(d['threshold'], d['net_benefit_model'], label=f'Model {mdl}' if mdl != 'baseline' else 'Naive baseline')
        d0 = dca[dca['horizon'] == h]
        if len(d0):
            d0 = d0[d0['model'] == d0['model'].iloc[0]]
            ax[1].plot(d0['threshold'], d0['net_benefit_treat_all'], 'k--', lw=1, label='Treat all')
            ax[1].axhline(0, color='grey', lw=1, label='Treat none')
            ymax = max(dca.loc[dca['horizon'] == h, 'net_benefit_model'].max(), 0.001)
            ax[1].set_ylim(-ymax * 0.2, ymax * 1.2)
        lim = max(rel.loc[rel['horizon'] == h, ['mean_pred', 'obs_rate']].max().max(), 0.05) * 1.05 if len(rel) else 1
        ax[0].plot([0, lim], [0, lim], color='grey', lw=1)
        ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel='Mean predicted probability', ylabel='Observed event rate',
                  title=f'Reliability diagram, {h} horizon')
        ax[1].set(xlabel='Threshold probability', ylabel='Net benefit', title=f'Decision curve, {h} horizon')
        for a in ax:
            a.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'{tag}_calibration_dca_{h}.png'), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='sqi_primary_w70')
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    variant = args.tag.rsplit('_w', 1)[0]
    onsets = load_event_onsets(variant)

    summary, thr_rows, rel_all, dca_all = [], [], [], []
    for mdl in MODELS:
        for h, h_s in HORIZONS_S.items():
            path = os.path.join(IN_DIR, f'{args.tag}_{mdl}_{h}_predictions.parquet')
            if not os.path.exists(path):
                print(f"[{mdl}/{h}] predictions not found — skipped")
                continue
            df = pd.read_parquet(path).sort_values(['caseid', 't']).reset_index(drop=True)
            thr = crossfit_thresholds(df)
            for k, v in thr.items():
                thr_rows.append({'model': mdl, 'horizon': h, 'fold': k, 'youden_threshold': v})
            y = df['y_true'].to_numpy().astype(int)
            p = df['calibrated_pred'].to_numpy()
            alarm = (p >= df['fold'].map(thr).to_numpy()).astype(int)
            t = df['t'].to_numpy().astype(float)
            ev = attach_event_ids(df, onsets, h_s)
            case_code = pd.factorize(df['caseid'])[0].astype(float)
            ev_key = np.where(np.isnan(ev), np.nan, case_code * 1e6 + ev)   # unique id per case+onset
            point = all_metrics(y, p, alarm, ev_key, t)
            point['lead_time_median_min'], point['lead_time_q1_min'], point['lead_time_q3_min'] = \
                lead_time_stats(ev, alarm, t, case_code)
            per_fold_raw = np.mean([roc_auc_score(g['y_true'], g['raw_pred']) for _, g in df.groupby('fold')])
            print(f"[{mdl}/{h}] n={len(df)} events={point['n_events']} | bootstrapping ({args.n_boot})...", flush=True)
            ci = bootstrap_with_lead(y, p, alarm, ev, t, case_code, df['caseid'].to_numpy(), args.n_boot)
            row = {'model': mdl, 'horizon': h, 'n_windows': len(df), 'n_cases': df['caseid'].nunique(),
                   'auroc_perfold_mean_raw': per_fold_raw,
                   'youden_thresholds': ';'.join(f'{thr[k]:.4f}' for k in sorted(thr))}
            for k, v in point.items():
                row[k] = v
                if k in ci:
                    row[f'{k}_ci_lo'], row[f'{k}_ci_hi'] = ci[k]
            row['calibration_intercept'], row['calibration_slope'] = calib_int_slope(y, p)
            summary.append(row)
            r = reliability_table(y, p); r['model'], r['horizon'] = mdl, h; rel_all.append(r)
            d = decision_curve(y, p); d['model'], d['horizon'] = mdl, h; dca_all.append(d)
            print(f"   AUROC {row['auroc']:.3f} [{row['auroc_ci_lo']:.3f}-{row['auroc_ci_hi']:.3f}] | "
                  f"sens {row['sensitivity']:.3f} spec {row['specificity']:.3f} PPV {row['ppv']:.3f} "
                  f"NPV {row['npv']:.4f} | alarms/h {row['alarms_per_hour']:.2f} | "
                  f"event sens {row['event_sensitivity']:.3f} lead {row['lead_time_median_min']:.1f} min | "
                  f"cal slope {row['calibration_slope']:.2f}", flush=True)

    res = pd.DataFrame(summary)
    res.to_csv(os.path.join(OUT_DIR, f'{args.tag}_diagnostic_metrics.csv'), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUT_DIR, f'{args.tag}_youden_thresholds.csv'), index=False)
    rel, dca = pd.concat(rel_all), pd.concat(dca_all)
    rel.to_csv(os.path.join(OUT_DIR, f'{args.tag}_reliability.csv'), index=False)
    dca.to_csv(os.path.join(OUT_DIR, f'{args.tag}_decision_curve.csv'), index=False)
    plot_figures(rel, dca, args.tag)
    print(f"\nSaved to {OUT_DIR}: diagnostic_metrics.csv, youden_thresholds.csv, reliability.csv, "
          f"decision_curve.csv, calibration_dca_<horizon>.png")


def lead_time_stats(ev, alarm, t, case_code):
    m = ~np.isnan(ev) & (alarm == 1)
    if not m.any():
        return np.nan, np.nan, np.nan
    d = pd.DataFrame({'c': case_code[m], 'ev': ev[m], 't': t[m]})
    first = d.groupby(['c', 'ev'])['t'].min().reset_index()
    lead = (first['ev'] - first['t']).to_numpy() / 60
    return float(np.median(lead)), float(np.percentile(lead, 25)), float(np.percentile(lead, 75))


def bootstrap_with_lead(y, p, alarm, ev, t, case_code, cases, n_boot):
    rng = np.random.RandomState(RANDOM_SEED)
    uniq = np.unique(cases)
    order = np.argsort(cases, kind='stable')
    bounds = np.searchsorted(cases[order], uniq)
    ends = np.r_[bounds[1:], len(cases)]
    idx_by_case = {c: order[s:e] for c, s, e in zip(uniq, bounds, ends)}
    vals = {m: [] for m in BOOT_METRICS}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        parts = [idx_by_case[c] for c in pick]
        idx = np.concatenate(parts)
        draw = np.repeat(np.arange(len(pick)), [len(q) for q in parts]).astype(float)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ek = np.where(np.isnan(ev[idx]), np.nan, draw * 1e6 + ev[idx])
        m = all_metrics(yy, p[idx], alarm[idx], ek, t[idx])
        m['lead_time_median_min'] = lead_time_stats(ev[idx], alarm[idx], t[idx], draw)[0]
        for k in BOOT_METRICS:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


if __name__ == '__main__':
    main()