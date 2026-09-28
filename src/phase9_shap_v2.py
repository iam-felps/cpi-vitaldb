"""
Phase 9 (v2) — SHAP explanations of the final Models A and B, out-of-fold. Entry 54.

For each model (A, B), horizon and outer fold k:
  1. Rebuild the Phase 7 data exactly (same labels, window rule, sort order) and check that
     the rows match the saved predictions.
  2. Refit the configuration Phase 7 selected for fold k (library + hyper-parameters from
     {tag}_{model}_{h}_config.csv) on the outer-training cases with the same preprocessing
     (MICE fitted on the training fold only). Reproducibility check: the refitted model's
     predictions on the outer-test cases are compared with the saved raw predictions.
  3. Exact TreeSHAP values (native implementations: LightGBM pred_contrib, XGBoost
     pred_contribs, CatBoost ShapValues; log-odds scale) for a random sample of the
     outer-TEST windows of fold k, so every explained window is scored by a model that never
     saw its case. Additivity check: sum(SHAP) + expected value = model log-odds.
SHAP explains the uncalibrated model; calibration is a monotone map applied afterwards and
does not change feature rankings.

Outputs (results/phase9_v2/):
  shap_{tag}_{model}_{h}.parquet   caseid, t, fold, y_true, feature values, shap_<feature>
  shap_importance.csv              mean |SHAP| per feature (pooled; per-fold; rank; rank range)
  shap_consistency.csv             across horizons (same model) and across folds:
                                   Spearman of mean |SHAP|, Jaccard of top-5 and top-10 sets
  shap_checks.csv                  refit reproducibility and additivity per fold
  shap_bar_*.png, shap_beeswarm_*.png, shap_dependence_*.png
Usage: python src/phase9_shap_v2.py [--models A B] [--horizons 5min 10min 15min]
                                    [--n-sample 2000] [--period anesthesia|surgery]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED, CORE_TRACKS, HORIZONS_S, PRIMARY_MIN_WINDOW_FRAC
from phase7_cv_pipeline_v2 import MODEL_SPECS, Prep, make_model, FOLDS_FILE, N_OUTER, OUT_DIR as P7_DIR
from cpi_config import MODEL_A_FEATURES, BASELINE_FEATURES

OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
DEPENDENCE = ['elapsed_time_s', 'PPF20_CE_mean', 'RFTN20_CE_mean']


def build_base(variant, min_frac, period):
    """Identical to phase7_cv_pipeline_v2.main() data construction."""
    labels = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labels_v2.parquet' if period == 'anesthesia'
                                          else 'phase6_labels_v2_surgery.parquet'))
    labels = labels[labels['variant'] == variant]
    feat_cols = sorted(set(MODEL_A_FEATURES) | set(BASELINE_FEATURES) | {f'{t}_frac' for t in CORE_TRACKS})
    feats = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'), columns=['caseid', 't'] + feat_cols)
    base = labels.merge(feats, on=['caseid', 't'], how='inner')
    ok = np.logical_and.reduce([base[f'{t}_frac'] >= min_frac for t in CORE_TRACKS])
    return base[ok].reset_index(drop=True)


def tree_shap(model, lib, X):
    """Returns (n, F) SHAP values and (n,) expected value, log-odds scale."""
    if lib == 'lightgbm':
        c = model.predict(X, pred_contrib=True)
    elif lib == 'xgboost':
        import xgboost as xgb
        c = model.get_booster().predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
    elif lib == 'catboost':
        from catboost import Pool
        c = model.get_feature_importance(Pool(X), type='ShapValues')
    else:
        raise ValueError(lib)
    c = np.asarray(c, dtype=float)
    return c[:, :-1], c[:, -1]


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def explain(base, model_key, h, tag, folds, n_sample, n_jobs):
    spec = MODEL_SPECS[model_key]
    feats = spec['features']
    data = base[base[f'label_{h}'].notna()].sort_values(['caseid', 't']).reset_index(drop=True)
    saved = pd.read_parquet(os.path.join(P7_DIR, f'{tag}_{model_key}_{h}_predictions.parquet'))
    cfg = pd.read_csv(os.path.join(P7_DIR, f'{tag}_{model_key}_{h}_config.csv'))
    saved = saved.sort_values(['caseid', 't']).reset_index(drop=True)
    if not (len(saved) == len(data) and (saved['caseid'].to_numpy() == data['caseid'].to_numpy()).all()
            and (saved['t'].to_numpy() == data['t'].to_numpy()).all()):
        raise RuntimeError(f"[{model_key}/{h}] rebuilt data does not match the saved Phase 7 predictions "
                           f"({len(data)} vs {len(saved)} rows) — check inputs/tag")
    y = data[f'label_{h}'].to_numpy().astype(int)
    fold_all = folds.reindex(data['caseid']).to_numpy()
    rng = np.random.RandomState(RANDOM_SEED)
    parts, checks = [], []
    for k in range(N_OUTER):
        t0 = time.time()
        row = cfg[cfg['fold'] == k].iloc[0]
        lib, params = row['library'], json.loads(row['params'])
        tr, te = np.where(fold_all != k)[0], np.where(fold_all == k)[0]
        dtr, dte = data.iloc[tr], data.iloc[te]
        prep = Prep(feats, spec['mice'], spec['linear']).fit(dtr)
        spw = float((y[tr] == 0).sum()) / max(float((y[tr] == 1).sum()), 1.0)
        m = make_model(lib, params, spw, n_jobs)
        Xtr, Xte = prep.transform(dtr), prep.transform(dte)
        m.fit(Xtr, y[tr])
        p_te = m.predict_proba(Xte)[:, 1]
        repro = float(np.max(np.abs(p_te - saved['raw_pred'].to_numpy()[te])))

        pick = np.sort(rng.choice(len(te), size=min(n_sample, len(te)), replace=False))
        sv, ev = tree_shap(m, lib, Xte[pick])
        additivity = float(np.max(np.abs(sv.sum(1) + ev - logit(p_te[pick]))))
        out = pd.DataFrame(Xte[pick], columns=feats)
        out.insert(0, 'y_true', y[te][pick])
        out.insert(0, 'fold', k)
        out.insert(0, 't', dte['t'].to_numpy()[pick])
        out.insert(0, 'caseid', dte['caseid'].to_numpy()[pick])
        for j, f in enumerate(feats):
            out[f'shap_{f}'] = sv[:, j]
        parts.append(out)
        checks.append({'model': model_key, 'horizon': h, 'fold': k, 'library': lib, 'params': row['params'],
                       'n_explained': len(pick), 'n_pos_explained': int(y[te][pick].sum()),
                       'refit_max_abs_diff_vs_saved': repro, 'additivity_max_abs_err_logodds': additivity,
                       'seconds': round(time.time() - t0)})
        flag = '' if repro < 1e-4 else '  <-- refit differs from saved model, see note'
        print(f"  [{model_key}/{h}] fold {k}: {lib} {params} | explained {len(pick)} windows "
              f"({checks[-1]['n_pos_explained']} pos) | refit diff {repro:.2e} | additivity {additivity:.1e} "
              f"({checks[-1]['seconds']}s){flag}", flush=True)
    return pd.concat(parts, ignore_index=True), checks


def importance(df, feats):
    pooled = pd.Series({f: df[f'shap_{f}'].abs().mean() for f in feats})
    per_fold = pd.DataFrame({k: {f: g[f'shap_{f}'].abs().mean() for f in feats} for k, g in df.groupby('fold')})
    ranks = per_fold.rank(ascending=False, method='min')
    return pooled, per_fold, ranks


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def compare(s1, s2):
    common = s1.index.intersection(s2.index)
    r = spearmanr(s1[common], s2[common]).correlation
    t1, t2 = s1[common].sort_values(ascending=False), s2[common].sort_values(ascending=False)
    return {'spearman': r, 'jaccard_top5': jaccard(t1.index[:5], t2.index[:5]),
            'jaccard_top10': jaccard(t1.index[:10], t2.index[:10])}


def plots(df, feats, pooled, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = pooled.sort_values(ascending=False).index[:20][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1.2))
    ax.barh(range(len(order)), pooled[order], color='#3b6ea5')
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel('mean |SHAP| (log-odds)')
    ax.set_title(label)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'shap_bar_{label}.png'), dpi=150); plt.close(fig)

    rng = np.random.RandomState(RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(order) + 1.2))
    for i, f in enumerate(order):
        v, x = df[f'shap_{f}'].to_numpy(), df[f].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(x, 5) if np.isfinite(x).any() else 0, np.nanpercentile(x, 95) if np.isfinite(x).any() else 1
        c = np.clip((x - lo) / (hi - lo if hi > lo else 1), 0, 1)
        nan = ~np.isfinite(c)
        yj = i + rng.uniform(-0.3, 0.3, len(v))
        ax.scatter(v[nan], yj[nan], s=3, c='#aaaaaa', alpha=0.5, linewidths=0)
        sc = ax.scatter(v[~nan], yj[~nan], s=3, c=c[~nan], cmap='coolwarm', vmin=0, vmax=1, alpha=0.6, linewidths=0)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel('SHAP value (log-odds); colour = feature value (low→high, grey = missing)')
    ax.set_title(label)
    fig.colorbar(sc, ax=ax, pad=0.01, ticks=[0, 1]).ax.set_yticklabels(['low', 'high'])
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'shap_beeswarm_{label}.png'), dpi=150); plt.close(fig)

    for f in [d for d in DEPENDENCE if d in feats]:
        x = df[f].to_numpy(dtype=float)
        x_plot = x / 60 if f == 'elapsed_time_s' else x
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x_plot, df[f'shap_{f}'], s=3, alpha=0.3, c=df['y_true'].map({0: '#3b6ea5', 1: '#d1495b'}), linewidths=0)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('elapsed time since anesthesia start (min)' if f == 'elapsed_time_s' else f)
        ax.set_ylabel(f'SHAP {f} (log-odds)')
        ax.set_title(f'{label} (red = positive window)')
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'shap_dependence_{label}_{f}.png'), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['A', 'B'], choices=['A', 'B'])
    ap.add_argument('--horizons', nargs='+', default=list(HORIZONS_S), choices=list(HORIZONS_S))
    ap.add_argument('--n-sample', type=int, default=2000, help='outer-test windows explained per fold')
    ap.add_argument('--variant', default='sqi_primary')
    ap.add_argument('--min-window-frac', type=float, default=PRIMARY_MIN_WINDOW_FRAC)
    ap.add_argument('--period', choices=['anesthesia', 'surgery'], default='anesthesia')
    ap.add_argument('--n-jobs', type=int, default=-1)
    args = ap.parse_args()
    tag = f"{args.variant}_w{int(round(args.min_window_frac * 100))}" + ('_surgery' if args.period == 'surgery' else '')
    os.makedirs(OUT_DIR, exist_ok=True)
    folds = pd.read_csv(FOLDS_FILE).set_index('caseid')['fold']
    base = build_base(args.variant, args.min_window_frac, args.period)
    print(f"=== SHAP v2 | run '{tag}' | {args.n_sample} outer-test windows per fold ===")

    imp_rows, checks, pooled_by = [], [], {}
    for mk in args.models:
        feats = MODEL_SPECS[mk]['features']
        for h in args.horizons:
            df, ch = explain(base, mk, h, tag, folds, args.n_sample, args.n_jobs)
            checks += ch
            df.to_parquet(os.path.join(OUT_DIR, f'shap_{tag}_{mk}_{h}.parquet'))
            pooled, per_fold, ranks = importance(df, feats)
            pooled_by[(mk, h)] = (pooled, per_fold)
            prank = pooled.rank(ascending=False, method='min')
            for f in feats:
                imp_rows.append({'run': tag, 'model': mk, 'horizon': h, 'feature': f, 'mean_abs_shap': pooled[f],
                                 'rank': int(prank[f]), 'rank_min_across_folds': int(ranks.loc[f].min()),
                                 'rank_max_across_folds': int(ranks.loc[f].max()),
                                 **{f'mean_abs_shap_fold{k}': per_fold.loc[f, k] for k in per_fold.columns}})
            top = pooled.sort_values(ascending=False)
            print(f"  [{mk}/{h}] top 10: " + ", ".join(f"{f} {v:.3f}" for f, v in top.iloc[:10].items()))
            plots(df, feats, pooled, f'{tag}_{mk}_{h}')

    cons = []
    for mk in args.models:
        hs = [h for h in args.horizons if (mk, h) in pooled_by]
        for h1, h2 in itertools.combinations(hs, 2):
            cons.append({'run': tag, 'model': mk, 'comparison': 'across horizons', 'a': h1, 'b': h2,
                         **compare(pooled_by[(mk, h1)][0], pooled_by[(mk, h2)][0])})
        for h in hs:
            pf = pooled_by[(mk, h)][1]
            pair = [compare(pf[a], pf[b]) for a, b in itertools.combinations(pf.columns, 2)]
            cons.append({'run': tag, 'model': mk, 'comparison': 'across folds (mean of pairs)', 'a': h, 'b': h,
                         'spearman': np.mean([p['spearman'] for p in pair]),
                         'jaccard_top5': np.mean([p['jaccard_top5'] for p in pair]),
                         'jaccard_top10': np.mean([p['jaccard_top10'] for p in pair])})
    cons = pd.DataFrame(cons)

    sfx = '' if args.period == 'anesthesia' else '_surgery'
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT_DIR, f'shap_importance{sfx}.csv'), index=False)
    cons.to_csv(os.path.join(OUT_DIR, f'shap_consistency{sfx}.csv'), index=False)
    pd.DataFrame(checks).to_csv(os.path.join(OUT_DIR, f'shap_checks{sfx}.csv'), index=False)
    pd.set_option('display.width', 200)
    print("\n=== Consistency of SHAP rankings ===")
    print(cons.drop(columns='run').round(3).to_string(index=False))
    ck = pd.DataFrame(checks)
    print(f"\nChecks: max refit difference vs saved predictions {ck['refit_max_abs_diff_vs_saved'].max():.2e}; "
          f"max additivity error {ck['additivity_max_abs_err_logodds'].max():.1e} (log-odds)")
    print(f"Saved shap_importance{sfx}.csv, shap_consistency{sfx}.csv, shap_checks{sfx}.csv, per-window SHAP "
          f"parquets and plots to {OUT_DIR}")


if __name__ == '__main__':
    main()