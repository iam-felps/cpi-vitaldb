"""
Phase 9 (v2) — SHAP ranking consistency with Spearman p-values (protocol v6/v7 Section 7).
Reads results/phase9_v2/shap_importance{_surgery}.csv written by phase9_shap_v2.py; no refitting.

Across horizons (same model): Spearman rho and p-value of mean |SHAP| over all features;
Jaccard index of the top-5 and top-10 feature sets.
Across outer folds (same model and horizon): the same statistics for every pair of folds,
summarised as mean rho, range of rho, largest p-value, mean Jaccard.
Output: results/phase9_v2/shap_consistency_with_p{_surgery}.csv
Usage: python src/phase9_shap_consistency_v2.py [--period anesthesia|surgery]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')


def stats(a, b):
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    rho, p = spearmanr(a, b)
    ta, tb = a.sort_values(ascending=False).index, b.sort_values(ascending=False).index
    jac = lambda k: len(set(ta[:k]) & set(tb[:k])) / len(set(ta[:k]) | set(tb[:k]))
    return {'n_features': len(common), 'spearman_rho': rho, 'spearman_p': p,
            'jaccard_top5': jac(5), 'jaccard_top10': jac(10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', choices=['anesthesia', 'surgery'], default='anesthesia')
    args = ap.parse_args()
    sfx = '' if args.period == 'anesthesia' else '_surgery'
    imp = pd.read_csv(os.path.join(OUT_DIR, f'shap_importance{sfx}.csv'))
    fold_cols = [c for c in imp.columns if c.startswith('mean_abs_shap_fold')]
    rows = []
    for m, gm in imp.groupby('model'):
        pooled = {h: g.set_index('feature')['mean_abs_shap'] for h, g in gm.groupby('horizon')}
        for h1, h2 in itertools.combinations(sorted(pooled, key=lambda h: int(h.replace('min', ''))), 2):
            rows.append({'model': m, 'comparison': 'across horizons', 'a': h1, 'b': h2, **stats(pooled[h1], pooled[h2])})
        for h, g in gm.groupby('horizon'):
            g = g.set_index('feature')
            pairs = [stats(g[a], g[b]) for a, b in itertools.combinations(fold_cols, 2)]
            rows.append({'model': m, 'comparison': 'across folds (all pairs)', 'a': h, 'b': h,
                         'n_features': pairs[0]['n_features'],
                         'spearman_rho': np.mean([p['spearman_rho'] for p in pairs]),
                         'spearman_rho_min': min(p['spearman_rho'] for p in pairs),
                         'spearman_rho_max': max(p['spearman_rho'] for p in pairs),
                         'spearman_p': max(p['spearman_p'] for p in pairs),
                         'jaccard_top5': np.mean([p['jaccard_top5'] for p in pairs]),
                         'jaccard_top10': np.mean([p['jaccard_top10'] for p in pairs])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, f'shap_consistency_with_p{sfx}.csv'), index=False)
    pd.set_option('display.width', 200)
    show = out.copy()
    show['spearman_p'] = show['spearman_p'].map(lambda p: f"{p:.1e}")
    print(show.round(3).to_string(index=False))
    print("\nFor folds, spearman_rho is the mean over all fold pairs and spearman_p the LARGEST p over pairs.")
    print(f"Saved {os.path.join(OUT_DIR, f'shap_consistency_with_p{sfx}.csv')}")


if __name__ == '__main__':
    main()