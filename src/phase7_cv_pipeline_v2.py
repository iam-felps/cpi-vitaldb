"""
Phase 7 (v2) — nested grouped CV with algorithm selection, class weighting and nested
calibration (replaces phase7_cv_pipeline_full.py + phase8_calibration.py +
phase9_predictions_and_bootstrap.py for model fitting). Decision log entry 47.

Per model (A, B, naive baseline) x horizon (5/10/15 min):
  Rows      label non-missing for the chosen variant (Phase 6 v2) AND every core vital
            sign >= --min-window-frac present in the window (primary 0.70).
            Model A and Model B use IDENTICAL rows and folds (paired comparison).
  Outer     5 folds, assigned per CASE (never split), stratified by whether the case has
            any 15-min positive; saved once to results/phase7_v2/outer_folds.csv and
            reused by every run (all models, horizons, variants, sensitivity analyses).
  Inner     3 folds per case within the outer-training set, stratified the same way.
  Preproc   MICE (IterativeImputer) on the model's own baseline/lab variables only
            (A: age, bmi, asa, preop_hb; B: age, bmi, asa), refit on every training
            partition (inner AND outer). All other missing values are left as NaN and
            handled natively by the GBDT. Naive baseline (logistic regression): median
            imputation + standardisation, refit on every training partition.
  Balance   class weighting only (scale_pos_weight = n_neg/n_pos of the training
            partition; logistic regression class_weight='balanced'). No SMOTE.
  Tune      library (LightGBM / XGBoost / CatBoost) x 4 hyperparameter settings each,
            selected by mean inner-fold AUROC on outer-training data only.
  Calibrate isotonic vs Platt chosen by case-grouped 3-fold CV (Brier) over the selected
            candidate's inner out-of-fold predictions; the winner is refit on all of
            them and applied to the outer-test predictions. Never touches outer test.
  Evaluate  selected candidate refit on the full outer-training set, predicts outer test
            once. Every prediction is saved with caseid and t.
Leakage checks (run fails if violated): no case in both train and test (outer and
inner); MICE fitted statistics differ across outer folds.

Ablations: --drop-features f1 f2 ... removes features from every model (tag gets a
'_drop-...' suffix); combine with --fixed-config-from for a quick run.

Sensitivity runs (Tests 3, 4, 6): --variant / --min-window-frac with
--fixed-config-from <tag> reuse the primary run's selected library + hyperparameters per
model/horizon/fold (no re-tuning), keeping the same outer folds.

Usage (from the project root):
  python src/phase7_cv_pipeline_v2.py                          # primary, all 9 combos
  python src/phase7_cv_pipeline_v2.py --models A --horizons 5min   # one combo
Each finished combo is saved immediately; rerunning skips finished combos (resume).
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import (PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED, CORE_TRACKS, HORIZONS_S,
                        MODEL_A_FEATURES, MODEL_B_FEATURES, BASELINE_FEATURES,
                        PRIMARY_MIN_WINDOW_FRAC)

warnings.filterwarnings('ignore')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase7_v2')
FOLDS_FILE = os.path.join(OUT_DIR, 'outer_folds.csv')
N_OUTER, N_INNER, N_CALIB = 5, 3, 3
N_TREES = 200

GRIDS = {
    'lightgbm': [{'learning_rate': lr, 'num_leaves': nl} for lr in (0.05, 0.1) for nl in (15, 31)],
    'xgboost': [{'learning_rate': lr, 'max_depth': d} for lr in (0.05, 0.1) for d in (4, 6)],
    'catboost': [{'learning_rate': lr, 'depth': d} for lr in (0.05, 0.1) for d in (4, 6)],
}
MODEL_SPECS = {
    'A': {'features': MODEL_A_FEATURES, 'mice': ['age', 'bmi', 'asa', 'preop_hb'], 'linear': False},
    'B': {'features': MODEL_B_FEATURES, 'mice': ['age', 'bmi', 'asa'], 'linear': False},
    'baseline': {'features': BASELINE_FEATURES, 'mice': [], 'linear': True},
}


# ---------------------------------------------------------------- preprocessing / models
class Prep:
    """MICE on baseline/lab variables (tree models) or median+standardise (logistic)."""

    def __init__(self, features, mice_vars, linear):
        self.features, self.mice_vars, self.linear = features, mice_vars, linear

    def fit(self, X):
        X = X[self.features].copy()
        self.mice = None
        if self.mice_vars:
            self.mice = IterativeImputer(random_state=RANDOM_SEED, max_iter=10).fit(X[self.mice_vars])
            X[self.mice_vars] = self.mice.transform(X[self.mice_vars])
        if self.linear:
            self.med = SimpleImputer(strategy='median').fit(X)
            self.sc = StandardScaler().fit(self.med.transform(X))
        return self

    def transform(self, X):
        X = X[self.features].copy()
        if self.mice is not None:
            X[self.mice_vars] = self.mice.transform(X[self.mice_vars])
        if self.linear:
            return self.sc.transform(self.med.transform(X))
        return X.to_numpy(dtype=float)

    def signature(self):
        return None if self.mice is None else tuple(np.round(self.mice.initial_imputer_.statistics_, 8))


def make_model(lib, params, spw, n_jobs):
    if lib == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=N_TREES, min_child_samples=100, scale_pos_weight=spw,
                                  random_state=RANDOM_SEED, n_jobs=n_jobs, verbose=-1, **params)
    if lib == 'xgboost':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=N_TREES, tree_method='hist', scale_pos_weight=spw,
                                 random_state=RANDOM_SEED, n_jobs=n_jobs, eval_metric='logloss', **params)
    if lib == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=N_TREES, scale_pos_weight=spw, random_seed=RANDOM_SEED,
                                  thread_count=n_jobs, verbose=False, allow_writing_files=False, **params)
    if lib == 'logistic':
        return LogisticRegression(class_weight='balanced', max_iter=2000)
    raise ValueError(lib)


def fit_and_predict(lib, params, Xtr, ytr, Xte, n_jobs):
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    m = make_model(lib, params, spw, n_jobs)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------- folds
def stratified_case_folds(case_strata, n_folds, seed):
    """case_strata: Series caseid -> 0/1. Round-robin within shuffled strata."""
    rng = np.random.RandomState(seed)
    fold = {}
    for s in sorted(case_strata.unique()):
        cases = np.array(sorted(case_strata[case_strata == s].index))
        rng.shuffle(cases)
        for i, c in enumerate(cases):
            fold[c] = i % n_folds
    return pd.Series(fold)


def load_outer_folds(labels_primary):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FOLDS_FILE):
        return pd.read_csv(FOLDS_FILE).set_index('caseid')['fold']
    strata = labels_primary.groupby('caseid')['label_15min'].max().fillna(0).astype(int)
    folds = stratified_case_folds(strata, N_OUTER, RANDOM_SEED)
    folds.rename_axis('caseid').rename('fold').reset_index().to_csv(FOLDS_FILE, index=False)
    print(f"Outer folds created and saved: {FOLDS_FILE}")
    return folds


# ---------------------------------------------------------------- calibration
def select_and_fit_calibrator(oof_pred, oof_y, oof_case):
    case_strata = pd.Series(oof_y, index=oof_case).groupby(level=0).max()
    cfold = stratified_case_folds(case_strata, N_CALIB, RANDOM_SEED)
    f = cfold.reindex(oof_case).to_numpy()
    b_iso, b_platt = [], []
    for k in range(N_CALIB):
        tr, va = f != k, f == k
        if oof_y[tr].sum() < 2 or oof_y[va].sum() < 1:
            continue
        iso = IsotonicRegression(out_of_bounds='clip').fit(oof_pred[tr], oof_y[tr])
        b_iso.append(brier_score_loss(oof_y[va], iso.predict(oof_pred[va])))
        pl = LogisticRegression().fit(oof_pred[tr].reshape(-1, 1), oof_y[tr])
        b_platt.append(brier_score_loss(oof_y[va], pl.predict_proba(oof_pred[va].reshape(-1, 1))[:, 1]))
    mi, mp = (np.mean(b_iso) if b_iso else np.inf), (np.mean(b_platt) if b_platt else np.inf)
    if mi <= mp:
        cal = IsotonicRegression(out_of_bounds='clip').fit(oof_pred, oof_y)
        return 'isotonic', mi, mp, lambda p: cal.predict(p)
    cal = LogisticRegression().fit(oof_pred.reshape(-1, 1), oof_y)
    return 'platt', mi, mp, lambda p: cal.predict_proba(p.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------- one combo
def run_combo(data, model_key, horizon, outer_folds, fixed_cfg, n_jobs, log):
    spec = MODEL_SPECS[model_key]
    y_all = data[f'label_{horizon}'].to_numpy().astype(int)
    fold_all = outer_folds.reindex(data['caseid']).to_numpy()
    if model_key == 'baseline':
        candidates = [('logistic', {})]
    else:
        candidates = [(lib, p) for lib, grid in GRIDS.items() for p in grid]

    pred_parts, cfg_rows, signatures = [], [], []
    for k in range(N_OUTER):
        t0 = time.time()
        tr_idx, te_idx = np.where(fold_all != k)[0], np.where(fold_all == k)[0]
        dtr, dte = data.iloc[tr_idx], data.iloc[te_idx]
        ytr, yte = y_all[tr_idx], y_all[te_idx]
        assert not set(dtr['caseid']) & set(dte['caseid']), "LEAK: case in outer train and test"

        # inner folds (per case, stratified by case-level positive for this horizon)
        strata = pd.Series(ytr, index=dtr['caseid'].to_numpy()).groupby(level=0).max()
        inner_f = stratified_case_folds(strata, N_INNER, RANDOM_SEED + 1 + k).reindex(dtr['caseid']).to_numpy()

        if fixed_cfg is not None:
            row = fixed_cfg[(fixed_cfg['fold'] == k)].iloc[0]
            cands = [(row['library'], json.loads(row['params']))]
        else:
            cands = candidates
        oof = np.full((len(cands), len(dtr)), np.nan)
        for j in range(N_INNER):
            itr, iva = inner_f != j, inner_f == j
            assert not set(dtr['caseid'][itr]) & set(dtr['caseid'][iva]), "LEAK: case in inner train and val"
            prep = Prep(spec['features'], spec['mice'], spec['linear']).fit(dtr[itr])
            Xi_tr, Xi_va = prep.transform(dtr[itr]), prep.transform(dtr[iva])
            for c, (lib, p) in enumerate(cands):
                oof[c, iva] = fit_and_predict(lib, p, Xi_tr, ytr[itr], Xi_va, n_jobs)
        inner_auc = []
        for c in range(len(cands)):
            scores = [roc_auc_score(ytr[inner_f == j], oof[c, inner_f == j]) for j in range(N_INNER)
                      if len(np.unique(ytr[inner_f == j])) == 2]
            inner_auc.append(np.mean(scores) if scores else -np.inf)
        best = int(np.argmax(inner_auc))
        lib, params = cands[best]

        method, b_iso, b_platt, calibrate = select_and_fit_calibrator(oof[best], ytr, dtr['caseid'].to_numpy())

        prep = Prep(spec['features'], spec['mice'], spec['linear']).fit(dtr)
        signatures.append(prep.signature())
        raw = fit_and_predict(lib, params, prep.transform(dtr), ytr, prep.transform(dte), n_jobs)
        cal = np.clip(calibrate(raw), 0, 1)

        pred_parts.append(pd.DataFrame({'model': model_key, 'horizon': horizon, 'fold': k,
                                        'caseid': dte['caseid'].to_numpy(), 't': dte['t'].to_numpy(),
                                        'y_true': yte, 'raw_pred': raw, 'calibrated_pred': cal}))
        auc = roc_auc_score(yte, raw)
        cfg_rows.append({'model': model_key, 'horizon': horizon, 'fold': k, 'library': lib,
                         'params': json.dumps(params), 'inner_auroc_best': inner_auc[best],
                         'inner_auroc_all': json.dumps({f"{l}|{json.dumps(p)}": round(a, 5)
                                                        for (l, p), a in zip(cands, inner_auc)}),
                         'calibration': method, 'cv_brier_isotonic': b_iso, 'cv_brier_platt': b_platt,
                         'n_train': len(dtr), 'n_test': len(dte), 'pos_train': int(ytr.sum()),
                         'pos_test': int(yte.sum()), 'outer_auroc_raw': auc,
                         'outer_auprc_raw': average_precision_score(yte, raw),
                         'outer_brier_cal': brier_score_loss(yte, cal), 'seconds': round(time.time() - t0)})
        log(f"  [{model_key}/{horizon}] fold {k}: {lib} {params} inner={inner_auc[best]:.4f} "
            f"outer AUROC={auc:.4f} AUPRC={cfg_rows[-1]['outer_auprc_raw']:.4f} calib={method} "
            f"({cfg_rows[-1]['seconds']}s)")

    sig = [s for s in signatures if s is not None]
    if sig and len(set(sig)) == 1:
        raise RuntimeError("LEAK CHECK FAILED: MICE statistics identical across outer folds")
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(cfg_rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['A', 'B', 'baseline'], choices=list(MODEL_SPECS))
    ap.add_argument('--horizons', nargs='+', default=list(HORIZONS_S), choices=list(HORIZONS_S))
    ap.add_argument('--variant', default='sqi_primary')
    ap.add_argument('--min-window-frac', type=float, default=PRIMARY_MIN_WINDOW_FRAC)
    ap.add_argument('--fixed-config-from', default=None, help='tag of a finished run to reuse configs from')
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--drop-features', nargs='+', default=[], help='ablation: remove these features')
    args = ap.parse_args()

    tag = f"{args.variant}_w{int(round(args.min_window_frac * 100))}"
    if args.drop_features:
        tag += '_drop-' + '-'.join(args.drop_features)
        for spec in MODEL_SPECS.values():
            spec['features'] = [f for f in spec['features'] if f not in args.drop_features]
            spec['mice'] = [f for f in spec['mice'] if f not in args.drop_features]
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f'{tag}_log.txt')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n')

    labels = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labels_v2.parquet'))
    outer_folds = load_outer_folds(labels[labels['variant'] == 'sqi_primary'])
    labels = labels[labels['variant'] == args.variant]
    feat_cols = sorted(set(MODEL_A_FEATURES) | set(BASELINE_FEATURES) | {f'{t}_frac' for t in CORE_TRACKS})
    feats = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'),
                            columns=['caseid', 't'] + feat_cols)
    base = labels.merge(feats, on=['caseid', 't'], how='inner')
    window_ok = np.logical_and.reduce([base[f'{t}_frac'] >= args.min_window_frac for t in CORE_TRACKS])
    log(f"=== Phase 7 v2 run '{tag}' | variant={args.variant} min_window_frac={args.min_window_frac} "
        f"fixed_config_from={args.fixed_config_from} drop_features={args.drop_features} ===")
    log(f"Labeled rows: {len(base)}; passing window rule: {int(window_ok.sum())} "
        f"({window_ok.mean()*100:.1f}%)")
    base = base[window_ok].reset_index(drop=True)

    for model_key in args.models:
        for h in args.horizons:
            pred_path = os.path.join(OUT_DIR, f'{tag}_{model_key}_{h}_predictions.parquet')
            cfg_path = os.path.join(OUT_DIR, f'{tag}_{model_key}_{h}_config.csv')
            if os.path.exists(pred_path) and os.path.exists(cfg_path):
                log(f"[{model_key}/{h}] already done — skipping")
                continue
            data = base[base[f'label_{h}'].notna()].sort_values(['caseid', 't']).reset_index(drop=True)
            fixed = None
            if args.fixed_config_from:
                fixed = pd.read_csv(os.path.join(OUT_DIR, f'{args.fixed_config_from}_{model_key}_{h}_config.csv'))
            log(f"\n[{model_key}/{h}] rows={len(data)} cases={data['caseid'].nunique()} "
                f"positives={int(data[f'label_{h}'].sum())} ({data[f'label_{h}'].mean()*100:.2f}%)")
            t0 = time.time()
            preds, cfg = run_combo(data, model_key, h, outer_folds, fixed, args.n_jobs, log)
            preds.to_parquet(pred_path)
            cfg.to_csv(cfg_path, index=False)
            log(f"[{model_key}/{h}] done in {(time.time()-t0)/60:.1f} min | pooled AUROC "
                f"{roc_auc_score(preds['y_true'], preds['raw_pred']):.4f} | libraries chosen "
                f"{cfg['library'].value_counts().to_dict()} | leakage checks passed")


if __name__ == '__main__':
    main()