"""
Phase 9 (v2) — computational latency benchmark (protocol Section 4.1). Entry 56.

Measures, per prediction window, the wall-clock time to go from the raw 1-s signals of the
last 5 minutes to a calibrated probability, for Model A and Model B, as it would run at the
bedside once per minute (stride 60 s):
  features   hemodynamic mean/slope/variance (4 tracks), DFA alpha1 and alpha2, sample
             entropy (HR), and for Model A the propofol/remifentanil effect-site means and
             real-time TIVA flag; the DFA validity rule; assembly of the feature vector
  inference  preprocessing (MICE transform of the clinical variables) + GBDT
             predict_proba + calibration map, one window at a time (batch size 1)
Timed separately: hemodynamic stats, DFA (both exponents), sample entropy, drug features,
inference A, inference B. Totals: A = all features + inference A; B = features without
drug features + inference B.

Hardware tiers (run the script once per tier):
  --tier workstation   default threading (all cores available to numpy/GBDT)
  --tier single        one thread for everything (OMP/MKL/OpenBLAS = 1, GBDT n_jobs = 1)
                       and the process pinned to one CPU core: a deliberately constrained
                       configuration standing in for low-resource hardware
Models: the configuration Phase 7 selected for outer fold 0 at the 5-min horizon, refitted
on fold-0 training cases (as in SHAP v2), plus a calibration map of the type Phase 7 chose.
The calibration map is fitted on fold-0 test predictions: it is used for timing only, and
none of the probabilities produced here are used for any performance claim.
Windows: random sample of outer-fold-0 test windows (primary analysis rows), after warm-up.
Correctness check: the benchmark recomputes every feature from the raw signals and compares
it with the stored Phase 5 value (must match to 1e-6).

Outputs (results/phase9_v2/): latency_{tier}_windows.csv (per-window timings, ms),
latency_summary.csv (mean, SD, median, p95, max per component and tier; appended per tier),
latency_environment_{tier}.json (CPU, cores, RAM, OS, Python and library versions).
Usage: python src/phase9_latency_benchmark_v2.py --tier workstation
       python src/phase9_latency_benchmark_v2.py --tier single
       [--n-windows 1000] [--warmup 50]
"""
import os
import sys

if '--tier' in sys.argv and sys.argv[sys.argv.index('--tier') + 1] == 'single':
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'

import argparse
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import nolds
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED, CORE_TRACKS, WINDOW_S,
                        FEATURE_MIN_WINDOW_FRAC, PRIMARY_MIN_WINDOW_FRAC, MODEL_A_FEATURES, MODEL_B_FEATURES)
from phase5_feature_engineering_v2 import DFA_ALPHA1_NVALS, DFA_ALPHA2_NVALS, DFA_FIT
from phase7_cv_pipeline_v2 import MODEL_SPECS, Prep, make_model, FOLDS_FILE, OUT_DIR as P7_DIR
from phase9_shap_v2 import build_base

warnings.filterwarnings('ignore')
OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
TAG, H = 'sqi_primary_w70', '5min'
_X = np.arange(WINDOW_S)
STATIC = ['elapsed_time_s', 'age', 'bmi', 'asa', 'preop_hb']


# ------------------------------------------------------------ single-window feature code
def hemo_stats(w):
    m = ~np.isnan(w)
    if m.sum() < WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
        return np.nan, np.nan, np.nan
    y, x = w[m], _X[m]
    xm = x.mean()
    den = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - y.mean())) / den if den > 0 else np.nan
    return y.mean(), slope, y.var()


def dfa_pair(hr):
    v = hr[~np.isnan(hr)]
    a1 = a2 = np.nan
    if len(v) >= WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
        try:
            a1 = nolds.dfa(v, nvals=DFA_ALPHA1_NVALS, order=1, fit_exp=DFA_FIT)
        except Exception:
            pass
        try:
            a2 = nolds.dfa(v, nvals=DFA_ALPHA2_NVALS, order=1, fit_exp=DFA_FIT)
        except Exception:
            pass
        if np.var(v) <= 1e-12:
            a1 = a2 = np.nan
        a1 = np.nan if a1 < 0 else a1
        a2 = np.nan if a2 < 0 else a2
    return a1, a2


def sampen(hr):
    v = hr[~np.isnan(hr)]
    if len(v) < WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
        return np.nan
    sd = np.std(v)
    if sd <= 0:
        return np.nan
    try:
        return nolds.sampen(v, emb_dim=2, tolerance=0.2 * sd)
    except Exception:
        return np.nan


def drug(w, started):
    """started = running flag 'this drug's TCI has produced a sample before t' (O(1) state)."""
    if not started:
        return 0.0
    w = w[np.isfinite(w)]
    return w.mean() if len(w) else np.nan


# ------------------------------------------------------------ helpers
def environment(tier, n_jobs):
    env = {'tier': tier, 'gbdt_n_jobs': n_jobs, 'os': platform.platform(), 'machine': platform.machine(),
           'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'python': sys.version.split()[0]}
    try:
        import psutil
        env['physical_cpus'] = psutil.cpu_count(logical=False)
        env['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        f = psutil.cpu_freq()
        env['cpu_max_mhz'] = f.max if f else None
    except Exception:
        env['psutil'] = 'not installed (pip install psutil for RAM/CPU details and core pinning)'
    for lib in ('numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 'catboost', 'nolds'):
        try:
            from importlib.metadata import version
            env[lib] = version('scikit-learn' if lib == 'sklearn' else lib)
        except Exception:
            env[lib] = getattr(__import__(lib), '__version__', None)
    return env


def pin_one_core():
    try:
        import psutil
        psutil.Process().cpu_affinity([0])
        return 'pinned to CPU 0 (psutil)'
    except Exception:
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {0})
            return 'pinned to CPU 0 (sched_setaffinity)'
    return 'NOT pinned (install psutil); single-thread settings still applied'


def fit_model(base, mk, folds, n_jobs):
    spec = MODEL_SPECS[mk]
    data = base[base[f'label_{H}'].notna()].sort_values(['caseid', 't']).reset_index(drop=True)
    cfg = pd.read_csv(os.path.join(P7_DIR, f'{TAG}_{mk}_{H}_config.csv'))
    row = cfg[cfg['fold'] == 0].iloc[0]
    lib, params = row['library'], json.loads(row['params'])
    f = folds.reindex(data['caseid']).to_numpy()
    dtr, dte = data[f != 0], data[f == 0]
    y = dtr[f'label_{H}'].to_numpy().astype(int)
    prep = Prep(spec['features'], spec['mice'], spec['linear']).fit(dtr)
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    model = make_model(lib, params, spw, n_jobs)
    model.fit(prep.transform(dtr), y)
    raw_te = model.predict_proba(prep.transform(dte))[:, 1]
    yte = dte[f'label_{H}'].to_numpy().astype(int)
    if row['calibration'] == 'isotonic':
        c = IsotonicRegression(out_of_bounds='clip').fit(raw_te, yte)
        cal = c.predict
    else:
        c = LogisticRegression().fit(raw_te.reshape(-1, 1), yte)
        cal = lambda p: c.predict_proba(p.reshape(-1, 1))[:, 1]
    return prep, model, cal, lib, params, row['calibration'], dte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['workstation', 'single'], required=True)
    ap.add_argument('--n-windows', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    n_jobs = 1 if args.tier == 'single' else -1
    pin = pin_one_core() if args.tier == 'single' else 'not pinned'
    env = environment(args.tier, n_jobs)
    env['affinity'] = pin
    print(f"=== Latency benchmark | tier '{args.tier}' | {pin} ===")
    print(f"CPU: {env['processor'] or env['machine']} | logical CPUs {env['logical_cpus']} | RAM {env.get('ram_gb')} GB")

    folds = pd.read_csv(FOLDS_FILE).set_index('caseid')['fold']
    base = build_base('sqi_primary', PRIMARY_MIN_WINDOW_FRAC, 'anesthesia')
    models = {}
    for mk in ('A', 'B'):
        t0 = time.time()
        prep, model, cal, lib, params, cm, dte = fit_model(base, mk, folds, n_jobs)
        models[mk] = (prep, model, cal)
        print(f"Model {mk}: {lib} {params}, calibration {cm} (fitted in {time.time()-t0:.0f}s)")

    stored = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'),
                             columns=['caseid', 't'] + sorted(set(MODEL_A_FEATURES)))
    rng = np.random.RandomState(RANDOM_SEED)
    pick = dte[['caseid', 't']].iloc[np.sort(rng.choice(len(dte), size=min(args.n_windows + args.warmup, len(dte)), replace=False))]
    pick = pick.merge(stored, on=['caseid', 't'], how='left').sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    cache = {}
    rows, max_err = [], 0.0
    feat_names_hemo = [f'{t}_{s}' for t in CORE_TRACKS for s in ('mean', 'slope', 'var')]
    for i, r in pick.iterrows():
        cid, t = int(r['caseid']), int(r['t'])
        if cid not in cache:
            if len(cache) > 50:
                cache.clear()
            w = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"), columns=CORE_TRACKS + ['PPF20_CE', 'RFTN20_CE'])
            arr = {c: w[c].to_numpy(float) for c in w.columns}
            first = {c: (int(np.argmax(np.isfinite(arr[c]))) if np.isfinite(arr[c]).any() else None) for c in ('PPF20_CE', 'RFTN20_CE')}
            cache[cid] = (arr, first)
        arr, first = cache[cid]
        win = {c: arr[c][t - WINDOW_S:t] for c in arr}
        started = {c: first[c] is not None and first[c] < t for c in first}   # running state, kept by a live system
        static = {k: r[k] for k in STATIC}

        t0 = time.perf_counter_ns()
        feats = {}
        for tr in CORE_TRACKS:
            feats[f'{tr}_mean'], feats[f'{tr}_slope'], feats[f'{tr}_var'] = hemo_stats(win[tr])
        t1 = time.perf_counter_ns()
        feats['HR_dfa_alpha1'], feats['HR_dfa_alpha2'] = dfa_pair(win['HR'])
        t2 = time.perf_counter_ns()
        feats['HR_sampen'] = sampen(win['HR'])
        t3 = time.perf_counter_ns()
        feats['PPF20_CE_mean'] = drug(win['PPF20_CE'], started['PPF20_CE'])
        feats['RFTN20_CE_mean'] = drug(win['RFTN20_CE'], started['RFTN20_CE'])
        feats['tiva_propofol'] = int(started['PPF20_CE'])
        t4 = time.perf_counter_ns()
        feats.update(static)
        out = {}
        tinf = {}
        for mk in ('A', 'B'):
            prep, model, cal = models[mk]
            s = time.perf_counter_ns()
            X = prep.transform(pd.DataFrame([feats]))
            out[mk] = float(cal(model.predict_proba(X)[:, 1])[0])
            tinf[mk] = time.perf_counter_ns() - s

        for k in feat_names_hemo + ['HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen', 'PPF20_CE_mean', 'RFTN20_CE_mean', 'tiva_propofol']:
            a, b = feats[k], r[k]
            if not (pd.isna(a) and pd.isna(b)):
                err = abs(float(a) - float(b)) if not (pd.isna(a) or pd.isna(b)) else np.inf
                if err > 1e-6:
                    raise RuntimeError(f"Feature {k} of case {cid} t={t}: benchmark {a} vs stored {b} — code paths differ")
                max_err = max(max_err, err)
        if i < args.warmup:
            continue
        ms = lambda ns: ns / 1e6
        rows.append({'caseid': cid, 't': t, 'hemodynamic_ms': ms(t1 - t0), 'dfa_ms': ms(t2 - t1), 'sampen_ms': ms(t3 - t2),
                     'drug_ms': ms(t4 - t3), 'inference_A_ms': ms(tinf['A']), 'inference_B_ms': ms(tinf['B']),
                     'total_A_ms': ms(t4 - t0 + tinf['A']), 'total_B_ms': ms(t3 - t0 + tinf['B'])})
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(pick)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f'latency_{args.tier}_windows.csv'), index=False)
    summ = []
    for c in [c for c in df.columns if c.endswith('_ms')]:
        v = df[c]
        summ.append({'tier': args.tier, 'component': c[:-3], 'n_windows': len(v), 'mean_ms': v.mean(), 'sd_ms': v.std(),
                     'median_ms': v.median(), 'p95_ms': v.quantile(0.95), 'max_ms': v.max()})
    summ = pd.DataFrame(summ)
    sp = os.path.join(OUT_DIR, 'latency_summary.csv')
    if os.path.exists(sp):
        old = pd.read_csv(sp)
        summ = pd.concat([old[old['tier'] != args.tier], summ], ignore_index=True)
    summ.to_csv(sp, index=False)
    with open(os.path.join(OUT_DIR, f'latency_environment_{args.tier}.json'), 'w') as fh:
        json.dump(env, fh, indent=2)

    print(f"\nFeature check: all benchmark features match the stored Phase 5 values (max abs diff {max_err:.1e})")
    print(f"\n=== Per-window latency, tier '{args.tier}' ({len(df)} windows, batch size 1) ===")
    print(summ[summ['tier'] == args.tier].drop(columns='tier').round(2).to_string(index=False))
    print(f"\nBudget: one prediction per 60 s stride. Model A p95 = {df['total_A_ms'].quantile(.95):.1f} ms "
          f"({df['total_A_ms'].quantile(.95)/600:.3f}% of the budget); Model B p95 = {df['total_B_ms'].quantile(.95):.1f} ms")
    print(f"Saved latency_{args.tier}_windows.csv, latency_summary.csv, latency_environment_{args.tier}.json to {OUT_DIR}")
    print("Software versions: " + ", ".join(f"{k} {env[k]}" for k in ('python', 'numpy', 'pandas', 'sklearn', 'lightgbm',
                                                                        'xgboost', 'catboost', 'nolds')))


if __name__ == '__main__':
    main()