"""
Phase 5 (v2) — feature engineering on the corrected Phase 4 cohort.

Changes from v1 (decision log entry 44):
  1. elapsed_time_s is time since ANESTHESIA START (VitalDB 'anestart'), not since
     the first monitor sample. Monitor-relative time is kept as 't' (row key) and
     case-relative time as 't_case_s'. Rows before anestart get negative values;
     whether they are analysed is decided in Phase 6, not here.
  2. TCI: tiva_propofol = 1 if the case has any Orchestra/PPF20_CE data. For cases
     with no propofol TCI track, PPF20_CE_mean = 0; with no remifentanil TCI track,
     RFTN20_CE_mean = 0 (no infusion = zero effect-site concentration, instead of a
     MICE-imputed value). Within TCI cases, CE means use all available samples in the
     window (a model-computed quantity, not a monitored signal); windows with none stay NaN.
  3. Window completeness is STORED ({track}_frac) instead of hard-coded: features are
     computed for windows >= 50% present; the primary >= 70% rule and the Test 6
     alternatives (90/70/50%) are applied downstream by filtering rows.
  4. {track}_last = last observed value in the window (naive baseline, protocol 5.2).
  5. Clinical variables merged here from a LOCAL snapshot of the VitalDB clinical table
     (created once, then always read from disk). ASA coded 6 is set to missing
     (entry 13 data-quality caveat).
  6. DFA/sample entropy reuse: interim HR is unaffected by the entry-43 fix, so v1
     values are reused for (caseid, t) rows already computed, after a spot check that
     recomputes 3 cached cases and requires identical values. Only cases new to the
     cohort are computed from scratch. Known, documented property kept for
     comparability: NaNs are dropped before DFA/SampEn (entry 44).

Safety checks (script stops before writing if they fail):
  - case-time mapping: raw dt - VITALDB_EPOCH must reproduce the clinical caseend
    (median |difference| <= 60 s across the cohort).
  - DFA cache spot check (identical values).
Output: data/processed/phase5_features_v2.parquet (v1 table is left untouched).
"""

import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import nolds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import (INTERIM_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, CLINICAL_SNAPSHOT,
                        VITALDB_EPOCH, WINDOW_S, STRIDE_S, CORE_TRACKS, FEATURE_MIN_WINDOW_FRAC,
                        RANDOM_SEED)

V1_TABLE = os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet')
OUT_TABLE = os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet')
STEP1_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2', 'BIS', 'EMG', 'PPF20_CE', 'RFTN20_CE']
DFA_ALPHA1_NVALS = np.unique(np.logspace(np.log10(4), np.log10(16), 8).astype(int))
DFA_ALPHA2_NVALS = np.unique(np.logspace(np.log10(16), np.log10(64), 8).astype(int))
NONLIN = ['HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen']
MAX_CASE_TIME_MISMATCH_S = 60
_X = np.arange(WINDOW_S)


def load_clinical():
    if not os.path.exists(CLINICAL_SNAPSHOT):
        print(f"No local clinical snapshot — downloading once to {CLINICAL_SNAPSHOT}")
        pd.read_csv("https://api.vitaldb.net/cases").to_csv(CLINICAL_SNAPSHOT, index=False)
    return pd.read_csv(CLINICAL_SNAPSHOT)


def grid_start_and_end(cid):
    raw = pd.read_parquet(os.path.join(RAW_DIR, f"{cid}.parquet"), columns=['dt', 'tname', 'nval'])
    g = raw[raw['tname'].isin(STEP1_TRACKS) & raw['nval'].notna()]['dt']
    return g.min(), g.max()


def window_ends(n):
    return list(range(WINDOW_S, n, STRIDE_S))


def rolling_stats(arr, ends):
    out = {k: np.full(len(ends), np.nan) for k in ('mean', 'slope', 'var', 'last', 'frac')}
    for j, end in enumerate(ends):
        w = arr[end - WINDOW_S:end]
        m = ~np.isnan(w)
        n_valid = int(m.sum())
        out['frac'][j] = n_valid / WINDOW_S
        if n_valid < WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
            continue
        y, x = w[m], _X[m]
        out['mean'][j] = y.mean()
        out['var'][j] = y.var()
        out['last'][j] = y[-1]
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom > 0:
            out['slope'][j] = np.sum((x - xm) * (y - y.mean())) / denom
    return out


def ce_mean(arr, ends):
    out = np.full(len(ends), np.nan)
    for j, end in enumerate(ends):
        w = arr[end - WINDOW_S:end]
        w = w[~np.isnan(w)]
        if len(w):
            out[j] = w.mean()
    return out


def nonlinear_hr(arr, ends):
    a1, a2, se = (np.full(len(ends), np.nan) for _ in range(3))
    for j, end in enumerate(ends):
        valid = arr[end - WINDOW_S:end]
        valid = valid[~np.isnan(valid)]
        if len(valid) < WINDOW_S * FEATURE_MIN_WINDOW_FRAC:
            continue
        try:
            a1[j] = nolds.dfa(valid, nvals=DFA_ALPHA1_NVALS, order=1)
        except Exception:
            pass
        try:
            a2[j] = nolds.dfa(valid, nvals=DFA_ALPHA2_NVALS, order=1)
        except Exception:
            pass
        try:
            sd = np.std(valid)
            if sd > 0:
                se[j] = nolds.sampen(valid, emb_dim=2, tolerance=0.2 * sd)
        except Exception:
            pass
    return a1, a2, se


def build_case(args):
    cid, need_nonlinear = args
    try:
        wide = pd.read_parquet(os.path.join(INTERIM_DIR, f"{cid}.parquet"))
    except Exception as e:
        return None, (cid, str(e))
    n = len(wide)
    if n <= WINDOW_S:
        return None, None
    ends = window_ends(n)
    feat = pd.DataFrame({'caseid': cid, 't': ends})
    for track in CORE_TRACKS:
        st = rolling_stats(wide[track].values.astype(float), ends)
        for k, v in st.items():
            feat[f'{track}_{k}'] = v
    if need_nonlinear:
        a1, a2, se = nonlinear_hr(wide['HR'].values.astype(float), ends)
        feat['HR_dfa_alpha1'], feat['HR_dfa_alpha2'], feat['HR_sampen'] = a1, a2, se
    ppf = wide['PPF20_CE'].values.astype(float) if 'PPF20_CE' in wide else np.full(n, np.nan)
    rft = wide['RFTN20_CE'].values.astype(float) if 'RFTN20_CE' in wide else np.full(n, np.nan)
    has_ppf, has_rft = bool(np.isfinite(ppf).any()), bool(np.isfinite(rft).any())
    feat['tiva_propofol'] = int(has_ppf)
    feat['PPF20_CE_mean'] = ce_mean(ppf, ends) if has_ppf else 0.0
    feat['RFTN20_CE_mean'] = ce_mean(rft, ends) if has_rft else 0.0
    return feat, None


if __name__ == '__main__':
    cohort = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'))['caseid'].tolist()
    clinical = load_clinical()
    clin = clinical.set_index('caseid')
    print(f"Cohort: {len(cohort)} cases")

    # ---- Check 1: case-time mapping (raw dt - epoch == seconds from casestart) ----
    print("Checking case-time mapping against clinical caseend...")
    starts, mism = {}, []
    for cid in cohort:
        g0, g1 = grid_start_and_end(cid)
        starts[cid] = g0 - VITALDB_EPOCH
        mism.append((g1 - VITALDB_EPOCH) - clin.at[cid, 'caseend'])
    mism = pd.Series(mism)
    start_off = pd.Series(starts)
    print(f"  grid start - casestart (s): median {start_off.median():.1f}, "
          f"min {start_off.min():.1f}, max {start_off.max():.1f}")
    print(f"  last sample - caseend (s): median {mism.median():.1f}, "
          f"median |diff| {mism.abs().median():.1f}, >300s in {(mism.abs() > 300).sum()} cases")
    if mism.abs().median() > MAX_CASE_TIME_MISMATCH_S:
        print("STOP: raw timestamps do not map onto clinical case time as assumed. Nothing written.")
        sys.exit(1)
    n_no_anestart = int(clin.loc[cohort, 'anestart'].isna().sum())
    print(f"  cases with missing anestart: {n_no_anestart}")

    # ---- DFA/SampEn cache from v1 ----
    cache = None
    if os.path.exists(V1_TABLE):
        cache = pd.read_parquet(V1_TABLE, columns=['caseid', 't'] + NONLIN)
        cached_cases = set(cache['caseid'].unique())
    else:
        cached_cases = set()
    jobs = [(cid, cid not in cached_cases) for cid in cohort]
    n_new = sum(need for _, need in jobs)
    print(f"Nonlinear HR features: reusing v1 values for {len(cohort) - n_new} cases, computing {n_new} new")

    # ---- Check 2: cache spot check ----
    if cache is not None and len(cached_cases & set(cohort)) > 0:
        rng = np.random.RandomState(RANDOM_SEED)
        spot = rng.choice(sorted(cached_cases & set(cohort)), size=min(3, len(cached_cases & set(cohort))), replace=False)
        ok = True
        for cid in spot:
            fresh, _ = build_case((cid, True))
            old = cache[cache['caseid'] == cid].set_index('t')[NONLIN]
            new = fresh.set_index('t')[NONLIN].reindex(old.index)
            same = np.allclose(old.values, new.values, equal_nan=True)
            print(f"  cache spot check case {cid}: {'identical' if same else 'MISMATCH'}")
            ok &= same
        if not ok:
            print("STOP: cached DFA/SampEn values differ from recomputation. Nothing written.")
            sys.exit(1)

    # ---- Build ----
    start = time.time()
    parts, failed = [], []
    n_workers = min(cpu_count(), 20)
    with Pool(n_workers) as pool:
        for i, (feat, err) in enumerate(pool.imap(build_case, jobs, chunksize=4)):
            if err is not None:
                failed.append(err)
            elif feat is not None:
                parts.append(feat)
            if (i + 1) % 250 == 0:
                print(f"[{i+1}/{len(jobs)}] elapsed={time.time()-start:.0f}s")
    table = pd.concat(parts, ignore_index=True)

    if cache is not None:
        cached_rows = table.loc[table['caseid'].isin(cached_cases), ['caseid', 't']]
        merged = cached_rows.merge(cache, on=['caseid', 't'], how='left')
        for c in NONLIN:
            if c not in table:
                table[c] = np.nan
            table.loc[cached_rows.index, c] = merged[c].values

    # ---- Times and clinical ----
    table['t_case_s'] = table['caseid'].map(starts) + table['t']
    table['elapsed_time_s'] = table['t_case_s'] - table['caseid'].map(clin['anestart'])
    table['aneend_case_s'] = table['caseid'].map(clin['aneend'])
    for c in ['age', 'bmi', 'asa', 'preop_hb', 'sex']:
        table[c] = table['caseid'].map(clin[c])
    n_asa6 = int(table.loc[table['asa'] == 6, 'caseid'].nunique())
    table.loc[table['asa'] == 6, 'asa'] = np.nan

    table.to_parquet(OUT_TABLE)

    # ---- Report ----
    print(f"\nDone in {(time.time()-start)/60:.1f} min. {len(cohort) - len(failed)} cases, {len(failed)} failed.")
    print(f"Rows: {len(table)}  -> {OUT_TABLE}")
    print(f"TIVA (propofol TCI) cases: {table.groupby('caseid')['tiva_propofol'].first().sum()} / {table['caseid'].nunique()}")
    tiva = table['tiva_propofol'] == 1
    print(f"  PPF20_CE_mean non-null within TIVA rows: {table.loc[tiva, 'PPF20_CE_mean'].notna().mean()*100:.1f}%")
    print(f"ASA coded 6 set to missing: {n_asa6} cases")
    print(f"Rows before anestart (elapsed_time_s < 0): {(table['elapsed_time_s'] < 0).sum()} "
          f"({(table['elapsed_time_s'] < 0).mean()*100:.1f}%)")
    print(f"Rows after aneend: {(table['t_case_s'] > table['aneend_case_s']).sum()}")
    for tr in CORE_TRACKS:
        f = table[f'{tr}_frac']
        print(f"  {tr}: windows >=90% {(f>=0.9).mean()*100:.1f}% | >=70% {(f>=0.7).mean()*100:.1f}% | >=50% {(f>=0.5).mean()*100:.1f}%")
    allcore70 = np.logical_and.reduce([table[f'{t}_frac'] >= 0.7 for t in CORE_TRACKS])
    print(f"Rows passing primary rule (all core tracks >=70% present): {allcore70.sum()} ({allcore70.mean()*100:.1f}%)")
    print(f"Nonlinear HR non-null: {table['HR_dfa_alpha1'].notna().mean()*100:.1f}%")
    if failed:
        print(f"First failures: {failed[:5]}")