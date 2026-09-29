"""
Phase 9 (v2) — cohort flow (attrition) diagram and Table 1, rebuilt from saved files only
(no network, no vitaldb.find_cases). Entry 57.

Cohort flow, each step recomputed and cross-checked against the saved case-ID lists:
  1. All VitalDB cases (local clinical snapshot)
  2. General anesthesia (ane_type == 'General')
  3. Excluding liver transplantation and donor hepatectomy
  4. Recording length >= 45 min (caseend - casestart)        == phase3_step4_caseids.csv
  5. All six core tracks present (verified per case with VitalFile.get_track_names,
     phase3_track_recheck.csv; replaces the find_cases() count of 3,213, entry 21)
                                                             == phase3_final_cohort_caseids.csv
     exclusions broken down by missing track; download errors reported separately
  6. Case-level missingness above the cohort 95th percentile (HR whole case, MAP whole
     case, BIS final 45 min), thresholds recomputed from phase4_case_missingness.csv
                                                             == phase4_final_cohort_caseids.csv
  7. Cases contributing >= 1 primary-analysis window (>= 70% completeness, observable
     label) — the two-case gap is identified and explained
Table 1 (final cohort), overall and by case with >= 1 sustained BIS>=60 event with onset
during anesthesia (primary SQI<50 labels) vs none; standardized mean differences (SMD).
Repeat patients: cases per VitalDB subjectid.

Outputs (results/phase9_v2/): cohort_flow.csv, cohort_flow.png, table1.csv, table1.md,
repeat_patients.csv
Usage: python src/phase9_table1_attrition_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, PROCESSED_DIR, CLINICAL_SNAPSHOT

OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')
P7 = os.path.join(RESULTS_DIR, 'phase7_v2')
EXCLUDED_OPS = ['Liver transplantation', 'Donor hepatectomy']


def ids(name):
    return set(pd.read_csv(os.path.join(RESULTS_DIR, name))['caseid'].astype(int))


def check(label, got, saved):
    if got != saved:
        print(f"  !! {label}: recomputed {len(got)} cases vs saved {len(saved)} "
              f"(only recomputed: {sorted(got - saved)[:10]}; only saved: {sorted(saved - got)[:10]})")
        return False
    print(f"  ok {label}: {len(got)} cases, identical to the saved list")
    return True


# ------------------------------------------------------------------ cohort flow
def cohort_flow(clin):
    flow, ok = [], True
    s1 = set(clin['caseid'])
    flow.append(('All VitalDB cases', len(s1), '', ''))
    s2 = set(clin.loc[clin['ane_type'] == 'General', 'caseid'])
    flow.append(('General anesthesia', len(s2), len(s1 - s2), 'not general anesthesia'))
    s3 = s2 - set(clin.loc[clin['opname'].isin(EXCLUDED_OPS), 'caseid'])
    flow.append(('Excluding liver transplantation / donor hepatectomy', len(s3), len(s2 - s3),
                 'liver transplantation or donor hepatectomy'))
    rec = clin.set_index('caseid')
    long_enough = set(rec.index[(rec['caseend'] - rec['casestart']) / 60 >= 45])
    s4 = s3 & long_enough
    flow.append(('Recording >= 45 min', len(s4), len(s3 - s4), 'recording < 45 min'))
    ok &= check('step 4 (recording length)', s4, ids('phase3_step4_caseids.csv'))

    rc = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_track_recheck.csv'))
    rc['missing_tracks'] = rc['missing_tracks'].fillna('')
    s5 = set(rc.loc[rc['has_all_core_tracks'].astype(str).str.lower() == 'true', 'caseid'].astype(int))
    bad = rc[~rc['caseid'].isin(s5)]
    err = bad[bad['missing_tracks'].str.startswith('ERROR')]
    miss = bad[~bad['missing_tracks'].str.startswith('ERROR')]
    has = lambda tr: miss['missing_tracks'].str.contains(tr, regex=False)
    art, bis = has('ART_MBP'), has('BIS/BIS') | has('BIS/EMG')
    other = has('PLETH_SPO2') | has('ETCO2') | has('Solar8000/HR')
    detail = (f"missing core track(s): arterial MAP only {int((art & ~bis & ~other).sum())}; "
              f"BIS/EMG only {int((bis & ~art & ~other).sum())}; MAP and BIS/EMG {int((art & bis & ~other).sum())}; "
              f"any involving HR/SpO2/ETCO2 {int(other.sum())}")
    if len(err):
        detail += f"; track list could not be downloaded {len(err)} (caseid {', '.join(map(str, err['caseid']))})"
    flow.append(('All six core tracks present', len(s5), len(s4 - s5), detail))
    ok &= check('step 5 (verified core tracks)', s5, ids('phase3_final_cohort_caseids.csv'))
    if set(rc['caseid']) != s4:
        print(f"  !! track recheck covered {rc['caseid'].nunique()} cases, step 4 has {len(s4)}")
        ok = False

    mp = os.path.join(RESULTS_DIR, 'phase4_case_missingness.csv')
    m = pd.read_csv(mp)
    cols = {'hr_missing_pct': 'HR', 'map_missing_pct': 'MAP', 'bis_final45_missing_pct': 'BIS final 45 min'}
    thr = {c: m[c].quantile(0.95) for c in cols}
    over = pd.DataFrame({c: m[c] > thr[c] for c in cols})
    s6 = set(m.loc[~over.any(axis=1), 'caseid'].astype(int))
    not_processed = s5 - set(m['caseid'].astype(int))
    parts = [f"{lab} > {thr[c]:.2f}% missing: {int(over[c].sum())}" for c, lab in cols.items()]
    detail = ("case-level missingness above cohort 95th percentile (" + "; ".join(parts) +
              f"; cases above >= 2 thresholds {int((over.sum(axis=1) >= 2).sum())})")
    if not_processed:
        detail += f"; not processed {len(not_processed)}"
    flow.append(('Final cohort (case-level missingness)', len(s6), len(s5 - s6), detail))
    ok &= check('step 6 (case missingness)', s6, ids('phase4_final_cohort_caseids.csv'))

    analysed = set()
    for h in ('5min', '10min', '15min'):
        analysed |= set(pd.read_parquet(os.path.join(P7, f'sqi_primary_w70_A_{h}_predictions.parquet'),
                                        columns=['caseid'])['caseid'].astype(int))
    gap = s6 - analysed
    why = []
    if gap:
        f = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'),
                            columns=['caseid', 't', 'HR_frac', 'ART_MBP_frac', 'PLETH_SPO2_frac', 'ETCO2_frac'])
        lab = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labels_v2.parquet'))
        lab = lab[lab['variant'] == 'sqi_primary']
        for c in sorted(gap):
            fc = f[f['caseid'] == c]
            passing = fc[(fc[['HR_frac', 'ART_MBP_frac', 'PLETH_SPO2_frac', 'ETCO2_frac']] >= 0.7).all(axis=1)]
            lc = lab[lab['caseid'] == c].merge(passing[['caseid', 't']], on=['caseid', 't'])
            obs = {h: int(lc[f'label_{h}'].notna().sum()) for h in ('5min', '10min', '15min')}
            if len(passing) == 0:
                reason = 'no window passes the 70% completeness rule'
            elif len(lc) == 0:
                reason = 'no window passing the 70% rule is eligible (outside anesthesia or active episode)'
            elif sum(obs.values()) == 0:
                reason = 'all labels censored (horizon beyond anesthesia end or BIS coverage < 50%)'
            else:
                reason = 'UNEXPECTED: observable labels exist'
            why.append(f"case {c}: {len(fc)} feature windows, {len(passing)} pass the 70% rule, {len(lc)} of these "
                       f"eligible, observable labels 5/10/15 min {obs['5min']}/{obs['10min']}/{obs['15min']} -> {reason}")
    flow.append(('Cases contributing analysis windows', len(analysed & s6), len(gap),
                 'no window with >= 70% completeness and an observable label' + (f" ({'; '.join(why)})" if why else '')))
    if analysed - s6:
        print(f"  !! {len(analysed - s6)} analysed cases are not in the final cohort")
        ok = False
    return pd.DataFrame(flow, columns=['step', 'n_remaining', 'n_excluded', 'reason']), s6, analysed, ok


def draw_flow(flow, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    n = len(flow)
    fig, ax = plt.subplots(figsize=(11, 1.55 * n + 0.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 1.55 * n + 0.2); ax.axis('off')
    for i, r in flow.iterrows():
        y = 1.55 * (n - 1 - i) + 0.35
        ax.add_patch(plt.Rectangle((0.2, y), 4.2, 0.9, fill=False, lw=1.2))
        ax.text(2.3, y + 0.45, f"{r['step']}\nn = {r['n_remaining']:,}", ha='center', va='center', fontsize=9)
        if i < n - 1:
            ax.annotate('', xy=(2.3, y - 0.65), xytext=(2.3, y), arrowprops=dict(arrowstyle='->', lw=1.1))
        if i > 0 and r['n_excluded']:
            ax.add_patch(plt.Rectangle((5.0, y + 0.95), 5.8, 0.55, fill=False, lw=0.8, ls='--'))
            txt = textwrap.fill(f"Excluded n = {int(r['n_excluded']):,}: {r['reason']}", 95)
            ax.text(5.1, y + 1.22, txt, ha='left', va='center', fontsize=6.3)
            ax.plot([2.3, 5.0], [y + 1.22, y + 1.22], lw=0.8, color='k')
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------ Table 1
def smd_cont(a, b):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    sd = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else np.nan


def smd_cat(a, b, level):
    p1, p2 = (a == level).mean(), (b == level).mean()
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / sd if sd > 0 else np.nan


def table1(clin, cohort, feats):
    d = clin[clin['caseid'].isin(cohort)].copy().set_index('caseid')
    for c in ('age', 'height', 'weight', 'bmi', 'asa', 'preop_hb', 'anestart', 'aneend', 'opstart', 'opend'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d.loc[d['asa'] == 6, 'asa'] = np.nan
    d['anesthesia_min'] = (d['aneend'] - d['anestart']) / 60
    d['surgery_min'] = (d['opend'] - d['opstart']) / 60
    g = feats.groupby('caseid')
    d['tiva'] = g['tiva_case'].first().reindex(d.index) if 'tiva_case' in feats else g['tiva_propofol'].max().reindex(d.index)
    d['remi_tci'] = (g['RFTN20_CE_mean'].max() > 0).astype(int).reindex(d.index)

    offset = (g['t_case_s'].first() - g['t'].first()).reindex(d.index)
    ev = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi_primary.csv'))
    ev['s'] = pd.to_timedelta(ev['start']).dt.total_seconds()
    ev = ev[ev['caseid'].isin(d.index)]
    ev['onset_case_s'] = ev['s'] + ev['caseid'].map(offset)
    ev = ev[(ev['onset_case_s'] >= ev['caseid'].map(d['anestart'])) & (ev['onset_case_s'] <= ev['caseid'].map(d['aneend']))]
    d['n_events'] = ev.groupby('caseid').size().reindex(d.index).fillna(0).astype(int)
    d['event_min'] = (ev.groupby('caseid')['duration_s'].sum() / 60).reindex(d.index).fillna(0)
    d['group'] = np.where(d['n_events'] > 0, 'event', 'no_event')
    d['asa_cat'] = d['asa'].map({1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'})

    groups = {'Overall': d, 'At least one event': d[d['group'] == 'event'], 'No event': d[d['group'] == 'no_event']}
    rows = []

    def cont(label, col, median=False, digits=1):
        r = {'variable': label}
        for k, x in groups.items():
            v = x[col].dropna()
            r[k] = (f"{v.median():.{digits}f} [{v.quantile(.25):.{digits}f}-{v.quantile(.75):.{digits}f}]" if median
                    else f"{v.mean():.{digits}f} ({v.std():.{digits}f})")
        miss = d[col].isna().sum()
        r['missing'] = f"{miss} ({miss/len(d)*100:.1f}%)" if miss else ''
        r['SMD'] = f"{smd_cont(groups['At least one event'][col], groups['No event'][col]):.2f}"
        rows.append(r)

    def binary(label, col):
        s = d[col]
        r = {'variable': label}
        for k, x in groups.items():
            n = int((x[col] == 1).sum())
            r[k] = f"{n} ({n/len(x)*100:.1f})"
        r['SMD'] = f"{smd_cat(groups['At least one event'][col], groups['No event'][col], 1):.2f}"
        r['missing'] = f"{s.isna().sum()} ({s.isna().mean()*100:.1f}%)" if s.isna().any() else ''
        rows.append(r)

    def cat(label, col, levels=None, top=None):
        s = d[col]
        lv = levels or list(s.value_counts().index[:top] if top else s.dropna().unique())
        rows.append({'variable': label, 'missing': f"{s.isna().sum()} ({s.isna().mean()*100:.1f}%)" if s.isna().any() else ''})
        shown = set()
        for l in lv:
            r = {'variable': f"   {l}"}
            for k, x in groups.items():
                n = int((x[col] == l).sum())
                r[k] = f"{n} ({n/len(x)*100:.1f})"
            r['SMD'] = f"{smd_cat(groups['At least one event'][col], groups['No event'][col], l):.2f}"
            rows.append(r); shown.add(l)
        rest = s.notna() & ~s.isin(shown)
        if rest.any():
            r = {'variable': '   other'}
            for k, x in groups.items():
                n = int((x[col].notna() & ~x[col].isin(shown)).sum())
                r[k] = f"{n} ({n/len(x)*100:.1f})"
            rows.append(r)

    rows.append({'variable': 'Cases, n', **{k: f"{len(x):,}" for k, x in groups.items()}})
    rows.append({'variable': 'Patients (unique subjectid), n', **{k: f"{x['subjectid'].nunique():,}" for k, x in groups.items()}})
    cont('Age, years, mean (SD)', 'age')
    cat('Sex, n (%)', 'sex', levels=['F', 'M'])
    cont('Height, cm, mean (SD)', 'height')
    cont('Weight, kg, mean (SD)', 'weight')
    cont('BMI, kg/m2, mean (SD)', 'bmi')
    cat('ASA physical status, n (%)', 'asa_cat', levels=['I', 'II', 'III', 'IV', 'V'])
    for c, lab in (('preop_htn', 'Hypertension, n (%)'), ('preop_dm', 'Diabetes, n (%)'), ('emop', 'Emergency surgery, n (%)')):
        if c in d:
            binary(lab, c)
    cont('Preoperative hemoglobin, g/dL, mean (SD)', 'preop_hb')
    cat('Department, n (%)', 'department', top=4)
    cat('Surgery type, n (%)', 'optype', top=6)
    cat('Surgical approach, n (%)', 'approach', top=3)
    binary('Propofol TCI (TIVA), n (%)', 'tiva')
    binary('Remifentanil TCI, n (%)', 'remi_tci')
    cont('Anesthesia duration, min, median [IQR]', 'anesthesia_min', median=True, digits=0)
    cont('Surgery duration, min, median [IQR]', 'surgery_min', median=True, digits=0)
    cont('Sustained BIS>=60 events during anesthesia per case, median [IQR]', 'n_events', median=True, digits=0)
    cont('Time in sustained BIS>=60 events, min, median [IQR]', 'event_min', median=True, digits=1)
    for r in rows[-2:]:
        r['SMD'] = ''   # groups are defined by these variables
    t1 = pd.DataFrame(rows).fillna('')
    return t1[['variable', 'Overall', 'At least one event', 'No event', 'SMD', 'missing']], d, len(ev)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clin = pd.read_csv(CLINICAL_SNAPSHOT)
    clin['caseid'] = clin['caseid'].astype(int)
    print("=== Cohort flow (recomputed from saved files) ===")
    flow, cohort, analysed, ok = cohort_flow(clin)
    old = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_attrition_table.csv'))
    for i in range(min(4, len(old))):
        if int(old['n_remaining'].iloc[i]) != int(flow['n_remaining'].iloc[i]):
            print(f"  !! step {i+1}: snapshot gives {flow['n_remaining'].iloc[i]}, phase3_attrition_table.csv says {old['n_remaining'].iloc[i]}")
            ok = False
    flow.to_csv(os.path.join(OUT_DIR, 'cohort_flow.csv'), index=False)
    draw_flow(flow, os.path.join(OUT_DIR, 'cohort_flow.png'))
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
    for _, r in flow.iterrows():
        exc = f"   (excluded {r['n_excluded']:,}: {r['reason']})" if r['n_excluded'] != '' and r['n_excluded'] else ''
        print(f"{r['n_remaining']:>6,}  {r['step']}{exc}")
    print("All cross-checks passed." if ok else "Some cross-checks FAILED — see '!!' lines above.")

    feats = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet'))
    keep = [c for c in ('caseid', 't', 't_case_s', 'tiva_case', 'tiva_propofol', 'RFTN20_CE_mean') if c in feats]
    feats = feats[keep]
    t1, d, n_ev = table1(clin, cohort, feats)
    t1.to_csv(os.path.join(OUT_DIR, 'table1.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'table1.md'), 'w', encoding='utf-8') as fh:
        fh.write('| ' + ' | '.join(t1.columns) + ' |\n|' + '---|' * len(t1.columns) + '\n')
        for _, r in t1.iterrows():
            fh.write('| ' + ' | '.join(str(v) for v in r.values) + ' |\n')
    print(f"\n=== Table 1 (final cohort, {len(d):,} cases; {n_ev:,} sustained BIS>=60 events with onset during anesthesia) ===")
    print(t1.to_string(index=False))

    rep = d.groupby('subjectid').size()
    multi = rep[rep > 1]
    pd.DataFrame({'subjectid': multi.index, 'n_cases': multi.values}).to_csv(os.path.join(OUT_DIR, 'repeat_patients.csv'), index=False)
    print(f"\nRepeat patients: {d['subjectid'].nunique():,} unique patients for {len(d):,} cases; "
          f"{len(multi)} patients contribute >1 case ({int(multi.sum())} cases, {multi.sum()/len(d)*100:.1f}%; max {rep.max()} cases)")
    if len(multi):
        folds = pd.read_csv(os.path.join(P7, 'outer_folds.csv')).set_index('caseid')['fold']
        dd = d[d['subjectid'].isin(multi.index)].copy()
        dd['fold'] = folds.reindex(dd.index)
        split = dd.groupby('subjectid')['fold'].nunique()
        print(f"  of these, {int((split > 1).sum())} patients have cases in different outer folds "
              f"(same patient in training and test of a fold)")
    print(f"\nSaved cohort_flow.csv, cohort_flow.png, table1.csv, table1.md, repeat_patients.csv to {OUT_DIR}")


if __name__ == '__main__':
    main()