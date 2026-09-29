"""
Phase 9 (v2) — selection-bias check: baseline characteristics of retained vs excluded cases
(protocol v6/v7 Section 2). Entry 61. Uses only the local clinical snapshot and saved case lists.

Groups (all 6,388 VitalDB cases):
  Retained                     final cohort (phase4_final_cohort_caseids.csv)
  Excluded, data availability  eligible by design (passed steps 1-3: general anesthesia, not liver
                               transplant/donor hepatectomy, recording >= 45 min) but excluded for
                               missing core tracks or extreme missingness
       of which: missing track(s)      not in phase3_final_cohort_caseids.csv
                 extreme missingness   in phase 3 cohort, not in final cohort
  Excluded by design           failed steps 1-3
Standardised mean differences (SMD) compare Retained with Excluded-data-availability, the
comparison relevant to selection bias (design exclusions are intended). |SMD| > 0.1 is a
conventional marker of imbalance.
Outputs: results/phase9_v2/selection_bias.csv and selection_bias.md
Usage: python src/phase9_selection_bias_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, CLINICAL_SNAPSHOT

OUT_DIR = os.path.join(RESULTS_DIR, 'phase9_v2')


def ids(name):
    return set(pd.read_csv(os.path.join(RESULTS_DIR, name))['caseid'].astype(int))


def smd_cont(a, b):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    sd = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else np.nan


def smd_cat(a, b, level):
    p1, p2 = (a == level).mean(), (b == level).mean()
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / sd if sd > 0 else np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = pd.read_csv(CLINICAL_SNAPSHOT)
    d['caseid'] = d['caseid'].astype(int)
    d = d.set_index('caseid')
    for c in ('age', 'height', 'weight', 'bmi', 'asa', 'preop_hb', 'anestart', 'aneend', 'opstart', 'opend',
              'casestart', 'caseend'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d.loc[d['asa'] == 6, 'asa'] = np.nan
    d['asa_cat'] = d['asa'].map({1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'})
    d['anesthesia_min'] = (d['aneend'] - d['anestart']) / 60
    d['surgery_min'] = (d['opend'] - d['opstart']) / 60

    final, p3, step4 = ids('phase4_final_cohort_caseids.csv'), ids('phase3_final_cohort_caseids.csv'), ids('phase3_step4_caseids.csv')
    assert final <= p3 <= step4 <= set(d.index), "case lists are not nested as expected"
    grp = pd.Series('Excluded by design', index=d.index)
    grp[list(step4 - p3)] = 'Excluded: missing track(s)'
    grp[list(p3 - final)] = 'Excluded: extreme missingness'
    grp[list(final)] = 'Retained'
    d['grp'] = grp
    data_excl = d['grp'].isin(['Excluded: missing track(s)', 'Excluded: extreme missingness'])

    groups = {'Retained': d[d['grp'] == 'Retained'],
              'Excluded, data availability (all)': d[data_excl],
              '  missing track(s)': d[d['grp'] == 'Excluded: missing track(s)'],
              '  extreme missingness': d[d['grp'] == 'Excluded: extreme missingness'],
              'Excluded by design': d[d['grp'] == 'Excluded by design']}
    A, B = groups['Retained'], groups['Excluded, data availability (all)']
    rows = [{'variable': 'Cases, n', **{k: f"{len(x):,}" for k, x in groups.items()}}]

    def cont(label, col, median=False, digits=1):
        r = {'variable': label}
        for k, x in groups.items():
            v = x[col].dropna()
            r[k] = (f"{v.median():.{digits}f} [{v.quantile(.25):.{digits}f}-{v.quantile(.75):.{digits}f}]" if median
                    else f"{v.mean():.{digits}f} ({v.std():.{digits}f})") if len(v) else ''
        r['SMD retained vs data-excluded'] = f"{smd_cont(A[col], B[col]):.2f}"
        rows.append(r)

    def cat(label, col, levels):
        rows.append({'variable': label})
        for l in levels:
            r = {'variable': f"   {l}"}
            for k, x in groups.items():
                n = int((x[col] == l).sum())
                r[k] = f"{n} ({n/len(x)*100:.1f})" if len(x) else ''
            r['SMD retained vs data-excluded'] = f"{smd_cat(A[col], B[col], l):.2f}"
            rows.append(r)

    def top(col, k):
        return list(B[col].value_counts().index[:k].union(A[col].value_counts().index[:k]))

    cont('Age, years, mean (SD)', 'age')
    cat('Sex, n (%)', 'sex', ['F', 'M'])
    cont('BMI, kg/m2, mean (SD)', 'bmi')
    cat('ASA physical status, n (%)', 'asa_cat', ['I', 'II', 'III', 'IV', 'V'])
    for c, lab in (('emop', 'Emergency surgery, n (%)'), ('preop_htn', 'Hypertension, n (%)'), ('preop_dm', 'Diabetes, n (%)')):
        if c in d:
            cat(lab, c, [1])
    cont('Preoperative hemoglobin, g/dL, mean (SD)', 'preop_hb')
    cat('Department, n (%)', 'department', top('department', 4))
    cat('Surgery type, n (%)', 'optype', top('optype', 5))
    cat('Surgical approach, n (%)', 'approach', ['Open', 'Videoscopic', 'Robotic'])
    cont('Anesthesia duration, min, median [IQR]', 'anesthesia_min', median=True, digits=0)
    cont('Surgery duration, min, median [IQR]', 'surgery_min', median=True, digits=0)

    t = pd.DataFrame(rows).fillna('')
    t.to_csv(os.path.join(OUT_DIR, 'selection_bias.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'selection_bias.md'), 'w', encoding='utf-8') as fh:
        fh.write('| ' + ' | '.join(t.columns) + ' |\n|' + '---|' * len(t.columns) + '\n')
        for _, r in t.iterrows():
            fh.write('| ' + ' | '.join(str(v) for v in r.values) + ' |\n')
    pd.set_option('display.width', 250)
    print(f"=== Retained vs excluded cases (all {len(d):,} VitalDB cases) ===")
    print(t.to_string(index=False))
    s = pd.to_numeric(t['SMD retained vs data-excluded'], errors='coerce')
    big = t.loc[s.abs() > 0.1, ['variable', 'SMD retained vs data-excluded']]
    print(f"\n|SMD| > 0.1 (retained vs excluded for data availability): {len(big)} rows")
    if len(big):
        print(big.to_string(index=False))
    print(f"\nSaved selection_bias.csv and selection_bias.md to {OUT_DIR}")


if __name__ == '__main__':
    main()