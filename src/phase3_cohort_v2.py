"""
Phase 3 (v2) — cohort selection, reproducible and safe to re-run. Replaces phase3_attrition.py,
whose track step used vitaldb.find_cases() (unreliable index, entry 21) and, if re-run,
overwrote the verified cohort list with a wrong one.

Steps 1-4 from the local clinical snapshot (data/raw/vitaldb_clinical_snapshot.csv):
  1 all cases; 2 ane_type == 'General'; 3 exclude liver transplantation / donor hepatectomy;
  4 recording length (caseend - casestart) >= 45 min           -> results/phase3_step4_caseids.csv
Step 5: all six core tracks present, verified per case from each file's own track list by
  src/phase3_recheck_tracks.py (results/phase3_track_recheck.csv; resumable, needs internet)
                                                               -> results/phase3_final_cohort_caseids.csv
Safety: an existing list is never overwritten. If it exists, the script checks the recomputed
list is identical and stops with a message if not. Rows marked ERROR in the recheck (download
failures) count as excluded and are listed; see entry 57 (case 3609).
Order on a fresh project: this script (writes the step-4 list) -> phase3_recheck_tracks.py ->
this script again (writes the step-5 list).
Usage: python src/phase3_cohort_v2.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import RESULTS_DIR, CLINICAL_SNAPSHOT

EXCLUDED_OPS = ['Liver transplantation', 'Donor hepatectomy']
STEP4 = os.path.join(RESULTS_DIR, 'phase3_step4_caseids.csv')
FINAL = os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv')
RECHECK = os.path.join(RESULTS_DIR, 'phase3_track_recheck.csv')


def write_or_verify(path, caseids, label):
    caseids = sorted(int(c) for c in caseids)
    if os.path.exists(path):
        old = sorted(pd.read_csv(path)['caseid'].astype(int))
        if old != caseids:
            print(f"STOP: {label}: existing {os.path.basename(path)} has {len(old)} cases, recomputed {len(caseids)}. "
                  f"Nothing overwritten. Investigate before changing the cohort.")
            sys.exit(1)
        print(f"  {label}: {len(caseids)} cases — identical to existing {os.path.basename(path)} (not rewritten)")
    else:
        pd.DataFrame({'caseid': caseids}).to_csv(path, index=False)
        print(f"  {label}: {len(caseids)} cases — written to {os.path.basename(path)}")


if __name__ == '__main__':
    if not os.path.exists(CLINICAL_SNAPSHOT):
        print(f"No local clinical snapshot — downloading once to {CLINICAL_SNAPSHOT}")
        os.makedirs(os.path.dirname(CLINICAL_SNAPSHOT), exist_ok=True)
        pd.read_csv("https://api.vitaldb.net/cases").to_csv(CLINICAL_SNAPSHOT, index=False)
    clin = pd.read_csv(CLINICAL_SNAPSHOT)
    n1 = len(clin)
    s = clin[clin['ane_type'] == 'General']
    n2 = len(s)
    s = s[~s['opname'].isin(EXCLUDED_OPS)]
    n3 = len(s)
    s = s[(s['caseend'] - s['casestart']) / 60 >= 45]
    print(f"1 All cases: {n1}\n2 General anesthesia: {n2}\n3 Excl. liver transplant/donor hepatectomy: {n3}\n"
          f"4 Recording >= 45 min: {len(s)}")
    write_or_verify(STEP4, s['caseid'], 'step 4')

    if not os.path.exists(RECHECK):
        print("Step 5 needs the per-case track check: run src/phase3_recheck_tracks.py, then this script again.")
        sys.exit(0)
    rc = pd.read_csv(RECHECK)
    if set(rc['caseid'].astype(int)) != set(s['caseid'].astype(int)):
        print("STOP: phase3_track_recheck.csv does not cover exactly the step-4 cases. Re-run phase3_recheck_tracks.py.")
        sys.exit(1)
    ok = rc['has_all_core_tracks'].astype(str).str.lower() == 'true'
    err = rc[rc['missing_tracks'].fillna('').str.startswith('ERROR')]
    print(f"5 All six core tracks present: {int(ok.sum())} (excluded {int((~ok).sum())}, "
          f"of which download errors: {len(err)} {list(err['caseid'])})")
    write_or_verify(FINAL, rc.loc[ok, 'caseid'], 'step 5')