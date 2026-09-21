import pandas as pd
import os

RESULTS_DIR = 'results'
INTERIM_DIR = os.path.join('data', 'interim')

recheck = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_track_recheck.csv'))
verified_valid = set(recheck[recheck['has_all_core_tracks'] == True]['caseid'])
old_cohort = set(pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist())
# Note: phase3_final_cohort_caseids.csv was just overwritten above — re-derive dropped set separately
old_cohort_original = set(pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_step4_caseids.csv'))['caseid'].tolist())

# Actually, simplest: just remove interim files for any case NOT in the corrected cohort
removed = 0
for f in os.listdir(INTERIM_DIR):
    if f.endswith('.parquet'):
        cid = int(f.replace('.parquet', ''))
        if cid not in verified_valid:
            os.remove(os.path.join(INTERIM_DIR, f))
            removed += 1

print(f"Removed {removed} stale interim files")
print(f"Remaining interim files should now be <= {len(verified_valid)}")