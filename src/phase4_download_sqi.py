import vitaldb
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Downloading BIS/SQI for {len(cohort_ids)} cases...")

failed = []
downloaded = 0
skipped = 0

for i, cid in enumerate(cohort_ids):
    path = os.path.join(RAW_DIR, f"{cid}.parquet")
    existing = pd.read_parquet(path)

    if 'SQI' in existing['tname'].unique():
        skipped += 1
        continue

    try:
        vf = vitaldb.VitalFile(cid, ['BIS/SQI'])
        data = vf.to_pandas(['BIS/SQI'], interval=1, return_timestamp=True)
        sqi_long = pd.DataFrame({
            'dt': data['Time'],
            'dname': 'BIS',
            'tname': 'SQI',
            'nval': data['BIS/SQI'],
        })
        combined = pd.concat([existing, sqi_long], ignore_index=True)
        combined.to_parquet(path)
        downloaded += 1
    except Exception as e:
        failed.append((cid, str(e)))
        if len(failed) <= 5:
            print(f"FAILED case {cid}: {e}")  # surface errors immediately, not just at the end

    if (i + 1) % 200 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] downloaded={downloaded} skipped={skipped} failed={len(failed)}")

print(f"\nDone. Downloaded={downloaded} Skipped={skipped} Failed={len(failed)}")
if failed:
    print(f"First few failures: {failed[:5]}")