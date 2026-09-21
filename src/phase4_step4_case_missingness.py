import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

cohort_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()
print(f"Computing case-level missingness for {len(cohort_ids)} cases...")

records = []
failed = []

for i, cid in enumerate(cohort_ids):
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    try:
        wide = pd.read_parquet(path)
    except Exception as e:
        failed.append((cid, str(e)))
        continue

    total_len = len(wide)
    if total_len == 0:
        continue

    # HR/MAP: missingness over the WHOLE case
    hr_missing_pct = wide['HR'].isna().sum() / total_len * 100
    map_missing_pct = wide['ART_MBP'].isna().sum() / total_len * 100

    # BIS: missingness over the FINAL 45 MINUTES only (2700 seconds)
    final_45min = wide.iloc[-2700:] if total_len >= 2700 else wide
    bis_missing_pct = final_45min['BIS'].isna().sum() / len(final_45min) * 100

    records.append({
        'caseid': cid,
        'hr_missing_pct': hr_missing_pct,
        'map_missing_pct': map_missing_pct,
        'bis_final45_missing_pct': bis_missing_pct,
    })

    if (i + 1) % 500 == 0:
        print(f"[{i+1}/{len(cohort_ids)}] done")

miss_df = pd.DataFrame(records)
miss_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_case_missingness.csv'), index=False)

print(f"\nDone. {len(records)} cases processed, {len(failed)} failed.")

# Compute 95th percentile thresholds
thresholds = {}
for col in ['hr_missing_pct', 'map_missing_pct', 'bis_final45_missing_pct']:
    p95 = miss_df[col].quantile(0.95)
    thresholds[col] = p95
    print(f"\n{col}:")
    print(miss_df[col].describe())
    print(f"95th percentile: {p95:.2f}%")

# Apply exclusion
excluded_mask = (
    (miss_df['hr_missing_pct'] > thresholds['hr_missing_pct']) |
    (miss_df['map_missing_pct'] > thresholds['map_missing_pct']) |
    (miss_df['bis_final45_missing_pct'] > thresholds['bis_final45_missing_pct'])
)
n_excluded = excluded_mask.sum()
final_cohort = miss_df[~excluded_mask]['caseid'].tolist()

print(f"\nCases excluded (above 95th percentile on any track): {n_excluded}")
print(f"Final cohort after Phase 4 case-level exclusion: {len(final_cohort)}")

pd.DataFrame({'caseid': final_cohort}).to_csv(
    os.path.join(RESULTS_DIR, 'phase4_final_cohort_caseids.csv'), index=False
)
if failed:
    print(f"First few failures: {failed[:5]}")