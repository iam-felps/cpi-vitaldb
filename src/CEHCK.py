import pandas as pd
import numpy as np
import os

INTERIM_DIR = 'data/interim'
WINDOW_S = 300
STRIDE_S = 60

cohort_ids = pd.read_csv('results/phase4_final_cohort_caseids.csv')['caseid'].tolist()[:200]  # sample for speed

completeness_records = []
for cid in cohort_ids:
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    wide = pd.read_parquet(path)
    n = len(wide)
    for end in range(WINDOW_S, n, STRIDE_S):
        for track in ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2']:
            window = wide[track].iloc[end-WINDOW_S:end]
            completeness = window.notna().sum() / WINDOW_S
            completeness_records.append(completeness)

comp = pd.Series(completeness_records)
print(f"Total window-track observations: {len(comp)}")
print(f"Below 50% (already excluded/NaN'd currently): {(comp < 0.5).mean()*100:.2f}%")
print(f"Between 50-70% (currently KEPT, would be EXCLUDED at 70% threshold): {((comp >= 0.5) & (comp < 0.7)).mean()*100:.2f}%")
print(f"70%+ (kept under either rule): {(comp >= 0.7).mean()*100:.2f}%")