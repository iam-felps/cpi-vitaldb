import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

sample_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'))['caseid'].tolist()[:5]

for cid in sample_ids:
    raw = pd.read_parquet(os.path.join(RAW_DIR, f'{cid}.parquet'))
    mbp = raw[raw['tname'] == 'ART_MBP']['nval']
    if len(mbp) > 0:
        out_of_range = mbp[(mbp < 30) | (mbp > 160)]
        print(f"Case {cid}: {len(out_of_range)}/{len(mbp)} out of range")
        if len(out_of_range) > 0:
            print(f"  Out-of-range value distribution: min={out_of_range.min()}, max={out_of_range.max()}")
            print(f"  Sample values: {out_of_range.head(10).tolist()}")