import pandas as pd

miss_df = pd.read_csv('results/phase4_case_missingness.csv')
zero_bis_cases = miss_df[miss_df['bis_final45_missing_pct'] == 100]['caseid'].tolist()

# Check total BIS coverage (not just final 45 min) for these cases
import os
for cid in zero_bis_cases[:5]:
    raw = pd.read_parquet(f'data/raw/{cid}.parquet')
    bis = raw[raw['tname'] == 'BIS']['nval']
    print(f"Case {cid}: {len(bis)} total raw BIS samples, {bis.notna().sum()} non-null")