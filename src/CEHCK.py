import pandas as pd
import numpy as np
import os

INTERIM_DIR = 'data/interim'
cohort_ids = pd.read_csv('results/phase4_final_cohort_caseids.csv')['caseid'].tolist()[:200]

gap_lengths = []
for cid in cohort_ids:
    path = os.path.join(INTERIM_DIR, f"{cid}.parquet")
    wide = pd.read_parquet(path)
    for track in ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2']:
        # Note: this reads POST-LOCF(15s) data, so already-filled gaps won't show as NaN runs.
        # We need RAW pre-LOCF gap lengths instead - reconstruct from raw data
        pass

# Better: check raw (pre-LOCF) gap-length distribution directly from raw parquet
RAW_DIR = 'data/raw'
gap_lengths = []
for cid in cohort_ids:
    raw = pd.read_parquet(os.path.join(RAW_DIR, f"{cid}.parquet"))
    for track_raw, track_name in [('HR','HR'), ('ART_MBP','ART_MBP'), ('PLETH_SPO2','PLETH_SPO2'), ('ETCO2','ETCO2')]:
        series = raw[raw['tname'] == track_raw].set_index('dt')['nval']
        if len(series) < 2:
            continue
        t0 = series.index.min()
        series.index = pd.to_timedelta(series.index - t0, unit='s')
        series = series.resample('1s').mean()
        is_na = series.isna()
        group = (~is_na).cumsum()
        run_lengths = is_na.groupby(group).sum()
        run_lengths = run_lengths[run_lengths > 0]
        gap_lengths.extend(run_lengths.tolist())

gaps = pd.Series(gap_lengths)
print(f"Total gap runs: {len(gaps)}")
print(f"Gaps <=10s (filled under all 3 thresholds): {(gaps <= 10).mean()*100:.2f}%")
print(f"Gaps 10-15s (filled at 15s/30s, NOT at 10s): {((gaps > 10) & (gaps <= 15)).mean()*100:.2f}%")
print(f"Gaps 15-30s (filled at 30s only): {((gaps > 15) & (gaps <= 30)).mean()*100:.2f}%")
print(f"Gaps >30s (unfilled under any threshold): {(gaps > 30).mean()*100:.2f}%")