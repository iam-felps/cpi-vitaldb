import pandas as pd
import os

cohort_ids = pd.read_csv('results/phase3_final_cohort_caseids.csv')['caseid'].tolist()[:20]

total_high_bis_time = 0
total_sqi_flagged_time = 0

for cid in cohort_ids:
    wide = pd.read_parquet(f'data/interim/{cid}.parquet')
    df_raw = pd.read_parquet(f'data/raw/{cid}.parquet')

    # We need SQI aligned to the same 1s grid as BIS_emg_unfiltered
    sqi_raw = df_raw[df_raw['tname'] == 'SQI'].set_index('dt')['nval']
    t0 = sqi_raw.index.min() if len(sqi_raw) > 0 else None
    if t0 is None:
        continue
    sqi_raw.index = pd.to_timedelta(sqi_raw.index - t0, unit='s')
    sqi_raw = sqi_raw.resample('1s').mean()
    sqi_aligned = sqi_raw.reindex(wide.index)

    high_bis_mask = wide['BIS_emg_unfiltered'] >= 60
    total_high_bis_time += high_bis_mask.sum()
    sqi_low_during_high_bis = (high_bis_mask & (sqi_aligned < 50)).sum()
    total_sqi_flagged_time += sqi_low_during_high_bis

print(f"Total BIS>=60 seconds (unfiltered): {total_high_bis_time}")
print(f"Seconds where SQI<50 during BIS>=60: {total_sqi_flagged_time} ({total_sqi_flagged_time/total_high_bis_time*100:.1f}%)")