import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/phase5_features_full.parquet')

mask_a1 = (df['HR_dfa_alpha1'] < 0) | (df['HR_dfa_alpha1'] > 2)
mask_a2 = (df['HR_dfa_alpha2'] < 0) | (df['HR_dfa_alpha2'] > 2)

print(f"alpha1 invalidated: {mask_a1.sum()}")
print(f"alpha2 invalidated: {mask_a2.sum()}")

df.loc[mask_a1, 'HR_dfa_alpha1'] = np.nan
df.loc[mask_a2, 'HR_dfa_alpha2'] = np.nan

df.to_parquet('data/processed/phase5_features_full.parquet', index=False)

print("Non-null rates after fix:")
print(df[['HR_dfa_alpha1', 'HR_dfa_alpha2']].notna().mean())