import pandas as pd
df = pd.read_parquet('data/processed/phase5_features_full.parquet')
print(df[['HR_mean', 'HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen']].describe())
print()
print("Non-null rates:")
print(df[['HR_mean', 'HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen']].notna().mean())