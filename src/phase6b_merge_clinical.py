import pandas as pd
import os

RESULTS_DIR = 'results'
PROCESSED_DIR = os.path.join('data', 'processed')

labeled = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
clinical = pd.read_csv("https://api.vitaldb.net/cases")[['caseid', 'age', 'bmi', 'asa', 'preop_hb']]

merged = labeled.merge(clinical, on='caseid', how='left')
merged.to_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))

print(f"Merged. Rows: {len(merged)}")
print("Non-null rates for clinical features:")
print(merged[['age', 'bmi', 'asa', 'preop_hb']].notna().mean())