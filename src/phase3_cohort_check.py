import pandas as pd

df = pd.read_csv("https://api.vitaldb.net/cases")
print("Shape:", df.shape)
print()
print("ane_type value counts:")
print(df['ane_type'].value_counts(dropna=False))
print()
print("optype value counts:")
print(df['optype'].value_counts(dropna=False))
print("Transplantation dx/opname samples:")
print(df[df['optype'] == 'Transplantation'][['dx', 'opname']].drop_duplicates().head(20))
print()
print("Vascular dx/opname samples:")
print(df[df['optype'] == 'Vascular'][['dx', 'opname']].drop_duplicates().head(20))
print()
print("Department value counts:")
print(df['department'].value_counts(dropna=False))
print("Thoracic surgery dx/opname samples:")
print(df[df['department'] == 'Thoracic surgery'][['dx', 'opname']].drop_duplicates().head(30))

# Step 2: general anesthesia only
step2 = df[df['ane_type'] == 'General']

# Step 3: exclude liver transplantation (the only applicable exclusion —
# no neurosurgery or true cardiac surgery cases exist in VitalDB)
step3 = step2[step2['opname'] != 'Liver transplantation']