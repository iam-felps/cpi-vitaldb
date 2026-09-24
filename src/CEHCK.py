import vitaldb

for cid in [1, 4, 7]:
    vf = vitaldb.VitalFile(cid, ['EVENT'])
    df = vf.to_pandas(['EVENT'], interval=1, return_timestamp=True)
    print(f"\n=== Case {cid} EVENT track ===")
    print(f"Shape: {df.shape}")
    non_null = df[df['EVENT'].notna()]
    print(f"Non-null entries: {len(non_null)}")
    if len(non_null) > 0:
        print(non_null.head(30))

import pandas as pd
df = pd.read_csv("https://api.vitaldb.net/cases")
print(df[['caseid', 'intraop_eph', 'intraop_phe', 'intraop_epi']].describe())
print(df[['intraop_eph', 'intraop_phe', 'intraop_epi']].head(10))