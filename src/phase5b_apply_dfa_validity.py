"""
Phase 5b — apply the DFA validity rule (decision log entry 45) to the EXISTING
phase5_features_v2.parquet, so the 76-min Phase 5 v2 run does not need repeating.
The identical rule is built into phase5_feature_engineering_v2.py, so a from-scratch
rerun reproduces this table. Idempotent: running it twice changes nothing.

Rule: DFA alpha1/alpha2 set to missing when undefined — alpha < 0, or zero HR variance
in the window. Valid DFA-1 exponents lie in ~[0, 2]; values up to 2.056 are kept.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpi_config import PROCESSED_DIR
from phase5_feature_engineering_v2 import apply_dfa_validity, DFA_COLS

path = os.path.join(PROCESSED_DIR, 'phase5_features_v2.parquet')
table = pd.read_parquet(path)
before = {c: int(table[c].notna().sum()) for c in DFA_COLS}
removed = apply_dfa_validity(table)
table.to_parquet(path)

for c in DFA_COLS:
    r = removed[c]
    print(f"{c}: {r['total']} set to missing (alpha<0: {r['negative']}, zero HR variance: {r['zero_HR_variance']}); "
          f"non-null {before[c]} -> {int(table[c].notna().sum())} of {len(table)}")
    print(f"   remaining range: {table[c].min():.3f} to {table[c].max():.3f}")
print(f"Saved (in place): {path}")