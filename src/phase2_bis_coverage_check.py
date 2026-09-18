import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

def bis_coverage_fraction(caseid):
    path = os.path.join(RAW_DIR, f"{caseid}.parquet")
    df = pd.read_parquet(path)
    hr = df[df['tname'] == 'HR']['dt']
    bis = df[(df['tname'] == 'BIS') & (df['nval'] > 0)]['dt']  # exclude BIS=0 dropout artifact
    if len(hr) == 0 or len(bis) == 0:
        return None
    case_duration = hr.max() - hr.min()
    bis_duration = bis.max() - bis.min()
    return bis_duration / case_duration if case_duration > 0 else None

# Sample 100 downloaded caseids
all_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')]
sample_files = all_files[:100]

results = []
for f in sample_files:
    cid = int(f.replace('.parquet', ''))
    frac = bis_coverage_fraction(cid)
    if frac is not None:
        results.append(frac)

coverage = pd.Series(results)
print(f"Cases analyzed: {len(coverage)}")
print(coverage.describe())
print("\nCoverage < 50%:", (coverage < 0.5).sum(), f"({(coverage < 0.5).mean()*100:.1f}%)")
print("Coverage < 80%:", (coverage < 0.8).sum(), f"({(coverage < 0.8).mean()*100:.1f}%)")
print("Coverage >= 95%:", (coverage >= 0.95).sum(), f"({(coverage >= 0.95).mean()*100:.1f}%)")
