import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def load_case_wide(caseid, tracks):
    path = os.path.join(RAW_DIR, f"{caseid}.parquet")
    df = pd.read_parquet(path)
    df = df[df['tname'].isin(tracks)]
    wide = df.pivot_table(index='dt', columns='tname', values='nval', aggfunc='first')
    return wide.sort_index()

sample_caseids = [1, 4, 7]

for cid in sample_caseids:
    wide = load_case_wide(cid, ['HR', 'BIS'])
    print(f"\n--- Case {cid} ---")
    print(wide.describe())

    t0 = wide.index.min()
    wide = wide.copy()
    wide.index = pd.to_timedelta(wide.index - t0, unit='s')

    # Downsample to a regular 5-second grid — for plotting/sanity-check only,
    # not the actual pipeline resampling decision (deferred to Phase 3/4)
    resampled = wide.resample('5s').mean()
    time_s = resampled.index.total_seconds()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    if 'HR' in resampled.columns:
        axes[0].plot(time_s, resampled['HR'])
        axes[0].set_ylabel('HR (bpm)')
        axes[0].set_title(f'Case {cid} — HR and BIS over time (5s resample, sanity check only)')
    if 'BIS' in resampled.columns:
        axes[1].plot(time_s, resampled['BIS'], color='orange')
        axes[1].set_ylabel('BIS')
        axes[1].set_xlabel('Time since case start (s)')
    plt.tight_layout()
    outpath = os.path.join(RESULTS_DIR, f'sanity_check_case_{cid}.png')
    plt.savefig(outpath)
    plt.close(fig)
    print(f"saved {outpath}")

    if 'BIS' in wide.columns:
        n_zero = (wide['BIS'] == 0).sum()
        n_valid = wide['BIS'].notna().sum()
        print(f"BIS=0 count: {n_zero} out of {n_valid}")