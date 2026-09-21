import vitaldb
import pandas as pd
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

CORE_TRACKS = [
    'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/PLETH_SPO2',
    'Solar8000/ETCO2', 'BIS/BIS', 'BIS/EMG',
]

OUTPUT_PATH = os.path.join(RESULTS_DIR, 'phase3_track_recheck.csv')
LOG_PATH = os.path.join(RESULTS_DIR, 'phase3_recheck_progress.log')

# Candidates: all cases that passed Phase 3 steps 1-4 (before the flawed
# track-availability filter), saved earlier as phase3_step4_caseids.csv
candidate_ids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_step4_caseids.csv'))['caseid'].tolist()
print(f"Re-checking {len(candidate_ids)} candidate cases against real track lists...")

# Resume support: load existing results if this was interrupted before
if os.path.exists(OUTPUT_PATH):
    existing = pd.read_csv(OUTPUT_PATH)
    done_ids = set(existing['caseid'].tolist())
    results = existing.to_dict('records')
    print(f"Resuming: {len(done_ids)} cases already checked.")
else:
    done_ids = set()
    results = []

remaining = [c for c in candidate_ids if c not in done_ids]
print(f"Remaining to check: {len(remaining)}")

start_time = time.time()
SAVE_EVERY = 50

for i, cid in enumerate(remaining):
    try:
        vf = vitaldb.VitalFile(cid)
        names = vf.get_track_names()
        missing = [t for t in CORE_TRACKS if t not in names]
        has_all_core = len(missing) == 0
        results.append({
            'caseid': cid,
            'has_all_core_tracks': has_all_core,
            'missing_tracks': ';'.join(missing) if missing else ''
        })
    except Exception as e:
        results.append({
            'caseid': cid,
            'has_all_core_tracks': False,
            'missing_tracks': f'ERROR: {e}'
        })
        with open(LOG_PATH, 'a') as f:
            f.write(f"ERROR case {cid}: {e}\n")

    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta_min = (len(remaining) - (i + 1)) / rate / 60 if rate > 0 else 0
        msg = f"[{i+1}/{len(remaining)}] elapsed={elapsed:.0f}s ETA={eta_min:.0f}min"
        print(msg)
        with open(LOG_PATH, 'a') as f:
            f.write(msg + '\n')

# Final save
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

final_df = pd.DataFrame(results)
n_verified_true = final_df['has_all_core_tracks'].sum()
print(f"\nDone. {len(final_df)} cases checked.")
print(f"Cases with all 6 core tracks (verified): {n_verified_true}")
print(f"Cases missing at least one core track: {len(final_df) - n_verified_true}")
print(f"Results saved to {OUTPUT_PATH}")