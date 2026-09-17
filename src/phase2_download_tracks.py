import vitaldb
import pandas as pd
import os
import time

CORE_TRACKS = [
    'Solar8000/HR',
    'Solar8000/ART_MBP',
    'Solar8000/PLETH_SPO2',
    'Solar8000/ETCO2',
    'BIS/BIS',
    'BIS/EMG',
]

TCI_TRACKS = [
    'Orchestra/PPF20_CE',
    'Orchestra/RFTN20_CE',
]

ALL_TRACKS = CORE_TRACKS + TCI_TRACKS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # parent of src/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, '_download_log.txt')

caseids = vitaldb.find_cases(CORE_TRACKS)
print(f"Cases to download: {len(caseids)}")

failed = []
skipped = 0
downloaded = 0
start_time = time.time()

for i, cid in enumerate(caseids):
    outpath = os.path.join(OUTPUT_DIR, f"{cid}.parquet")
    if os.path.exists(outpath):
        skipped += 1
        continue
    try:
        vf = vitaldb.VitalFile(cid, ALL_TRACKS)
        vf.to_parquet(outpath)
        downloaded += 1
    except Exception as e:
        failed.append((cid, str(e)))
        with open(LOG_PATH, 'a') as f:
            f.write(f"FAILED case {cid}: {e}\n")

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"[{i+1}/{len(caseids)}] downloaded={downloaded} skipped={skipped} failed={len(failed)} elapsed={elapsed:.0f}s")

elapsed_total = time.time() - start_time
print(f"\nDone. Downloaded={downloaded} Skipped={skipped} Failed={len(failed)} Time={elapsed_total:.1f}s")
if failed:
    print(f"Failed caseids logged to {LOG_PATH}")

# --- Sanity checks on the test batch ---
if downloaded > 0:
    sample_id = caseids[0]
    sample_path = os.path.join(OUTPUT_DIR, f"{sample_id}.parquet")
    if os.path.exists(sample_path):
        df = pd.read_parquet(sample_path)
        print(f"\n--- Sanity check: case {sample_id} ---")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head())

    # Folder size for the test batch
    total_bytes = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith('.parquet')
    )
    print(f"\nTotal size of {downloaded} downloaded files: {total_bytes / 1e6:.2f} MB")
    if downloaded > 0:
        avg_size = total_bytes / downloaded
        avg_time = elapsed_total / downloaded
        print(f"Avg per case: {avg_size/1e6:.2f} MB, {avg_time:.2f} sec")
        print(f"Extrapolated for 3383 cases: ~{avg_size*3383/1e9:.2f} GB, ~{avg_time*3383/60:.1f} min")