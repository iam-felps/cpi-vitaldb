import pandas as pd
import os
import vitaldb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

df = pd.read_csv("https://api.vitaldb.net/cases")

attrition = []

# Step 1: total cases
n0 = len(df)
attrition.append(("Total cases in VitalDB", n0))

# Step 2: general anesthesia only
step2 = df[df['ane_type'] == 'General'].copy()
attrition.append(("Exclude non-general-anesthesia", len(step2)))

# Step 3: exclude liver transplantation and donor hepatectomy
excluded_opnames = ['Liver transplantation', 'Donor hepatectomy']
step3 = step2[~step2['opname'].isin(excluded_opnames)].copy()
attrition.append(("Exclude liver transplant / donor hepatectomy", len(step3)))
# Note: neurosurgical and cardiovascular (cardiac) procedures remain eligible
# per protocol; VitalDB contains none in this dataset, so no filter is applied
# for these categories (documented, not a silent omission — see decision log).

# Step 4: exclude cases below 45 minutes total recording length
step3['recording_length_min'] = (step3['caseend'] - step3['casestart']) / 60
step4 = step3[step3['recording_length_min'] >= 45].copy()
attrition.append(("Exclude recording length < 45 min", len(step4)))

# Print attrition table
print("\n--- Attrition Table ---")
prev = None
for label, n in attrition:
    if prev is None:
        print(f"{label}: {n}")
    else:
        print(f"{label}: {n} (removed {prev - n})")
    prev = n

# Save attrition table
attrition_df = pd.DataFrame(attrition, columns=['step', 'n_remaining'])
attrition_df.to_csv(os.path.join(RESULTS_DIR, 'phase3_attrition_table.csv'), index=False)
print(f"\nSaved to {os.path.join(RESULTS_DIR, 'phase3_attrition_table.csv')}")

# Save the step-4 caseid list for the next steps (track-availability exclusion)
step4[['caseid']].to_csv(os.path.join(RESULTS_DIR, 'phase3_step4_caseids.csv'), index=False)

CORE_TRACKS = [
    'Solar8000/HR',
    'Solar8000/ART_MBP',
    'Solar8000/PLETH_SPO2',
    'Solar8000/ETCO2',
    'BIS/BIS',
    'BIS/EMG',
]

track_available_caseids = set(vitaldb.find_cases(CORE_TRACKS))

step4_caseids = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_step4_caseids.csv'))['caseid'].tolist()

step5_caseids = [c for c in step4_caseids if c in track_available_caseids]
attrition.append(("Exclude insufficient core track availability", len(step5_caseids)))

# Full attrition table print (steps 1-5)
print("\n--- Full Attrition Table ---")
prev = None
for label, n in attrition:
    if prev is None:
        print(f"{label}: {n}")
    else:
        print(f"{label}: {n} (removed {prev - n})")
    prev = n

# Save final attrition table (overwrite with full 5-step version)
attrition_df = pd.DataFrame(attrition, columns=['step', 'n_remaining'])
attrition_df.to_csv(os.path.join(RESULTS_DIR, 'phase3_attrition_table.csv'), index=False)

# Save the final locked caseid list
pd.DataFrame({'caseid': step5_caseids}).to_csv(
    os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv'), index=False
)
print(f"\nFinal cohort: {len(step5_caseids)} cases")
print(f"Saved to {os.path.join(RESULTS_DIR, 'phase3_final_cohort_caseids.csv')}")

final_cohort_ids = step5_caseids
final_df = df[df['caseid'].isin(final_cohort_ids)]

print("\n--- ASA Class Distribution (Final Cohort) ---")
print(final_df['asa'].value_counts(dropna=False).sort_index())

excluded_ids = set(df['caseid']) - set(final_cohort_ids)
excluded_df = df[df['caseid'].isin(excluded_ids)]

def summarize(d, label):
    print(f"\n--- {label} (n={len(d)}) ---")
    print("Age: mean", round(d['age'].mean(), 1), "sd", round(d['age'].std(), 1))
    print("Sex:", d['sex'].value_counts(normalize=True).round(3).to_dict())
    print("BMI: mean", round(d['bmi'].mean(), 1), "sd", round(d['bmi'].std(), 1))
    print("ASA distribution:", d['asa'].value_counts(normalize=True).round(3).sort_index().to_dict())
    print("Top 5 optype:", d['optype'].value_counts(normalize=True).round(3).head(5).to_dict())

summarize(final_df, "RETAINED cohort")
summarize(excluded_df, "EXCLUDED cases")

asa6_cases = final_df[final_df['asa'] == 6.0]
print("\n--- ASA=6 cases in final cohort ---")
print(asa6_cases[['caseid', 'age', 'dx', 'opname']].to_string())