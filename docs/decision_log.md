# Decision Log — CPI Project

## Locked design decisions (from protocol v2)

1. **Outcome scope:** broader inadequate-depth framing (sustained BIS≥60 anywhere in the case), with phase stratification (early/mid/late) as a pre-specified sensitivity analysis — not a narrowed late-case-only emergence window.

2. **Matched negative sampling:** included as a secondary design, but purely as an internal robustness/construct-validity check within the paper. Not framed as a head-to-head comparison against Qian et al.

3. **Nonlinear HR features (DFA, sample entropy):** included in both Model A and Model B.

## Phase 0 decisions

4. **Python version:** 3.13 selected (Windows environment). 3.14 ruled out — CatBoost has no working install path on 3.14 as of testing (confirmed open compatibility issue, no published wheels). 3.13 confirmed compatible with CatBoost (supported since release 1.2.8) and expected compatible with the rest of the stack (LightGBM, miceforest, neurokit2, nolds, vitaldb) based on pre-install research; actual `pip install` served as final confirmation — all libraries installed successfully.

5. **Random seed:** 2311, applied globally wherever randomness is used (splits, class-weighting in place of SMOTE, bootstrap resampling).

6. **imbalanced-learn:** installed per the original Phase 0 library list, but not used in the pipeline — SMOTE was replaced with class-weighting (decision documented separately: linear interpolation between nonlinear HR dynamics features generates physiologically impossible synthetic trajectories).
## Phase 2 decisions

7. **Confirmed VitalDB track names** (verified programmatically via `vitaldb.VitalFile(caseid).get_track_names()`, not assumed from external literature):
   - BIS: `BIS/BIS`
   - BIS/EMG power: `BIS/EMG`
   - HR: `Solar8000/HR`
   - MAP: `Solar8000/ART_MBP`
   - SpO2: `Solar8000/PLETH_SPO2`
   - ETCO2: `Solar8000/ETCO2`
   - Propofol effect-site concentration: `Orchestra/PPF20_CE` (confirmed via a case drawn from `vitaldb.caseids_tiva`, since not all cases use TIVA/propofol)
   - Remifentanil effect-site concentration: `Orchestra/RFTN20_CE`

8. **Confirmed VitalDB clinical metadata fields** (via `vitaldb.load_clinical_data()`):
   - Age: `age`
   - BMI: `bmi`
   - ASA class: `asa`
   - Surgery type: `optype`
   - Anesthesia type: `ane_type`
   - Case start/end: `casestart`, `caseend`

9. **Track-discovery method note:** the package's `vitaldb.get_track_names()` global function relies on a deprecated index and returned an empty result — track names must be confirmed per-case via `VitalFile(caseid).get_track_names()` instead.

10. **Phase 2 download completed.** Full tracks-only download from the VitalDB open dataset, via `vitaldb.find_cases()` + `vitaldb.VitalFile(caseid, track_names).to_parquet()`, at native (unresampled) resolution, saved in raw long-format (columns: dt, dname, tname, unit, ivals, fvals, nval, sval, srate, gain, bias) — no wide-format reshape applied, per the "raw and untouched" principle; reshape deferred to Phase 3/4 preprocessing.
    - Cohort: 3,383 cases with all 6 core tracks present (HR, MAP, SpO2, ETCO2, BIS, BIS/EMG), out of 6,388 total VitalDB cases. TCI tracks (Orchestra/PPF20_CE, Orchestra/RFTN20_CE) included where present; expected NaN/absent for non-TIVA cases — handled by Phase 3 exclusion criteria and Phase 4 imputation scope, not filtered at download time.
    - Result: 3,383/3,383 downloaded successfully, 0 failures.
    - Total size: 0.92 GB in `data/raw/` (well under the ~5.5 GB extrapolated from an early small-sample test — later-downloaded cases were substantially lighter on average than the initial sample, a reminder that small-sample extrapolations for this dataset are unreliable).
