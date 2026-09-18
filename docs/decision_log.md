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

11. **Sanity-check plotting note:** naive line plots of raw (non-resampled) BIS/HR data intermittently failed to render correctly in matplotlib for some cases (blank or truncated traces), a reproducible but unresolved rendering artifact — not a data integrity issue (raw values were independently verified correct via direct array inspection in all affected cases). Resolved for visualization by resampling to a regular 5-second grid before plotting. This does not affect the raw stored data in `data/raw/`, which remains at native resolution per the Phase 2 "raw and untouched" principle.

12. **BIS coverage check (Phase 2, informing Phase 3 design):** across a sample of 100 downloaded cases, BIS coverage (active BIS duration relative to case duration, excluding BIS=0 dropout artifacts) is high — median 98.9%, mean 97.7%, only 2/100 cases below 80% coverage, 0/100 below 50%. Confirms partial-coverage cases (e.g., case 7, ~45% coverage) are outliers, not representative — Phase 3 exclusion/missingness thresholds should be informed by this distribution rather than anecdotal single-case inspection. Note: one case showed coverage slightly >100% (1.02), likely a minor BIS/HR timestamp alignment edge case, worth flagging but not immediately concerning at this sample size.

13. **Phase 3 cohort locked: 3,213 cases** (from 6,388 total VitalDB cases). Attrition: 6,388 → 6,043 (general anesthesia only) → 5,865 (exclude liver transplantation and donor hepatectomy) → 5,825 (exclude <45 min recording) → 3,213 (exclude insufficient core track availability: HR, MAP, SpO2, ETCO2, BIS, BIS/EMG).
    - **Scope note on step 3:** the guide's exclusion criterion (neurosurgical, cardiovascular, liver transplant) is only partially applicable to this dataset. VitalDB (this sample) contains no neurosurgery cases and no true cardiac surgery (its "Vascular" category is peripheral vascular surgery, not cardiac surgery with cardiopulmonary bypass; its "Thoracic surgery" department is pulmonary/thoracic oncologic surgery, not cardiac). These categories remain eligible per protocol; they are simply absent from the dataset, not excluded. The only applicable exclusion was liver transplantation and donor hepatectomy (living liver donor surgery, excluded alongside transplant recipients on Mand's instruction, despite being physiologically distinct from a transplant recipient).
    - **ASA class distribution, final cohort:** ASA I 21.2%, II 63.5%, III 14.1%, IV 0.9%, coded "6" 0.4% (11 cases, data-quality caveat below), 71 cases (2.2%) missing/NaN, retained as missing rather than imputed or excluded (ASA is descriptive only, not an exclusion criterion, per protocol).
    - **ASA=6 data-quality caveat:** 11 cases are coded ASA 6 in VitalDB, which in the standard ASA-PS scale denotes a declared brain-dead organ donor. Manual review of these 11 cases' dx/opname (7 empyema/bronchopleural-fistula debridements, 1 metastasectomy, 1 lung wedge resection, 1 incision and drainage) confirms none are organ-procurement cases — these are ordinary therapeutic thoracic-surgery patients. This is very likely a coding artifact or non-standard use of the ASA field in VitalDB's source data, not a true reflection of physical status. Report these 11 cases as a distinct "coded ASA=6 (data-quality caveat)" category in Table 1 rather than as genuine ASA VI, to avoid a factually incorrect implication (organ donor) reaching the manuscript.
    - **Selection bias (retained n=3,213 vs. excluded n=3,175), for manuscript Section 2 / Discussion:** retained cohort is measurably older (mean 58.9 vs 55.7 years), more male (55.4% vs 46.1%), and higher-ASA (ASA I 21.2% vs 36.2%; ASA III 14.1% vs 8.4%) than excluded cases. Excluded cases skew toward Colorectal (31%) and Breast (13.1%) surgery; retained cases spread more evenly across Major/Minor resection, Stomach, and Colorectal. This pattern is most plausibly driven by which surgery types/departments use BIS monitoring routinely (the track-availability exclusion, step 5) rather than by the general-anesthesia or liver-transplant exclusions. Practical implication: the model is trained on a population skewing older, more male, and higher surgical risk than the general VitalDB population — a genuine generalizability limitation, and arguably relevant context for the paper's "low-resource, pragmatic deployment" framing, since it implies real-world BIS-monitored populations may already look like this retained cohort rather than the general surgical population.