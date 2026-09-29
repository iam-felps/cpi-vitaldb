# CPI — Deviations from Protocol v6 and Their Justification

Supplementary document. Protocol v6 was written before model development and is kept unchanged
as the pre-specified plan. Protocol v7 describes the study as executed. This document lists every
difference between the two, why it happened, where it is recorded in the decision log
(`docs/decision_log.md`, entry numbers in brackets), and its expected effect on the results.

Types of change:
- **Correction:** an error found and fixed during execution (data or code).
- **Specification:** v6 left a detail open or ambiguous; the choice made is recorded here.
- **Deviation:** a pre-specified element was changed, replaced or not done.
- **Addition:** an analysis not in v6, added after results were available. Reported as
  post hoc and interpreted as such.

## A. Data, cohort and preprocessing

| # | Protocol v6 | As executed (v7) | Type | Reason | Entry | Expected effect |
|---|---|---|---|---|---|---|
| A1 | Exclude neurosurgical, cardiovascular and liver-transplant procedures | Liver transplantation and donor hepatectomy excluded (178 cases); VitalDB contains no neurosurgical or true cardiac-surgery cases | Specification | Categories absent from the dataset (already noted in v6) | 13, 57 | None |
| A2 | Exclude cases < 45 min of *total surgical duration* | Applied as total *recording* length (caseend − casestart) < 45 min (40 cases) | Deviation (wording) | Recording length was the quantity implemented. Surgery lies within the recording, so a surgical-duration rule could only exclude the same or more cases | 13, 57 | Small; cases with recording ≥ 45 min but surgery < 45 min are retained |
| A3 | Exclude insufficient core-track availability | Applied per case from each file's own track list (`VitalFile.get_track_names`), requiring all six tracks (HR, MAP, SpO2, ETCO2, BIS, BIS/EMG); 3,066 of 5,825 retained | Correction | `vitaldb.find_cases()` index gave ~5% false positives (claimed 3,213 cases) | 21, 57 | Cohort corrected (159 wrongly included, 12 wrongly excluded) |
| A4 | — | One eligible case (3609) excluded because the VitalDB server returned HTTP 503 during the track check; a later recheck showed all six tracks. Kept excluded and reported | Deviation | Re-inclusion would shift the cohort-level 95th-percentile missingness thresholds and require re-running Phases 4–9 | 57 | Negligible (1 of 3,067 eligible cases) |
| A5 | EMG > 40 dB removed 63% of BIS ≥ 60 time (v6 §3.2 rationale); an earlier 20-case check gave 6.3% for SQI < 50 (decision log entry 18, not cited in v6) | Re-measured on the full aligned cohort: EMG > 40 dB removes 41.1% of BIS ≥ 60 time (4.1% of all valid BIS); EMG > 35 dB 49.6%; SQI < 50 5.6% (2.5% of all valid BIS); SQI < 15 0.2% | Correction | Earlier figures came from subsamples or pre-alignment data | 18, 43, 44 | Qualitative conclusion unchanged: EMG filtering preferentially removes genuine BIS ≥ 60 time (≈10-fold enrichment vs ≈2.3-fold for SQI < 50) |
| A6 | — | BIS, SQI and EMG tracks re-aligned to the common 1-s grid (v1 started each track's clock at its own first sample; BIS shifted > 60 s in 3.6% of cases, EMG in 59.8%). All BIS label variants, events, case exclusions and downstream results re-derived | Correction | Implementation error found on audit; verified fixed by exact reproduction of the interim EMG column (640,715/640,715 s) | 43, 45 | v1 results superseded; final cohort 2,673 (v1: 2,665) |
| A7 | SQI < 15 as a stricter alternative threshold (Test 4) | Run as planned, but on valid BIS readings SQI < 15 removes only 0.002% of samples, so Test 4 effectively compares SQI < 50 with no SQI filtering | Specification | Property of the data | 43, 52 | Test 4 interpreted accordingly |
| A8 | Case-level missingness thresholds to be derived (95th percentile); HR/MAP over the total surgical duration, BIS over the final 45 minutes | Derived as planned: HR 15.66%, MAP 25.89%, BIS 40.29%; 393 cases excluded → 2,673. HR/MAP missingness computed over the whole recording and BIS over the last 45 minutes of the recording, not of surgery | Specification (values) + Deviation (wording) | Recording boundaries were the quantity implemented, as for A2 | 45, 57 | Thresholds are relative (percentiles), so the choice of time base mainly changes which cases are extreme, not how many |
| A9 | Window-level exclusion: > 30% missing (primary); 10% / 50% sensitivity | Primary: every core vital sign ≥ 70% present in the window; Test 6 at ≥ 90% and ≥ 50%. Features are computed for windows ≥ 50% present so the alternatives can be applied by filtering | Correction (v1 used 50% by mistake; v2 conforms) | v1 placeholder threshold (entry 34) corrected in v2 | 34, 45, 52 | Conforms to v6 |
| A10 | LOCF ≤ 15 s; Test 5 re-runs at 10 s and 30 s | Primary LOCF ≤ 15 s (all-or-nothing per gap). Test 5 not re-run; replaced by a gap-length check on a 200-case sample: 99.96% of gaps ≤ 10 s, 0.01% in 10–15 s, 0.01% in 15–30 s | Deviation | The threshold choice affects only ~0.02% of gaps, which does not justify a full rebuild (Phases 4–9) | 16, 41 | None expected; reported as a data characterisation, not a test |
| A11 | Model A labs: Hb and Hct | Hb only | Deviation | VitalDB has no preoperative Hct field; Hct was not derived from Hb | 24 | Minimal: Model A without drug features performs like Model B, so Hb adds little (entry 47) |
| A12 | Drug concentrations as continuous TCI tracks (handling of periods without infusion not specified) | Effect-site concentrations are 0 when no infusion has started (including cases never given the drug); a real-time propofol-TCI flag (propofol TCI data seen before t) added to Model A; case-level technique kept for description only | Deviation | The v1 pipeline imputed concentrations for ~1,277 cases that never received TCI drugs, and a case-level TIVA flag encoded the future (look-ahead) | 45, 47 | Removes look-ahead (1,078 windows in 117 cases affected; Model A retrained; AUROC changed by +0.001 to +0.002) |
| A13 | DFA/sample entropy parameterised per Qian et al. | Qian et al. report no reproducible parameters (automated HCTSA features). DFA box sizes adapted from Peng et al. (1995): α1 4–16 s (7 sizes), α2 16–63 s (8 sizes), log-spaced, first-order detrending, ordinary least-squares fit; sample entropy m = 2, r = 0.2 × window SD. Validity rule: exponent set missing when < 0 or when HR has zero variance | Specification / Correction | Precedent not reproducible. The default robust (RANSAC) fit is randomised, so OLS was used for reproducibility. Negative exponents are undefined | 23, 45 | α2 changed in 1.9% of windows vs v1; 555 exponents set missing |
| A14 | Elapsed time since induction | Seconds since the EMR-recorded anesthesia start (`anestart`) | Specification | v1 measured from the first monitor sample | 45 | — |
| A15 | Grouping unit "patient ID (caseid)" | Folds grouped by case. The 2,673 cases come from 2,601 patients; 60 patients had > 1 case, and 43 had cases in different folds. Sensitivity analysis without these 98 cases | Specification + Addition | VitalDB `caseid` identifies surgeries; `subjectid` identifies patients | 57, 58 | AUROC changed by +0.0005 to +0.003 (no inflation) |

## B. Outcome and labelling

| # | Protocol v6 | As executed (v7) | Type | Reason | Entry | Expected effect |
|---|---|---|---|---|---|---|
| B1 | Positive if a sustained BIS ≥ 60 event begins within the horizon after t | Unchanged, plus: (R1) analysis period restricted to anesthesia — windows from `anestart` onward, and only events with onset in [`anestart`, `aneend`] count | Specification | BIS ≥ 60 before induction or after anesthesia end is not inadequate depth | 46 | Fewer positives than v1 |
| B2 | Event-in-progress rule: exclude windows inside an active (not-yet-sustained) BIS ≥ 60 episode | Implemented fully: window excluded if t lies inside a sustained event **or** BIS at t is ≥ 60 | Correction | v1 excluded only windows inside sustained events | 46 | 45,333 windows excluded (v1: 24,686) |
| B3 | Not specified | (R3) Censoring: the label is missing (not negative) if t + horizon runs past `aneend` or the end of the recording, or BIS is present for < 50% of the horizon | Specification | An unobservable horizon is not evidence of no event | 46 | 44,724 / 44,750 / 48,739 windows censored at 5 / 10 / 15 min |
| B4 | — | Surgery-period sensitivity analysis: labels rebuilt with windows and events restricted to [`opstart`, `opend`]; models refit with the primary folds and selected configurations (no re-tuning) | Addition | Most BIS ≥ 60 after surgery ends may be deliberate lightening, not inadequate depth | 49, 50 | Reported alongside the primary analysis |

## C. Modelling and validation

| # | Protocol v6 | As executed (v7) | Type | Reason | Entry | Expected effect |
|---|---|---|---|---|---|---|
| C1 | Normalisation step (robust sigmoid scaling) inside CV | No scaling for GBDTs (tree splits are invariant to monotone transforms); the logistic baseline uses median imputation and standardisation, refit per training partition | Deviation | Scaling has no effect on tree models; kept where it matters (baseline) | 47 | None for GBDTs |
| C2 | SMOTE and/or class weighting, nested; SMOTE-provenance leakage test | Class weighting only (`scale_pos_weight` = n_neg / n_pos per training partition; logistic `class_weight='balanced'`). SMOTE not used, so the provenance test does not apply | Deviation | Decision by the study team (2026-09-24): linear interpolation between nonlinear HR-dynamics features generates physiologically impossible synthetic trajectories | 47 | — |
| C3 | Algorithm chosen in the inner loop among XGBoost, LightGBM, CatBoost | As planned: 12 candidates (3 libraries × 4 settings of learning rate and depth/leaves, 200 trees) chosen by mean inner-fold AUROC | Specification | Grid kept small for computational feasibility | 47 | Top candidates typically within 0.005 inner AUROC |
| C4 | Outer/inner fold numbers not specified | 5 outer folds (case level, stratified by whether the case has any 15-min positive window), 3 inner folds; outer folds saved once and reused by every run | Specification | — | 47 | — |
| C5 | MICE with age, BMI, ASA | scikit-learn `IterativeImputer` on each model's baseline variables (Model A: age, BMI, ASA, Hb; Model B: age, BMI, ASA), refit on every inner and outer training partition; other missing values handled natively by the GBDT | Specification | `miceforest` incompatible with the installed LightGBM | 28, 47 | — |
| C6 | Whether the naive baseline is calibrated: to be decided | The baseline receives the same nested calibration as Models A and B | Specification | Identical treatment of all three models | 47 | — |
| C7 | Output: continuous 0–100 index | Calibrated probability (isotonic or Platt, selected per fold by cross-validated Brier score; isotonic chosen in 44 of 45 folds), reported as a probability; the 0–100 CPI is this probability × 100 | Specification | — | 47 | — |

## D. Statistical analysis

| # | Protocol v6 | As executed (v7) | Type | Reason | Entry | Expected effect |
|---|---|---|---|---|---|---|
| D1 | Test 1–2 AUROC: DeLong's test | Patient-level paired bootstrap (1,000 iterations; case-level resampling) for AUROC and AUPRC | Deviation | DeLong assumes independent observations; windows are clustered within cases (design effect 4–12, entry 55). v6 §5.7 already required patient-level testing | 49 | Wider, valid intervals |
| D2 | Test 2 horizon comparisons | Paired on windows eligible at both horizons | Specification | Horizons have different censored windows | 49 | Separates true degradation from different window sets |
| D3 | Shapiro–Wilk before parametric CIs | All intervals are bootstrap percentile intervals; the normality check is not needed and was not repeated in v2 (v1 found non-normal bootstrap distributions in some combinations) | Deviation | Percentile intervals make no normality assumption | 36 | None |
| D4 | Test 3 on the same held-out folds | Each label variant is compared on the windows it shares with the primary run, each keeping its own labels; same folds and primary configurations | Specification | Variants censor different windows | 52 | — |
| D5 | Test 7: vasopressor / fluid-bolus negative control (Mann–Whitney U) | Not performed | Deviation | VitalDB records only case-level total doses of bolus vasopressors, with no administration times | 37 | Reported as a data limitation |
| D6 | Matched sampling: 1–2 negatives per patient ≥ 20 min from any event, reused across horizons | Up to 2 windows per case, drawn at random (seed 2311), ≥ 20 min from the start and end of every event, ≥ 10 min apart, negative at 15 min and present at all horizons (5,304 windows, 2,659 cases); positives = all primary positive windows; scored with the saved out-of-fold predictions; class separation as standardised mean differences | Specification / Correction | v1 farthest-point rule always picked the first and last window of event-free cases | 53 | — |
| D7 | Phase stratification: onset relative to anesthesia, early 0–0.4 / mid 0.4–0.8 / late > 0.8; recall + AUROC/AUPRC | As planned; within-phase AUROC and event-level recall | Specification | — | 49, 50 | — |
| D8 | SHAP consistency: Spearman (with p-value) and Jaccard of top-50/top-100 features | Out-of-fold TreeSHAP on 2,000 test windows per fold; Spearman (with p-value) across horizons and folds; Jaccard of top-5/top-10 features | Deviation | Models have 19–23 features, so top-50/100 sets are undefined | 54, 59 | p-values computed by `phase9_shap_consistency_v2.py` (entry 59) |
| D9 | Effective sample size per Riley et al. / Wang et al. | ANOVA intraclass correlation of the label within cases, design effect with unequal-cluster correction (Kish; Eldridge), plus an empirical AUROC design effect (patient-level vs window-level bootstrap variance) | Specification | — | 55 | — |
| D10 | Latency on two hardware tiers | Both tiers on one laptop (Intel Core i7-1280P): all cores vs single thread pinned to one core | Specification (limitation) | Both configurations ran on the same laptop | 56 | Absolute latency on low-end hardware not measured |
| D11 | Multiplicity: only Test 1 (A vs B) confirmatory | Unchanged | — | — | — | — |
| D12 | 1,000 case-level bootstrap iterations for all CIs and tests | 1,000 for overall metrics and all paired tests; 500 for subgroup and phase-stratified CIs (script default, to be confirmed in the run logs) and for the effective-sample-size variances; the repeat-patient analysis resamples patients | Deviation (minor) | Computation time for many subgroup cells | 49, 55, 58 | Slightly wider Monte Carlo error in subgroup CIs |

## E. Analyses added after results were available (post hoc)

| # | Analysis | Reason | Entry |
|---|---|---|---|
| E1 | v6 pre-specified reliability diagrams and "decision-relevant operating points" without defining them. Added: a cross-fitted Youden operating point with sensitivity, specificity, PPV, NPV, alarms per hour, event-level sensitivity and lead time; calibration intercept and slope; decision curves | Clinical interpretability | 48 |
| E2 | Drug-feature ablation (Model A without propofol/remifentanil concentrations and TCI flag) | Explain the A–B difference | 47 |
| E3 | Reference models: elapsed time only; anesthetic course (elapsed time + drug features); elapsed-time ablation. Paired comparisons with Models A and B | Quantify how much discrimination the time course of anesthesia alone explains | 51, 52 |
| E4 | Surgery-period sensitivity analysis | See B4 | 49, 50 |
| E5 | Repeat-patient sensitivity analysis | See A15 | 58 |

## F. Items specified in v6 and still outstanding

| # | Item | Status |
|---|---|---|
| O1 | Descriptive comparison of excluded vs retained cases (selection-bias check, v6 §2) | Done (entry 61): retained cases are older, more often male, ASA III, thoracic and longer surgeries than cases excluded for data availability, reflecting the requirement for invasive arterial pressure |
| O2 | Spearman p-values for SHAP ranking consistency (v6 §7) | Done (entry 59): all p < 0.0001 |