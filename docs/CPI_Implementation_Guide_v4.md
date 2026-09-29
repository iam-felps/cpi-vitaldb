# **CPI Project: Step-by-Step Implementation Guide (v4)**

Version 4 records the pipeline **as executed** and replaces v3 as the working guide. v3 is kept for the history of how the plan evolved. The methods are defined in protocol v7; the differences from the pre-specified protocol v6 are in `docs/CPI_Protocol_Deviations_v6_to_v7.md`; every decision with its numbers is in `docs/decision_log.md`; the exact commands are in `README.md`.

Main changes from v3:
- **All tracks share one 1-second clock.** v1 shifted BIS/EMG/SQI in some cases; everything from Phase 4 onward was rebuilt.
- **Cohort track availability is verified per case,** not with `find_cases()`.
- **Labelling rules are specified:** anesthesia period, active-episode rule, censoring.
- **Drug features are real-time,** with no imputation for drugs never given.
- **Class weighting replaces SMOTE.**
- **Patient-level paired bootstrap replaces DeLong's test.**
- **Added analyses:** surgery-period sensitivity, reference models, repeat-patient check.

**Standing decisions (locked):**

* **Outcome:** sustained BIS≥60 (≥3 min) with onset during anesthesia, anywhere in the case; phase stratification and a surgery-period analysis characterize where events occur.
* **BIS artifact filtering:** SQI<50 primary; SQI<15, EMG>40 dB, EMG>35 dB and unfiltered as sensitivity comparators (full-cohort figures: EMG>40 dB removes 41% of BIS≥60 time vs 5.6% for SQI<50).
* **Matched negative sampling:** secondary internal robustness check, not a head-to-head comparison with Qian et al.
* **Nonlinear HR features (DFA α1/α2, sample entropy):** in both Model A and Model B, fixed a priori.
* **Class imbalance:** class weighting only (no SMOTE).
* **Unit of analysis:** the case; folds grouped by case, with a repeat-patient sensitivity analysis.

---

## **Phase 0 — Environment and Project Setup**

1. Python 3.13 in a dedicated virtual environment (`.venv`); packages: numpy, pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, nolds, scipy, matplotlib, psutil, vitaldb. Record exact versions (protocol v7 Section 9).
2. Folder structure: `data/raw/` (untouched downloads and the clinical snapshot), `data/interim/` (1-s grid per case), `data/processed/` (feature and label tables), `src/` (code; superseded scripts in `src/archive/`), `results/` (outputs), `docs/` (protocol, guide, deviations, decision log).
3. All shared constants live in `src/cpi_config.py`: paths, seed 2311, window 300 s, stride 60 s, horizons, core tracks, feature lists, completeness thresholds.
4. Git: commit every script change and every decision-log entry.
5. Run scripts from the project root in a terminal, with their arguments. The VS Code Run button cannot pass arguments.

**Done when:** environment imports cleanly, `cpi_config.py` exists, Git is initialized.

---

## **Phase 1 — Literature Foundation**

Bibliography covering: forecasting vs concurrent estimation; TCI dependence; HPI; Qian et al. (2025) and their methods; DFA/sample-entropy foundations (Peng et al. 1995; Richman & Moorman 2000); BIS SQI and EMG artifact precedent (Huang et al. 2022; Kim et al. 2022); Wang et al. (HypoBridCast 2026); Chaari et al. 2026; Qi et al. 2026 (AnesNet); effective-sample-size references (Kish 1965; Eldridge et al. 2006; Riley et al.); events-per-variable guidance for machine learning (van der Ploeg et al. 2014); TRIPOD+AI.

**Done when:** every claim flagged in protocol v7 Section 1 has a citation.

---

## **Phase 2 — Data Acquisition**

1. Download raw tracks per case with the `vitaldb` package (`VitalFile(caseid, tracks)`), stored unmodified in long format in `data/raw/`: HR, MAP (invasive), SpO2, ETCO2, BIS, BIS/EMG, BIS/SQI, propofol and remifentanil effect-site concentrations (Orchestra PPF20_CE, RFTN20_CE).
2. Store a local snapshot of the VitalDB clinical table (`data/raw/vitaldb_clinical_snapshot.csv`), created once and always read from disk afterwards.
3. Confirm track names per case with `VitalFile(caseid).get_track_names()`; the global index behind `find_cases()` is unreliable (~5% false positives).

**Done when:** raw tracks and the clinical snapshot are on disk.

---

## **Phase 3 — Cohort Definition**

1. `phase3_cohort_v2.py`: from the clinical snapshot — all cases (6,388) → general anesthesia (6,043) → excluding liver transplantation and donor hepatectomy (5,865) → recording ≥45 min (5,825). Writes the step-4 list.
2. `phase3_recheck_tracks.py`: per-case track check for all six core tracks (resumable; needs internet).
3. `phase3_cohort_v2.py` again: writes the verified cohort (3,066). It never overwrites an existing list; it only checks that the recomputed list is identical and stops if it isn't. Download failures count as excluded and are listed (one case, 3609, reported separately).

**Done when:** `phase3_final_cohort_caseids.csv` exists and the script reports it identical on re-run.

---

## **Phase 4 — Signal Processing, Labels' Raw Material, Cohort Lock**

1. `phase4_step1_reshape_plausibility.py`: 1-s grid per case (clock starts at the earliest sample of any track); plausibility bounds HR 30–200, MAP 30–160, SpO2 50–100, ETCO2 10–80 → missing.
2. `phase4_step3_locf.py`: LOCF on vital signs for gaps ≤15 s, all-or-nothing per gap.
3. `phase4_download_sqi.py` (if SQI was not in the original download), then `phase4_step2d_v2_aligned.py`: rebuild every BIS variant from raw on the common clock — SQI<50 (primary), SQI<15, EMG>40 dB, EMG>35 dB, unfiltered (BIS = 0 removed in all), each with LOCF ≤15 s. The script first verifies that rebuilding EMG reproduces the step-1 interim EMG exactly and stops without writing if not.
4. `phase4_step2e_event_detection_v3.py`: sustained BIS≥60 events (≥180 s) per variant.
5. `phase4_step4_case_missingness.py`: case-level missingness (HR and MAP whole case; BIS final 45 min); exclude above the cohort 95th percentile (HR 15.66%, MAP 25.89%, BIS 40.29%) → 2,673 cases.

**Checks:** diagnostics `check_track_alignment.py` and `check_filter_overlap.py` reproduce the alignment and filter-overlap figures (entries 43–44).

**Done when:** five BIS variants and event files exist for all cases, and `phase4_final_cohort_caseids.csv` has 2,673 cases.

---

## **Phase 5 — Feature Engineering**

1. `phase5_feature_engineering_v2.py`: 5-minute trailing windows every 60 s. Per track: mean, least-squares slope, variance, last value, completeness fraction (features computed if the track is ≥50% present). HR: DFA α1 (box sizes 4–16 s) and α2 (16–63 s) with OLS fit, sample entropy (m = 2, r = 0.2 SD), DFA validity rule. Elapsed time since anesthesia start. Clinical variables from the snapshot (ASA 6 → missing). Safety checks: raw time maps onto clinical case time; cached sample-entropy values reproduce exactly.
2. `phase5b_apply_dfa_validity.py`: applies the DFA validity rule in place (already built into step 1; no effect on a fresh run).
3. `phase5c_realtime_tci.py`: real-time drug features — concentration 0 before any infusion data, propofol-TCI flag from data seen before *t*; case-level technique kept as `tiva_case` (descriptive only). Safe to re-run.
4. Keep the feature audit table (protocol v7 Section 8) identical to `MODEL_A_FEATURES`, `MODEL_B_FEATURES` and `BASELINE_FEATURES` in `cpi_config.py`.

**Done when:** `phase5_features_v2.parquet` exists with real-time drug features, and the audit table matches the config.

---

## **Phase 6 — Labelling**

1. `phase6_labeling_v2.py` (primary, anesthesia period) and `phase6_labeling_v2.py --period surgery` (sensitivity analysis). All five label variants, long format.
2. Rules per window ending at *t*:
   - **R1 (analysis period):** windows from anesthesia start; only events with onset in [anesthesia start, anesthesia end] count.
   - **R2 (active episode):** exclude the window if *t* is inside a sustained event or BIS at *t* ≥60.
   - **R3 (censoring):** label missing if *t + h* passes anesthesia end or the end of the recording, or BIS covers <50% of the horizon.
3. Check the attrition and class-balance table it prints (primary SQI<50: 1.90% / 3.70% / 5.02% positive at 5 / 10 / 15 min).

**Done when:** label tables and attrition files exist for both periods.

---

## **Phase 7 — Nested Cross-Validation (highest leakage risk)**

`phase7_cv_pipeline_v2.py` — one pipeline for every run:

* Rows: labelled windows with every core vital sign ≥70% present (option `--min-window-frac`).
* Outer: 5 case-level folds, stratified by case-level 15-min positive, created by the primary run and saved to `results/phase7_v2/outer_folds.csv`; every later run reuses them.
* Inner: 3 case-level folds. Candidates: LightGBM, XGBoost, CatBoost × 4 settings (200 trees), chosen by mean inner AUROC.
* MICE on baseline/laboratory variables only, refit on every training partition; other missing values handled natively by the trees; logistic baseline uses median imputation + standardization.
* Class weighting (`scale_pos_weight` = negatives/positives; baseline `class_weight='balanced'`).
* Calibration: isotonic vs Platt by case-grouped 3-fold CV (Brier) over inner out-of-fold predictions; applied to outer-test predictions only.
* Built-in leakage checks: no case in both training and test (outer and inner); MICE parameters differ across outer folds.
* Every prediction is saved with case ID and window time.

Runs (options recorded in the first line of each run's log):

| Run | Options |
|---|---|
| Primary | none (Models A, B, baseline; 5/10/15 min) |
| Test 3 | `--variant emg40` / `emg35` / `unfiltered` with `--models A B --fixed-config-from sqi_primary_w70` |
| Test 4 | `--variant sqi15 --models A B --fixed-config-from sqi_primary_w70` |
| Test 6 | `--min-window-frac 0.9` / `0.5` with `--models A B --fixed-config-from sqi_primary_w70` |
| Surgery period | `--period surgery --fixed-config-from sqi_primary_w70` |
| Drug-feature ablation | `--models A --drop-features tiva_propofol PPF20_CE_mean RFTN20_CE_mean --fixed-config-from sqi_primary_w70` |
| Reference: timing only | `--models A --only-features elapsed_time_s --fixed-config-from sqi_primary_w70` (and with `--period surgery`) |
| Reference: anesthetic course | `--models A --only-features elapsed_time_s PPF20_CE_mean RFTN20_CE_mean tiva_propofol --fixed-config-from sqi_primary_w70` (and with `--period surgery`) |
| Elapsed-time ablation | `--models A B --drop-features elapsed_time_s --fixed-config-from sqi_primary_w70` |

The decision log states that these runs reused the primary configurations (Tests 3, 4, 6: entry 52; surgery period: entries 49–50; ablations and reference models: entries 47 and 51); confirm against each run's log header, which records the options actually used.

**Done when:** every run has predictions and configuration files for all its model/horizon combinations and its log reports "leakage checks passed".

---

## **Phase 8 — Calibration**

Calibration is nested inside Phase 7 (step 5 of the pipeline in protocol v7 §5.3), so there is no separate Phase 8 script. Check the selected method per fold in the `_config.csv` files (isotonic selected in 44 of 45 primary folds).

---

## **Phase 9 — Evaluation and Statistical Analysis**

All from saved predictions unless noted. All intervals and tests resample cases (1,000 iterations; 500 for subgroup and phase-stratified CIs by default); never test on pooled windows.

1. `phase9_diagnostic_metrics_v2.py --tag sqi_primary_w70` (and `_surgery`): AUROC/AUPRC with CIs; calibration (Brier, ECE, intercept, slope, reliability diagrams); cross-fitted Youden operating point (sensitivity, specificity, PPV, NPV, alarms per hour, event-level sensitivity, lead time); decision curves.
2. `phase9_tests_subgroups_v2.py --tag ...`: Test 1 (A vs B; each vs baseline), Test 2 (horizons, on windows eligible at both), ASA and age subgroups (descriptive), phase stratification (within-phase AUROC and event recall).
3. `phase9_compare_runs_v2.py`: Tests 3, 4, 6 and reference-model comparisons (paired, on shared windows, each run with its own labels).
4. `phase9_matched_sampling_v2.py`: matched negative sampling with class-separation diagnostic.
5. `phase9_shap_v2.py`: out-of-fold TreeSHAP (refits each fold's selected model; checks refit reproduces saved predictions and SHAP additivity); rankings and consistency across horizons and folds. Then `phase9_shap_consistency_v2.py` adds Spearman p-values from `shap_importance.csv`.
6. `phase9_effective_n_v2.py`: ICC design effect and empirical AUROC design effect.
7. `phase9_latency_benchmark_v2.py --tier workstation` and `--tier single`: per-window latency (recomputes features and checks them against stored values).
8. `phase9_table1_attrition_v2.py`: cohort flow (cross-checked against every saved case list), flow diagram, Table 1, repeat-patient count.
9. `phase9_repeat_patients_v2.py`: repeat-patient sensitivity analysis.

Not performed, with documented reasons: Test 5 re-runs (replaced by a gap-length check, entry 41); Test 7 negative control (no vasopressor timestamps in VitalDB, entry 37).

**Outstanding before submission:**
- A descriptive comparison of excluded vs retained cases on the final cohort.

**Done when:** every analysis above has a results file and a decision-log entry.

---

## **Phase 10 — Interpretability**

SHAP is produced in Phase 9 (step 5). Write the interpretation as "the model relies on X, consistent with known physiology of Y", never "this proves mechanism Y". Cross-check the SHAP story against the reference models and ablations: they must tell the same story before it goes in the paper.

---

## **Phase 11 — Manuscript Assembly**

1. Map each TRIPOD+AI checklist item to a manuscript section.
2. Methods from protocol v7; every deviation from v6 stated, with the deviations table as a supplement.
3. Results in this order:
   - cohort and Table 1;
   - primary discrimination and calibration;
   - Test 1–2;
   - operating points and clinical utility;
   - subgroups;
   - phase stratification and the surgery-period analysis;
   - reference models and ablations;
   - sensitivity analyses (Tests 3, 4, 6; Test 5 characterization);
   - matched sampling;
   - repeat patients;
   - SHAP;
   - effective sample size;
   - latency.
4. Discussion: see protocol v7 Section 11 and Limitations (Section 10). Framing to be agreed with co-authors before drafting.
5. Supplement:
   - deviations table;
   - feature audit table;
   - flow diagram;
   - artifact-bound comparison (Table A);
   - all sensitivity-analysis tables;
   - software versions and hardware.

**Done when:** every TRIPOD+AI item is addressed and every analysis run appears in Results or the Supplement, whatever its result.

---

## **Standing Rules**

* Never fit anything (imputer, scaler, calibrator, model, threshold) on data outside the training partition it belongs to — outer or inner.
* No data-dependent feature selection; the feature set is fixed a priori.
* Every threshold is derived, computed from the cohort's own distribution, or paired with a sensitivity analysis.
* All statistical tests resample cases; never test on pooled windows.
* Never overwrite raw data or an existing cohort list; every script reads from one stage and writes to the next.
* Log everything in the decision log, with numbers, the same day.
* When a result surprises you, investigate before moving on. In this project that rule caught the EMG/arousal confound, the track misalignment, the `find_cases()` false positives and the TCI look-ahead.