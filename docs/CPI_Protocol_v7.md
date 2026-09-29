Title: Development and Internal Validation of the Consciousness Prediction Index (CPI): A Pragmatic Time-Series Forecasting Model for Impending Inadequate Anesthesia Depth

Note on scope: This is a single-center (VitalDB, single academic institution), retrospective, internally validated model-development study. No external or independent-site validation was performed. Claims throughout this protocol and in any resulting manuscript are scoped accordingly — see Section 10 (Scope, Generalizability, and Limitations).

Version note (v7): Protocol v7 describes the study **as executed**. Protocol v6 was written before model development and is kept unchanged as the pre-specified plan. Every difference between v6 and v7 — corrections, specifications of details v6 left open, deviations, and analyses added after results were available — is listed with its justification and decision-log entry in `docs/CPI_Protocol_Deviations_v6_to_v7.md` (cited below as "Deviations, item X"). The main changes:
- class weighting replaced SMOTE;
- the artifact-filtering figures were re-measured on the full, time-aligned cohort;
- the labelling rules (analysis period, active-episode rule, censoring) were specified;
- patient-level paired bootstrap tests replaced DeLong's test;
- the vasopressor negative control was found infeasible in VitalDB;
- a surgery-period sensitivity analysis, reference models and a repeat-patient analysis were added.

## **1. Context and Clinical Rationale**

### **The Clinical Gap in Anesthesia Depth Monitoring**

The standard of care for intraoperative consciousness monitoring relies on processed electroencephalographic (EEG) hardware, such as the Bispectral Index (BIS). These proprietary tools are cost-prohibitive, limiting their availability in low-resource and global health settings. Clinicians in these environments must rely on delayed clinical signs of inadequate depth, which carries risk of intraoperative awareness, autonomic stress response, and unsafe airway/positioning events, as well as compounding operational inefficiencies such as prolonged operating room turnover.

### **The Outcome: Inadequate Anesthesia Depth, Not Emergence Specifically**

This study's target outcome is a sustained BIS value ≥60 for at least 3 consecutive minutes, with onset during anesthesia — an inadequate anesthesia depth event. Such events can arise at any point during a case (e.g., a mid-case episode driven by a surgical stimulus outpacing anesthetic depth) and are not necessarily equivalent to emergence (the terminal, drug-washout-driven return of consciousness at case end). Framing this outcome as generalized inadequate depth, rather than emergence specifically, is both more accurate to what the BIS≥60 threshold captures and consistent with how comparable prior work on this same dataset has framed the equivalent target (Section 1.2). Where the events occur within the anesthetic is characterized by a pre-specified phase-stratified sensitivity analysis (Section 5.5) and, because BIS≥60 after surgery ends is often deliberate lightening rather than inadequate depth, by a surgery-period sensitivity analysis (Section 5.8).

### **The Limitations of Current AI Models**

Recent advancements in computational anesthesiology have attempted to bridge this gap using machine learning applied to high-resolution perioperative datasets. Existing studies exhibit limitations that constrain clinical adoption [additional references to be added prior to submission — this section must not be submitted without full citation support]:

1. **Concurrent Estimation vs. Forecasting:** Many models estimate the patient's current state rather than forecasting a future state, providing no lead time for preemptive clinical decision-making.
2. **Binary Classification Framing:** Framing inadequate depth as a binary classification ("Adequate: Yes or No") communicates less of the underlying physiological trajectory than a continuous risk index would.
3. **The "Low-Resource" Paradox:** Many high-performing models rely on continuous, computer-calculated drug concentrations from Target-Controlled Infusion (TCI) pumps, or on high-fidelity waveform digitization. Hospitals that cannot afford BIS monitors rarely have this equipment either, limiting real-world applicability in the settings that most need an alternative.

### **1.1 Precedent: The Hypotension Prediction Index (HPI)**

A structurally analogous precedent exists in the Hypotension Prediction Index (HPI), a commercial tool outputting a continuous, unitless 0–100 score forecasting a clinically significant drop in mean arterial pressure 5–15 minutes before it occurs. Its mechanism is not directly analogous to the model proposed here: HPI derives its signal from high-fidelity (≥100 Hz) arterial waveform morphology, whereas the CPI operates on standard-resolution (1 Hz) vital-sign numerics without requiring arterial-line waveform digitization. HPI is cited as conceptual precedent for a continuous, probabilistic, forward-looking clinical index — not as evidence of mechanistic similarity.

### **1.2 Positioning Relative to Prior Work: Qian et al. (2025)**

Qian and colleagues (Br J Anaesth 2025) published a closely related study on this same database, predicting BIS >60 events at 0/5/10/15-minute horizons from ECG-derived heart rate dynamics, using a highly comparative time-series feature-extraction framework (HCTSA) followed by a two-stage reduction to a compact 27-feature model, evaluated via grouped nested 10-fold cross-validation. This is directly relevant prior art and is cited and discussed explicitly — including where the present study's outcome definition, and the decision to frame it as inadequate depth rather than emergence, follows their precedent. The present study differs in the following ways:

|  | Qian et al. (2025) | Present study (CPI) |
| ----- | ----- | ----- |
| Input signal | ECG-derived HR only | Multi-parameter: HR, MAP, SpO2, ETCO2 (+ preoperative Hb and TCI effect-site concentrations in Model A) |
| Feature approach | Automated high-dimensional extraction (HCTSA), reduced via importance ranking + clustering | Hand-engineered rolling statistics plus a small, literature-defined, **fixed a priori** set of HR nonlinear/fractal descriptors (Section 4.1) |
| Output | Per-horizon binary classifier | Continuous, calibrated probability (0–100 index), per horizon |
| Sampling design | Matched case-control-style negative sampling (1–2 negatives/patient, ≥20 min from any event) | Continuous rolling-window evaluation across the anesthetic (primary), with a matched-sampling design as a secondary robustness analysis (Section 5.4) |
| Confounder testing | Phase-stratified robustness only | Phase-stratified analysis (adapted from their method, Section 5.5), surgery-period sensitivity analysis (Section 5.8), and reference models isolating the anesthetic time course (Section 5.9). A vasopressor/fluid-bolus negative control was pre-specified but is not possible in VitalDB (Section 7) |
| Acuity range | Not stratified by ASA | ASA and age-group subgroup reporting (Section 7) |
| Baseline comparator | None reported | Naive logistic-regression baseline (last value + trailing slope) evaluated identically within the same nested CV structure (Section 5.2) |
| Covariate finding | Found no consistent benefit from adding age/sex/BMI/ASA/comorbidities/anesthetic technique to an HR-only model | Cited as prior evidence bearing on this study's Model A vs. Model B hypothesis, which is tested directly with a formal paired hypothesis test (Section 7) |

Note on feature-set size: this study's feature set (19 features in Model B, 23 in Model A) is deliberately smaller than Qian et al.'s 27-feature model, consistent with the pragmatic/low-resource design goal (Section 1).

### **The Novel Solution: The Consciousness Prediction Index (CPI)**

This study proposes the Consciousness Prediction Index (CPI), a continuous, calibrated forecasting index for impending inadequate anesthesia depth, using windowed feature extraction (including nonlinear HR dynamics, Section 4.1) and gradient-boosted decision trees. A comparative design — a full-feature benchmark model against a pragmatic model restricted to basic hemodynamics, plus a naive baseline and reference models for context (Sections 5.2, 5.9) — tests whether forecasting is feasible without BIS, TCI, or waveform-level data. Given the single-center design, this is presented as a feasibility and internal-validity demonstration, not a claim of demonstrated real-world equity impact (Section 10).

## **2. Study Design & Cohort Definition**

This is a retrospective, observational time-series forecasting study using the open-access VitalDB high-resolution dataset (6,388 cases), reported according to TRIPOD+AI.

**Inclusion Criteria:** Patients undergoing surgical procedures under general anesthesia, ASA physical status Class I–V. Performance is reported overall and by ASA class and age group (Section 7).

**Exclusion Criteria** (applied in this order; counts in the flow diagram):

* Neurosurgical, cardiovascular, and liver transplant procedures. VitalDB contains no neurosurgical and no true cardiac-surgery cases (its "Vascular" category is peripheral vascular surgery; its "Thoracic surgery" department is pulmonary/oncologic, not cardiac), so the criterion was applied as liver transplantation and donor hepatectomy (Deviations A1).
* **Total recording length under 45 minutes** (case end − case start; Deviations A2). The 5-minute feature window plus the maximum 15-minute prediction horizon set an absolute floor of 20 minutes for a single labelled window; the 45-minute minimum is set above this floor because cases this short (e.g., brief cystoscopy, aborted procedures) typically lack a steady-state maintenance phase and a standard emergence trajectory.
* **Missing core tracks:** any of HR, MAP (invasive), SpO2, ETCO2, BIS, or BIS/EMG absent from the case file, verified per case from each file's own track list (Deviations A3). One eligible case whose track list could not be downloaded (server error) was excluded and is reported separately (Deviations A4).
* **Case-level missingness** above the cohort's 95th percentile for any of: HR (whole case), MAP (whole case), BIS (final 45 minutes) — Section 3.3.

**Target Definition:** A sustained BIS value ≥60 for at least 3 consecutive minutes (≥180 s), with onset during anesthesia (from the EMR-recorded anesthesia start to anesthesia end), on the SQI-filtered BIS signal (Section 3.2).

**Labelling Rules** (applied per 5-minute window ending at time *t*, per label variant; Deviations B1–B3):

* **R1 — Analysis period:** windows are eligible from anesthesia start onward; only events whose onset falls within [anesthesia start, anesthesia end] count as outcomes. BIS≥60 before induction or after anesthesia end is not an inadequate-depth event.
* **R2 — Event-in-progress rule:** a window is excluded if *t* lies inside a sustained event, or if BIS at *t* is already ≥60 (an episode in progress, even one that never reaches 3 minutes). This preserves the forecasting framing by excluding what would effectively be concurrent detection.
* **R3 — Censoring:** for each horizon *h*, the label is missing (not negative) if *t + h* runs past anesthesia end or the end of the recording, or BIS is present for fewer than 50% of the seconds in (*t*, *t + h*]. An unobservable horizon is not evidence of no event. Applied to both classes.
* **Positive** for horizon *h*: a sustained event begins in (*t*, *t + h*].

**Case-Level Attrition Reporting:** A TRIPOD+AI-style flow diagram reports case counts at each exclusion stage, reproduced from saved case lists by a single script. Baseline characteristics of excluded and retained cases are compared descriptively to assess selection bias (Deviations O1).

**Repeat patients:** VitalDB `caseid` identifies surgeries and `subjectid` identifies patients. The analysis unit is the case; the number of patients contributing more than one case is reported, and a sensitivity analysis removes cases of patients whose surgeries fall in different cross-validation folds (Section 5.3; Deviations A15).

**Sample Size and Precision Reporting:** Sample size is fixed by VitalDB availability rather than derived a priori. Achieved precision is reported via patient-level bootstrap 95% CIs (Section 7). Because the continuous rolling-window design produces many temporally correlated windows per case, the effective (clustering-adjusted) number of independent windows and events is estimated (Section 7) and used to contextualize precision.

## **3. Data Preprocessing & Artifact Management**

All tracks are placed on a common 1-second grid per case, starting at the earliest sample of any of the eight grid tracks (HR, MAP, SpO2, ETCO2, BIS, EMG, propofol and remifentanil concentrations) and binned on integer seconds, so that every track, including SQI, shares the same clock (Deviations A6).

### **3.1 Biological Plausibility Filters**

Raw values outside predefined physiological boundaries are classified as artifacts and set to missing: HR 30–200 bpm; MAP 30–160 mmHg; SpO2 50–100%; ETCO2 10–80 mmHg. BIS = 0 is treated as sensor-off/dropout and set to missing.

These bounds are compared against precedent in comparable VitalDB-based intraoperative forecasting literature (Table A). Most notably, the HR upper bound (200 bpm) is substantially more permissive than the 120 bpm cutoff used by Wang et al. (HypoBridCast); a 120 bpm ceiling would misclassify genuine intraoperative tachycardia as noise.

**Table A. Artifact-bound comparison against comparable VitalDB-based literature**

| Variable | This study | Wang et al. (HypoBridCast, 2026) | Rationale for this study's choice |
| ----- | ----- | ----- | ----- |
| HR | 30–200 bpm | <20 or >120 bpm flagged | 120 bpm would exclude genuine tachycardia |
| MAP | 30–160 mmHg | <20 or >160 mmHg | Broadly consistent |
| ETCO2 | 10–80 mmHg | <0 or >70 mmHg | 10 mmHg is more conservative |
| Negative-sampling buffer (secondary analysis, Section 5.4) | 20 min | 20 min (also used by Qian et al.) | Convergent across independent studies |
| BIS artifact rejection (Section 3.2) | SQI<50 (primary); SQI<15, EMG>40 dB, EMG>35 dB (Kim et al. 2022) and no filtering as sensitivity comparators | Not reported | Manufacturer-intended signal-validity metric; resolves the EMG/arousal confound (Section 3.2) |

### **3.2 Target Sanitization and Artifact-Rejection Method**

**Primary method: Signal Quality Index (SQI)-based filtering.** BIS values are excluded from event detection when the concurrent BIS SQI is below 50 (following the precedent of Huang et al. 2022, *Sensors*). SQI is the manufacturer's own signal-validity metric and resolves a specific confound: a genuine arousal produces high BIS, high EMG and high SQI, whereas mechanical/electrical artifact (e.g., electrocautery interference, sensor dislodgement) produces high BIS and high EMG but low SQI.

**Why the original EMG-threshold approach was revised.** The v5 protocol specified a standalone EMG>40 dB filter. Measured on the full, time-aligned cohort (3,066 cases; 38.2 million valid BIS seconds, of which 9.2% are BIS≥60), EMG>40 dB removes 4.1% of all valid BIS but 41.1% of all BIS≥60 time (about 10-fold enrichment); EMG>35 dB removes 49.6% of BIS≥60 time. By contrast, SQI<50 removes 2.5% of all valid BIS and 5.6% of BIS≥60 time (about 2.3-fold). (Protocol v6 cited 63% for EMG>40 dB, from earlier subsample or pre-alignment measurements; Deviations A5.) Elevated frontalis EMG is therefore concentrated during exactly the episodes this study aims to detect — consistent with EMG acting as a correlate of arousal and lightening anesthesia rather than as independent noise. A targeted literature review found no manufacturer-specified or widely used absolute EMG dB cutoff for BIS artifact rejection; the one clinical precedent for an EMG cutoff (Kim et al. 2022, *Ann Palliat Med*, EMG>35 dB during emergence) is used as a sensitivity comparator. SQI<50 is not perfectly neutral either (2.3-fold enrichment); this residual tradeoff is discussed in Section 10.

**Stored label variants.** Five BIS signals are built in one aligned pass, each with LOCF ≤15 s: SQI<50 filtered (primary); SQI<15 filtered; EMG>40 dB filtered; EMG>35 dB filtered; unfiltered (BIS = 0 removed only). Sustained events are detected on each variant separately. On valid BIS readings, SQI<15 removes only 0.002% of samples, so the SQI-threshold comparison (Test 4) effectively compares SQI<50 with no SQI filtering (Deviations A7).

### **3.3 Missing Data Imputation & Exclusion**

All imputation and calibration steps are fit only within the training partition of each cross-validation fold (Section 5.3), refit at both outer- and inner-fold level. No global, dataset-wide fit is performed at any stage.

* **Baseline data (MAR) via MICE:** scikit-learn `IterativeImputer` (MICE-equivalent) on each model's baseline/laboratory variables (Model A: age, BMI, ASA class, preoperative Hb; Model B: age, BMI, ASA class), fit on each training partition. ASA coded 6 is treated as missing. All other missing feature values are left missing and handled by the gradient-boosted trees' native missing-value splits.
* **Intraoperative tracks:** gaps ≤15 s are filled by last observation carried forward (all-or-nothing per gap: a longer gap is left entirely missing), a local per-case operation with no cross-fold leakage risk. The pre-specified 10 s / 30 s re-runs were replaced by a gap-length characterization: on a 200-case sample, 99.96% of gaps are ≤10 s, 0.01% fall in 10–15 s and 0.01% in 15–30 s, so the threshold cannot materially change the data (Deviations A10).
* **Drug concentrations:** propofol and remifentanil effect-site concentrations are 0 before any infusion data exist in the case (including cases never given the drug), rather than imputed; they are missing only when an infusion has started but the window has no sample (Deviations A12).
* **Window-level exclusion:** features are computed for windows in which a track is ≥50% present; the primary analysis keeps windows in which **every** core vital sign (HR, MAP, SpO2, ETCO2) is ≥70% present (≤30% missing). Sensitivity analysis (Test 6): ≥90% and ≥50% (Deviations A9).
* **Case-level exclusion:** thresholds set empirically at the 95th percentile of the cohort's missingness distribution: HR 15.66% and MAP 25.89% (whole recording), BIS 40.29% (final 45 minutes of the recording); 393 cases excluded (Deviations A8). Median BIS missingness in the final 45 minutes is about 23%, because recording often continues after BIS removal at emergence; the percentile rule is relative and unaffected.

## **4. Feature Engineering, Leakage Control, and the Comparative Paradigm**

Features are computed on 5-minute, strictly trailing windows ending at prediction time *t*, with a new window every 60 seconds.

* **Model A (the benchmark, 23 features):** all Model B features plus preoperative Hb; propofol and remifentanil effect-site concentrations (window mean); and a real-time propofol-TCI flag (1 if propofol TCI data have appeared in the case before *t*). Hematocrit was pre-specified but is not recorded in VitalDB (Deviations A11).
* **Model B (the pragmatic CPI, 19 features):** mean, slope (least-squares) and variance of HR, MAP, SpO2 and ETCO2 (12); HR DFA α1, α2 and sample entropy (3, Section 4.1); elapsed time since anesthesia start; age, BMI, ASA class. No BIS-adjacent, TCI-derived or laboratory features.
* **Naive baseline (reference comparator, Section 5.2):** last observed value and trailing 5-minute slope of HR, MAP, SpO2 and ETCO2 (8 features).

A window-length note: the 5-minute trailing window is a design choice, not an empirically validated optimum for this task; this is acknowledged as a limitation (Section 10).

### **4.1 Nonlinear (Fractal) HR Feature Set**

Motivated by evidence from prior work on this dataset that nonlinear HR dynamics carry signal for this class of outcome, three nonlinear HR descriptors are computed on the trailing 5-minute HR window (HR only; no comparable precedent exists for MAP, SpO2 or ETCO2):

* **DFA α1 and α2:** detrended fluctuation analysis with box sizes adapted from Peng et al. (1995) to 1 Hz sampling — α1 box sizes 4, 5, 7, 8, 10, 13, 16 s; α2 16, 19, 23, 28, 35, 43, 52, 63 s (log-spaced); first-order detrending, ordinary least-squares fit (`nolds`, `fit_exp='poly'`). Qian et al. report no reproducible fixed parameters (their features came from an automated HCTSA search), so standard literature conventions are used (Deviations A13). The default robust (RANSAC) fit is randomized and was not used.
* **Sample entropy:** embedding dimension m = 2, tolerance r = 0.2 × window SD (Richman & Moorman 2000).
* **Validity rule:** a DFA exponent is set to missing when it is undefined — α < 0, or zero HR variance in the window.

**Feature set is fixed a priori — no post hoc, data-dependent pruning.** All candidate nonlinear HR features are retained in both Model A and Model B. The cross-horizon SHAP consistency check (Section 7) is an interpretive check, not a feature-selection mechanism.

**Computational cost reporting:** per-window latency from raw 1-s signals to a calibrated probability (feature computation + inference, batch size 1) is measured for Models A and B as mean ± SD, 95th percentile (p95) and maximum, on two configurations: all cores, and a single thread pinned to one core (Section 9; Deviations D10).

### **4.2 Feature Leakage Audit**

Every feature is listed in the audit table (Section 8) with source track, window definition and confirmation of zero look-ahead. No feature derived from total case duration or time-to-case-end is included; elapsed time since anesthesia start is the only temporal feature. All drug features use only data before *t*.

## **5. Machine Learning Architecture & Internal Validation**

Gradient-boosted decision trees (GBDTs) were selected over deep sequence architectures as a deliberate tradeoff: interpretability (SHAP), strong performance on moderate-sized tabular data with missingness, and lower overfitting risk.

### **5.1 Algorithm Selection**

The choice among XGBoost, LightGBM and CatBoost is part of the inner cross-validation loop: 12 candidates (3 libraries × 4 settings of learning rate {0.05, 0.1} and tree size {depth 4, 6} or {15, 31 leaves}; 200 trees) are compared by mean inner-fold AUROC, using only outer-training data (Deviations C3).

### **5.2 Naive Baseline Comparator**

A logistic regression on the last observed value and trailing 5-minute slope of HR, MAP, SpO2 and ETCO2, evaluated identically to Models A and B at all horizons, within the same nested cross-validation (median imputation and standardization refit per training partition; class-balanced weights; same nested calibration; Deviations C1, C6).

### **5.3 Nested Cross-Validation and Leakage-Prevention Protocol**

Folds are grouped by case (VitalDB `caseid`; Deviations A15). Outer: 5 folds, assigned once at case level, stratified by whether the case has any 15-minute positive window, saved and reused by every analysis (Deviations C4). Inner: 3 case-level folds within each outer-training fold, stratified by case-level positive at the modelled horizon. The pipeline executes, in fixed order:

1. **Split** — outer train/test and inner train/validation, all at case level.
2. **Impute** — MICE on baseline/laboratory variables, fit on each training partition (outer and inner), applied without refitting to the held-out partition.
3. **Weight** — class weighting refit on each training partition: `scale_pos_weight` = negatives / positives (GBDTs); `class_weight='balanced'` (logistic baseline). SMOTE is not used (Deviations C2).
4. **Tune** — the inner loop selects library and hyperparameters (Section 5.1).
5. **Calibrate** — isotonic vs Platt selected by case-grouped 3-fold CV (Brier score) over the selected candidate's inner out-of-fold predictions, refit on all of them (Section 6).
6. **Fit and evaluate** — the selected configuration is refit on the full outer-training fold, its predictions on the untouched outer-test fold are calibrated, and every prediction is stored with its case and window time.

No feature scaling is applied for GBDTs (trees are invariant to monotone transforms; Deviations C1).

**Automated leakage checks** (run in every model/horizon combination): no case appears in both training and test at the outer or inner level; for models with imputed variables, MICE parameters differ across outer folds (identical values would indicate accidental global fitting).

### **5.4 Secondary Analysis: Matched Negative Sampling**

The primary models are re-evaluated (same trained models, saved out-of-fold predictions; no refitting) against a sparse negative set: per case, up to 2 windows drawn at random (fixed seed), each ≥20 minutes from the start and end of every sustained event in the case, ≥10 minutes apart, negative at the 15-minute horizon and eligible at all horizons; the same negative set is used for all horizons and models. Positives are all positive windows of the primary design. Random selection replaces a farthest-point rule, which picks the first and last window of event-free cases and biases elapsed time (Deviations D6). The analysis is reported alongside, not blended with, the primary evaluation, with a class-separation diagnostic (Section 7). The two designs are not expected to give comparable numbers.

### **5.5 Phase-Stratified Sensitivity Analysis**

Adapted from Qian et al.: each event is assigned a relative onset time within the anesthetic (0 = anesthesia start, 1 = anesthesia end) and categorized as early (<0.4), mid (0.4–0.8) or late (>0.8). Each window is assigned the phase of its time *t*. Discrimination (AUROC) and event-level recall are evaluated within each phase. This characterizes whether late-phase events — closer in character to emergence — behave differently, without redefining the primary outcome.

### **5.6 Handling Class Imbalance**

Class weighting (Section 5.3, step 3) reflects the clinical priority that missing a true event is costlier than an excess alarm. The resulting tradeoff is quantified with AUPRC and operating-point metrics (Section 7).

### **5.7 Bootstrap Resampling**

1,000 bootstrap iterations resampling cases, on pooled outer-test predictions, give 95% percentile CIs for overall performance metrics and all paired tests; subgroup and phase-stratified CIs use 500 iterations (Deviations D12). All hypothesis tests resample cases (paired where models share windows); no test operates on pooled window-level predictions, because windows within a case are correlated.

### **5.8 Surgery-Period Sensitivity Analysis (added)**

Because BIS≥60 after surgery ends is often deliberate lightening rather than inadequate depth, labels are rebuilt with the analysis period restricted to [surgery start, surgery end] (rules R1 and R3 use the surgical times), and models are refit reusing the primary outer folds and selected model configurations, without re-tuning (Deviations B4).

### **5.9 Reference Models and Ablations (added)**

To quantify how much discrimination the anesthetic time course alone explains, models are refit with the primary configurations and folds using: (a) elapsed time only; (b) elapsed time + drug features (the "anesthetic course"); (c) Model A without drug features; (d) Models A and B without elapsed time. Model A is compared with (b) and Model B with (a) by paired bootstrap, in both analysis periods (Deviations E2–E3).

## **6. Multi-Horizon Forecasting & Probability Calibration**

**Prediction Horizons:** 5, 10 and 15 minutes, trained and evaluated as separate horizon-specific models.

**Calibration Method Selection:** within each outer-training fold, isotonic regression and Platt scaling are compared on the inner-loop out-of-fold predictions of the selected candidate by a further case-grouped 3-fold cross-validation (Brier score); the method with the lower cross-validated Brier score is refit on all inner out-of-fold predictions and applied to the outer-test predictions only. The method may differ by fold, model and horizon. The CPI (0–100) is the calibrated probability × 100 (Deviations C7).

## **7. Statistical Analysis & Clinical Robustness**

All hypothesis tests and bootstrap procedures resample cases (Section 5.7).

**Discrimination:** AUROC and AUPRC with 95% CIs, per model (naive baseline, Model B, Model A) and horizon.

**Calibration:** Brier score, expected calibration error, calibration intercept and slope, reliability diagrams (deciles of predicted risk).

**Operating-point and clinical-utility metrics (added):** at a cross-fitted Youden threshold (each outer fold's threshold is chosen on the other four folds' predictions): sensitivity, specificity, PPV, NPV, F1, balanced accuracy, likelihood ratios, alarms per hour of eligible monitoring time, event-level sensitivity (share of events with ≥1 alarm in the horizon before onset) and lead time; decision curves across thresholds (Deviations E1).

**Hypothesis Test 1 — Model A vs. Model B (primary confirmatory comparison):** per horizon, patient-level paired bootstrap difference in AUROC and AUPRC (same windows, cases resampled once per iteration), with 95% CI and two-sided bootstrap p-value. The same test compares each model with the naive baseline. DeLong's test was pre-specified but assumes independent observations (Deviations D1).

**Hypothesis Test 2 — Horizon Comparisons:** paired bootstrap differences between horizon pairs within each model, on windows eligible at both horizons (Deviations D2).

**Subgroup Performance:** discrimination and calibration by ASA class (I, II, III, IV–V pooled) and age group (<40, 40–65, 65–80, ≥80 years), descriptive with bootstrap CIs; no formal testing between subgroups (small cells, multiple comparisons, spurious-finding risk). Interpreted as hypothesis-generating.

**Effective (Clustering-Adjusted) Sample Size:** (1) outcome design effect from the ANOVA intraclass correlation of the window label within cases, with the unequal-cluster-size correction (Kish; Eldridge et al. 2006), giving effective windows, effective positives and effective positives per candidate feature; (2) empirical design effect of AUROC: variance under the case-level bootstrap divided by variance under a naive window-level bootstrap (Deviations D9).

**Phase-Stratified Sensitivity Analysis:** Section 5.5. **Surgery-Period Sensitivity Analysis:** Section 5.8. **Reference Models:** Section 5.9.

**Matched-Sampling Secondary Analysis:** Section 5.4, with a class-separation diagnostic: standardized mean difference between matched positive and negative windows for elapsed time, key hemodynamic features and drug concentrations (analogous to the "ΔMean" diagnostic of Chaari et al., 2026), so readers can judge how much of any performance difference reflects an easier, more separated task.

**Hypothesis Test 3 — Artifact-Filtering Sensitivity Analysis:** paired bootstrap difference in AUROC and AUPRC between the primary SQI<50 labels and each comparator (EMG>40 dB, EMG>35 dB, unfiltered), each run refit with the primary configurations and folds, compared on the windows the two runs share, each keeping its own labels (Deviations D4). Reported as a substantive methodological finding.

**Hypothesis Test 4 — SQI Threshold Sensitivity:** as Test 3, SQI<50 vs SQI<15 (Deviations A7).

**Hypothesis Test 5 — LOCF Threshold Sensitivity:** replaced by the gap-length characterization in Section 3.3 (Deviations A10).

**Hypothesis Test 6 — Window-Missingness Sensitivity:** as Test 3, ≥70% (primary) vs ≥90% and ≥50% window completeness.

**Hypothesis Test 7 — Negative Control (not performed):** the pre-specified comparison of CPI during vasopressor or fluid-bolus administration vs matched windows without intervention cannot be constructed from VitalDB, which records only case-level total doses of bolus vasopressors, with no administration times (Deviations D5). Reported as a data limitation.

**Repeat-Patient Sensitivity Analysis (added):** saved predictions re-evaluated after removing all cases of patients whose surgeries fall in different outer folds; bootstrap resamples patients (Deviations A15, E5).

**Explainability and Cross-Horizon Consistency:** out-of-fold SHAP: for each outer fold the selected configuration is refit on the outer-training cases and exact TreeSHAP values (log-odds scale) are computed for a random sample of 2,000 outer-test windows, so no case is explained by a model trained on it. Mean |SHAP| rankings are compared across horizons and across folds by Spearman correlation (with p-value; Deviations O2) and Jaccard index of the top-5 and top-10 features (the models have 19–23 features, so the pre-specified top-50/100 sets are undefined; Deviations D8). SHAP summary and dependence plots characterize model behaviour consistent with plausible physiology, not causal mechanism.

**Achieved Precision Reporting:** CI widths discussed relative to the effective sample size and event count.

**Multiplicity:** Test 1's Model A vs. Model B comparison is the primary confirmatory comparison. All other comparisons (baseline comparisons, Tests 2–6, subgroups, and the added analyses) are robustness and sensitivity checks, not multiple tests of one confirmatory hypothesis, and are not corrected for multiplicity; analyses added after results were available are labelled post hoc.

## **8. Feature Audit Table**

All windows are 5 minutes, strictly trailing, ending at *t*; features recomputed every 60 s. "A/B" = model membership.

| Feature | Source | Window (relative to t) | Real-time available? | Model | Notes |
| ----- | ----- | ----- | ----- | ----- | ----- |
| HR, MAP, SpO2, ETCO2 — mean | 1-s grid, plausibility-filtered, LOCF ≤15 s | trailing 5 min | Yes | A, B | Computed if track ≥50% present |
| HR, MAP, SpO2, ETCO2 — slope | same | trailing 5 min | Yes | A, B | Least-squares slope per second |
| HR, MAP, SpO2, ETCO2 — variance | same | trailing 5 min | Yes | A, B | |
| HR DFA α1 | HR | trailing 5 min | Yes | A, B | Box sizes 4–16 s (7 sizes), OLS fit; missing if α<0 or zero HR variance |
| HR DFA α2 | HR | trailing 5 min | Yes | A, B | Box sizes 16–63 s (8 sizes), OLS fit; same validity rule |
| HR sample entropy | HR | trailing 5 min | Yes | A, B | m = 2, r = 0.2 × SD |
| Elapsed time since anesthesia start | EMR anesthesia start | instantaneous at t | Yes | A, B | Only temporal feature |
| Age, BMI, ASA class | Clinical record | preoperative | Yes | A, B | MICE within training folds; ASA 6 → missing |
| Preoperative Hb | Clinical record | preoperative | Yes | A | MICE within training folds; Hct not available |
| Propofol effect-site concentration | TCI pump (Orchestra PPF20_CE) | trailing 5 min | Yes | A | 0 before any propofol TCI data in the case |
| Remifentanil effect-site concentration | TCI pump (Orchestra RFTN20_CE) | trailing 5 min | Yes | A | 0 before any remifentanil TCI data in the case |
| Propofol TCI flag | TCI pump | any data before t | Yes | A | Real-time; the case-level TIVA indicator is descriptive only |
| HR, MAP, SpO2, ETCO2 — last value | 1-s grid | last observed second in window | Yes | Baseline | |
| HR, MAP, SpO2, ETCO2 — slope | 1-s grid | trailing 5 min | Yes | Baseline | |
| ~~Total case duration / time to case end~~ | — | — | No — excluded | — | Leakage risk |

*BIS, SQI and EMG are used only to build labels (Section 3.2); they are not predictive features.*

## **9. Software and Reproducibility**

Python 3.13.15; numpy 2.5.3; pandas 2.3.3; scikit-learn 1.9.1; LightGBM 4.7.0; XGBoost 3.4.1; CatBoost 1.2.10; nolds 0.6.3; scipy; matplotlib; psutil; vitaldb. SHAP values are computed with each GBDT library's native TreeSHAP implementation. Latency benchmarking: Intel Core i7-1280P laptop processor (14 cores, 20 threads, 1.8 GHz base), 34 GB RAM, Windows; configurations: all cores, and single thread pinned to one core. Random seed 2311 throughout. All preprocessing, feature-engineering, modelling and analysis code, the run order (README) and the decision log will be made available. A local snapshot of the VitalDB clinical table is stored with the data so cohort selection is reproducible.

## **10. Scope, Generalizability, and Limitations**

This study is conducted on a single-center, single-country, high-resource academic dataset (VitalDB) in which BIS monitoring, TCI infusion, and standardized anesthetic protocols were widely available. The pragmatic Model B uses only features plausibly available in lower-resource settings, but the physiological relationships it learns were observed exclusively in a well-resourced, single-institution population. Consequently:

* This study is development and internal validation, not external validation; no claim of demonstrated performance in low-resource or non-Korean, non-tertiary-academic populations should appear anywhere in the manuscript.
* The outcome is generalized inadequate anesthesia depth, not emergence specifically; the phase-stratified and surgery-period analyses characterize, but do not redefine, this scope.
* Applicability to low-resource settings is a motivating hypothesis for future external validation, not a result of this study.
* Liver transplantation and donor hepatectomy are excluded, and VitalDB contains no neurosurgical or cardiac-surgery cases, so the highest-acuity case types are underrepresented.
* The 5-minute window length is a design choice without a validated optimum for this task.
* The feature set is smaller than Qian et al.'s by design and fixed a priori; whether a larger feature space would change performance is not explored.
* **BIS-based ground truth and EMG/arousal correlation:** elevated frontalis EMG is concentrated during genuine BIS≥60 episodes (Section 3.2), so any BIS-based artifact filter trades noise removal against loss of genuine positive signal; even SQI<50 removes BIS≥60 time preferentially (2.3-fold). This is a limitation of BIS-based labelling in this domain generally.
* **Data availability:** VitalDB has no preoperative hematocrit and no timestamps for bolus vasopressor or fluid administration, so Hct was dropped and the negative control (Test 7) could not be performed.
* **Timing of the first prediction:** the first eligible window falls a median of about 15 minutes after the recorded anesthesia start, so induction is largely not covered.
* **Unit of analysis:** folds are grouped by case, not patient; a small number of patients contribute several cases (addressed by a sensitivity analysis).
* **Sample size:** effective events per candidate feature are reported and compared with published guidance for flexible machine-learning methods.
* **Computational benchmark:** both latency configurations ran on one laptop processor; absolute latency on low-end hardware was not measured.
* **Case 3609:** one eligible case was excluded because its track list could not be downloaded during cohort assembly.

## **11. Manuscript Discussion Reminders (Not Structural Design Changes)**

* **AnesNet ablation finding (Qi et al., 2026):** their non-BIS multichannel model (ABP/ECG/PLETH/ETCO2) achieved AUROC <0.63 versus 0.82–0.90 for BIS-inclusive models on a related deep-anesthesia forecasting task in the same database. Must be cited in the Introduction and/or Discussion as evidence bearing on the Model A vs. Model B hypothesis, noting that their outcome (BIS<40, excessive depth) is physiologically distinct from this study's (BIS≥60).
* **Class-separation diagnostic:** carry the matched-sampling interpretive caveat (a more separated task can raise performance without better models) into the Discussion.
* **EMG/arousal correlation finding:** deserves its own Discussion paragraph, using the full-cohort figures (Section 3.2).
* **Time course of the anesthetic:** discuss how much discrimination reflects when in the anesthetic risk is high (phase-stratified, surgery-period and reference-model results) and what the model does and does not anticipate.
* **Drug concentrations and anesthetic technique:** drug features partly encode technique (TIVA vs volatile); discuss the A–B difference in that light.
* **Cost-benefit of nonlinear HR features:** relate their computational cost (latency benchmark) to their contribution (SHAP).
* **Alarm burden:** relate operating-point alarm rates to clinical usability and to alarm-reduction strategies as future work.