# Methods (draft v1 — journal-neutral, TRIPOD+AI structure)

*Drafting notes are in italics and square brackets; remove before submission. [REF] marks a citation still to be inserted. Word count of the text below, excluding notes: about 2,200; most journals will need it shortened or partly moved to the Supplement.*

## Study design and data source

We conducted a retrospective development and internal-validation study of prediction models using VitalDB, an open-access database of 6,388 non-cardiac surgical cases performed at Seoul National University Hospital, Seoul, Republic of Korea, between 2016 and 2017 [REF Lee et al. 2022, Sci Data]. VitalDB contains high-resolution intraoperative monitoring data and perioperative clinical information. *[CHECK: hospital, years and 'non-cardiac' are from memory, not from the project files — verify against Lee et al. 2022 before submission.]* The study is reported according to the TRIPOD+AI statement [REF]. The analysis followed a written protocol finalised before model development (version 6); all departures from it are listed with their reasons in Supplementary Table S1.

*[Ethics: VitalDB data are de-identified and publicly available; state the institutional review board's position for secondary analysis of public data (exemption or waiver) and cite the original VitalDB approval.]*

## Participants

We included cases performed under general anesthesia, of all ASA physical status classes. We excluded:
1. liver transplantation and donor hepatectomy (the database contains no neurosurgical or cardiac-surgery cases, which the protocol also excluded);
2. cases with a total recording length under 45 minutes;
3. cases lacking any of the six required monitoring tracks (heart rate [HR], invasive mean arterial pressure [MAP], peripheral oxygen saturation [SpO2], end-tidal carbon dioxide [ETCO2], BIS and BIS electromyography [EMG]), verified from each case file's track list;
4. cases with extreme missingness, defined empirically as missingness above the cohort's 95th percentile for HR or MAP over the whole case, or for BIS over the final 45 minutes of the recording.

Characteristics of retained cases and of cases excluded for data availability were compared descriptively using standardised mean differences. The 45-minute minimum exceeds the 20-minute floor needed for a single labelled window (5-minute feature window plus 15-minute maximum horizon), to exclude very short procedures that lack a maintenance phase.

## Data preprocessing

All tracks were placed on a common 1-second grid per case. Values outside physiological bounds were set to missing: HR 30–200 bpm, MAP 30–160 mmHg, SpO2 50–100% and ETCO2 10–80 mmHg (comparison with published bounds in Supplementary Table S2). BIS values of 0 were treated as sensor dropout. Gaps of 15 seconds or less were filled by last observation carried forward. Longer gaps were left missing in their entirety.

**BIS artifact filtering.** BIS values recorded while the signal quality index (SQI) was below 50 were excluded before outcome detection [REF Huang et al. 2022]. We used SQI rather than an EMG power threshold because, in this cohort, EMG thresholds removed genuine BIS elevations disproportionately:
- **EMG > 40 dB:** 4.1% of all valid BIS seconds, but 41.1% of seconds with BIS ≥ 60.
- **EMG > 35 dB** [REF Kim et al. 2022]: 49.6% of BIS ≥ 60 seconds.
- **SQI < 50:** 2.5% of all valid BIS seconds and 5.6% of BIS ≥ 60 seconds.

This pattern is consistent with frontalis EMG rising with arousal rather than acting as independent artifact. Four alternative label sets were built for sensitivity analyses: SQI < 15, EMG > 40 dB, EMG > 35 dB, and no filtering.

## Outcome

The outcome was an inadequate-depth event: BIS ≥ 60 sustained for at least 3 consecutive minutes (180 s), with onset between the recorded start and end of anesthesia, detected on the SQI-filtered BIS signal. For each horizon *h* (5, 10 and 15 minutes), a prediction window ending at time *t* was labelled positive if an event began within (*t*, *t* + *h*]. Three rules defined eligibility:
1. **Analysis period.** Windows were eligible from anesthesia start onward. BIS elevations beginning before induction or after the end of anesthesia were not counted as events.
2. **Event in progress.** Windows were excluded if *t* fell within a sustained event or if BIS at *t* was already ≥ 60, so the task remained forecasting rather than detection of an ongoing episode.
3. **Censoring.** A window's label was set to missing, rather than negative, when *t* + *h* extended beyond the end of anesthesia or of the recording, or when BIS was available for fewer than half of the seconds in the horizon.

The outcome was determined from BIS, which was not used as a predictor.

## Predictors

Predictors were computed on 5-minute windows ending at *t*, using only data recorded before *t*, with a new window every 60 seconds. Windows were analysed if every core vital sign was at least 70% present. Three models were developed:

- **Model B (pragmatic, 19 predictors):**
  - mean, least-squares slope and variance of HR, MAP, SpO2 and ETCO2;
  - three nonlinear HR descriptors: detrended fluctuation analysis short- and long-range exponents (DFA α1, box sizes 4–16 s; α2, 16–63 s; first-order detrending, least-squares fit [REF Peng et al. 1995]) and sample entropy (m = 2, r = 0.2 × window standard deviation [REF Richman & Moorman 2000]);
  - elapsed time since anesthesia start;
  - age, body mass index (BMI) and ASA class.

  DFA exponents that were undefined (negative, or zero HR variance) were set to missing.
- **Model A (benchmark, 23 predictors):** Model B plus preoperative hemoglobin; the window means of propofol and remifentanil effect-site concentrations calculated by the target-controlled infusion (TCI) pumps; and an indicator of whether propofol TCI had started before *t*.

  Effect-site concentrations were set to 0 before an infusion had started, including in cases in which the drug was never given. They were set to missing only when an infusion was running but no value was recorded in the window.
- **Naive baseline (8 predictors):** a logistic regression using the last observed value and 5-minute slope of HR, MAP, SpO2 and ETCO2.

The predictor set was fixed before model development, with no data-driven selection (full list with sources in Supplementary Table S3). No predictor used information about total case duration or time to the end of the case. Preoperative hematocrit, specified in the protocol, is not recorded in VitalDB.

## Sample size

The sample size was determined by data availability. Because consecutive windows from the same case are correlated, we estimated the effective sample size in two ways:
- **Label clustering:** a design effect based on the intraclass correlation of the outcome label within cases, with a correction for unequal numbers of windows per case [REF Kish 1965; Eldridge et al. 2006].
- **Precision of the reported metric:** the ratio of AUROC variance under case-level versus window-level bootstrap resampling.

We report effective events per candidate predictor [REF Riley et al.; van der Ploeg et al. 2014].

## Missing data

Missing age, BMI, ASA class and preoperative hemoglobin were imputed by multiple imputation by chained equations (scikit-learn IterativeImputer), fitted within each training partition only (ASA class coded 6 was treated as missing). All other missing predictor values were left missing and handled by the gradient-boosted trees' native missing-value handling. For the logistic baseline, missing values were imputed with training-partition medians.

## Model development

Models A and B were gradient-boosted decision trees. For each model and horizon, the library (LightGBM, XGBoost or CatBoost) and hyperparameters were selected from 12 candidates (3 libraries × 2 learning rates × 2 tree sizes; 200 trees) within nested cross-validation. Class imbalance was handled by weighting positive windows by the ratio of negative to positive windows in each training partition (balanced class weights for the logistic baseline); no resampling or synthetic oversampling was used.

**Nested cross-validation.**
- **Outer loop:** five folds, assigned at the case level and stratified by whether a case had any positive window at the 15-minute horizon. The same outer folds were used for every model, horizon and sensitivity analysis.
- **Inner loop:** within each outer training set, three case-level folds were used to select the candidate with the highest mean inner-fold AUROC.
- **Refitting:** imputation, class weights and models were refitted within each inner and outer training partition. No case appeared in both the training and test data at either level.

**Calibration.** Within each outer training set, isotonic regression and Platt scaling were compared by case-grouped three-fold cross-validation over the selected candidate's inner out-of-fold predictions. The method with the lower Brier score was refitted on all inner out-of-fold predictions and applied to that fold's outer-test predictions. The same procedure was applied to the logistic baseline. The Consciousness Prediction Index (CPI) is the calibrated probability multiplied by 100.

Each outer-test prediction was stored with its case and time. All reported performance measures use these pooled out-of-fold predictions.

## Performance evaluation

**Discrimination and calibration.** Discrimination was assessed by the area under the receiver operating characteristic curve (AUROC) and the area under the precision–recall curve (AUPRC). Calibration was assessed by the Brier score, expected calibration error, calibration intercept and slope, and reliability diagrams by deciles of predicted risk.

**Operating point.** For clinical interpretation, we chose an operating point by the Youden index, cross-fitted so that each outer fold's threshold was derived from the other four folds' predictions. At that point we report:
- sensitivity, specificity, positive and negative predictive values;
- alarms per hour of eligible monitoring time;
- event-level sensitivity (the proportion of events preceded by at least one alarm within the horizon) and lead time.

Decision curves summarise net benefit across thresholds [REF Vickers].

**Confidence intervals.** All 95% confidence intervals are percentile intervals from bootstrap resampling of cases (1,000 resamples; 500 for subgroup and phase-stratified estimates).

## Statistical analysis

All comparisons used paired bootstrap resampling of cases (1,000 iterations). In each iteration, cases were resampled once and the difference in AUROC or AUPRC between two sets of predictions on the same windows was computed. The 95% confidence interval and a two-sided p-value were derived from the resulting distribution. We did not use DeLong's test, specified in the protocol, because it assumes independent observations and would understate variance for correlated windows.

**Primary comparison.** Model A versus Model B at each horizon. Each model was also compared with the naive baseline.

**Horizon comparison.** Performance across horizons was compared within each model on windows eligible at both horizons.

**Subgroups.** Performance was reported descriptively by ASA class (I, II, III, IV–V) and age group (< 40, 40–65, 65–80, ≥ 80 years), without formal testing.

**Phase-stratified analysis.** Adapted from Qian et al. [REF]. Each event's onset was expressed relative to the anesthetic (0 = start, 1 = end) and classified as early (< 0.4), mid (0.4–0.8) or late (> 0.8). Discrimination and event-level recall were evaluated within each phase.

**Pre-specified sensitivity analyses.** Each was refitted with the primary configurations and folds and compared with the primary analysis on shared windows, each keeping its own labels:
1. alternative BIS artifact filtering (EMG > 40 dB, EMG > 35 dB, none);
2. a stricter SQI threshold (< 15);
3. window completeness of ≥ 90% and ≥ 50% instead of ≥ 70%.

The pre-specified re-analysis with alternative gap-filling limits (10 and 30 s) was replaced by a characterisation of gap lengths, because, in a 200-case sample, 99.96% of gaps were 10 s or shorter and the limit could therefore affect only about 0.02% of gaps.

The pre-specified negative-control analysis during vasopressor or fluid-bolus administration could not be performed, because VitalDB records only total doses of bolus vasopressors, without administration times.

**Matched negative sampling (secondary analysis).** The primary models' out-of-fold predictions were re-evaluated against a sparse negative set:
- **Negatives:** up to two randomly selected windows per case, each at least 20 minutes from any event and at least 10 minutes apart, shared across horizons [REF Qian et al.].
- **Positives:** all positive windows.
- **Class separation:** standardised mean differences between positive and negative windows.

**Additional analyses (not pre-specified).**
- **Surgery-period analysis:** labels were rebuilt with the analysis period restricted to the time between surgical start and end, and the models refitted with the primary folds and configurations, because BIS elevation after surgery ends is often intentional.
- **Reference models:** to quantify the contribution of the anesthetic time course, we refitted models on (a) elapsed time only and (b) elapsed time plus drug features, and compared them with Models B and A respectively. We also refitted Model A without drug features, and Models A and B without elapsed time.
- **Repeat patients:** we repeated the evaluation after removing all cases from patients with surgeries in different outer folds, resampling patients in the bootstrap.

**Multiplicity.** The Model A versus Model B comparison was the primary confirmatory analysis. All other analyses were secondary or sensitivity analyses and were not adjusted for multiple comparisons.

## Model interpretation

We computed SHAP values out of fold [REF Lundberg & Lee]. For each outer fold, the selected configuration was refitted on the training cases and exact TreeSHAP values were computed for 2,000 randomly selected test windows. We verified that refitted models reproduced the stored predictions exactly and that SHAP values summed to the model output. Features were ranked by mean absolute SHAP value. Ranking agreement across horizons and folds was assessed with Spearman's correlation (with p-values) and the Jaccard index of the five and ten top-ranked features.

## Computational cost

We measured per-window latency, from the raw 1-second signals of the last 5 minutes to a calibrated probability, one window at a time, for 1,000 windows after warm-up. Latency was measured with all processor cores available and with a single thread pinned to one core, on a laptop with an Intel Core i7-1280P processor and 34 GB of memory. The benchmark recomputed every feature and confirmed identity with the stored values.

## Software and code availability

Analyses used Python 3.13.15 with numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, nolds 0.6.3 and scipy. The random seed was fixed (2311). The complete code, run order and decision log will be made available at [repository URL].