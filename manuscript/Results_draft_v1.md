# Results (draft v1 — journal-neutral)

*Drafting notes are in italics and square brackets; remove before submission. All numbers come from the decision log or from the run outputs pasted in the project chats. Framing follows decision-log entry 60: the calibrated index and what it predicts lead; the methodological findings on label filtering and pooled discrimination come second; Model B is reported as a feasibility result. Decision-log entries are cited in brackets for checking and must be removed before submission.*

## Cohort and outcome events

Of 6,388 VitalDB cases, 2,673 were included (Figure 1) [57]:
- **Excluded by design:** 345 cases without general anesthesia, 178 liver transplantations or donor hepatectomies, and 40 cases with recordings shorter than 45 minutes.
- **Missing core tracks (2,759 cases):** arterial pressure only in 2,249; BIS/EMG only in 273; both in 220; HR, SpO2 or ETCO2 in 16. For one further case, the track list could not be downloaded.
- **Extreme missingness (393 cases):** above cohort 95th-percentile thresholds of 15.66% (HR), 25.89% (MAP) and 40.29% (BIS in the final 45 minutes).

The 2,673 cases came from 2,601 patients. Two cases contributed no analysable window, leaving 2,671 cases in the analyses.

**Characteristics (Table 1)** [57]:
- Mean age 59.2 (SD 14.6) years; 44.1% female; 62.9% ASA II.
- Mainly general (62.5%) and thoracic (29.3%) surgery.
- Propofol TCI used in 52.2% and remifentanil TCI in 84.9%.
- Median anesthesia duration 230 minutes (IQR 171–300).

**Comparison with cases excluded for data availability (Supplementary Table S4)** [61]:
- Included cases were older, more often male and ASA III, far more often thoracic surgery (29.3% vs 9.9%), and had longer surgery (median 160 vs 80 minutes).
- Breast (0.4% vs 13.3%) and colorectal surgery (10.5% vs 27.9%) were underrepresented.

**Events.** In total, 2,254 cases (84.3%) had at least one sustained BIS ≥ 60 event with onset during anesthesia: 3,930 events, median 1 (IQR 1–2) per case and 7.5 (4.0–13.0) minutes per case spent in events [57].
- Relative to the anesthetic, 1,462 events (37.2%) began early, 173 (4.4%) mid and 2,295 (58.4%) late (final fifth) [49].
- The first eligible prediction window fell a median of 15.2 minutes (IQR 9.6–24.1) after the recorded anesthesia start [47].
- Only about 250–290 of the early events were preceded by an eligible window [49].

**Analysis windows.** Of 615,526 five-minute windows, 420 preceded anesthesia start and 45,333 fell within an ongoing BIS elevation [46]. 513,825 windows met the completeness rule [47]. After censoring, 498,941, 497,959 and 493,316 windows were analysed at the 5-, 10- and 15-minute horizons [55]. Of these, 8,603 (1.72%), 17,243 (3.46%) and 23,514 (4.77%) were positive, from 1,984, 2,028 and 2,037 cases [55].

## Discrimination and calibration

**Discrimination (Table 2)** [48, 58]. Model A had AUROCs of:
- 0.866 (95% CI 0.858–0.874) at 5 minutes;
- 0.835 (0.826–0.844) at 10 minutes;
- 0.798 (0.790–0.807) at 15 minutes.

Model B reached 0.770 (0.761–0.780), 0.750 (0.740–0.760) and 0.732 (0.722–0.742), and the naive baseline 0.641 (0.631–0.652), 0.588 (0.578–0.597) and 0.561 (0.552–0.570).

AUPRCs were 0.176 (0.161–0.193), 0.239 (0.220–0.260) and 0.231 (0.213–0.251) for Model A, 0.088 (0.080–0.097), 0.118 (0.109–0.127) and 0.129 (0.121–0.138) for Model B, and 0.032, 0.048 and 0.060 for the baseline, against prevalences of 1.7–4.8%.

**Calibration.** All models were well calibrated (Table 2; reliability diagrams in Figure 2) [48]:
- calibration slopes were 1.02, 1.03 and 1.01 for Model A, 1.02, 1.05 and 1.02 for Model B, and 0.94, 0.92 and 0.90 for the baseline;
- calibration intercepts were between −0.064 and 0.004, and expected calibration error was at most 0.0022.

Brier scores for Model A (0.0153, 0.0293 and 0.0406) were about 10–12% lower than those of a model predicting only the prevalence, against about 4% for Model B and under 1% for the baseline. Isotonic regression was selected in 44 of 45 outer folds [47].

**Primary comparison (Model A vs Model B).** Model A discriminated better than Model B at every horizon [49]:
- ΔAUROC +0.096 (0.088–0.104), +0.085 (0.077–0.093) and +0.067 (0.060–0.074);
- ΔAUPRC +0.088, +0.121 and +0.102;
- all p < 0.001.

Both models outperformed the naive baseline at all horizons (ΔAUROC +0.225 to +0.247 for Model A and +0.129 to +0.171 for Model B; all p < 0.001) [49].

**Horizons.** On windows eligible at both horizons, Model A's AUROC declined from 5 to 15 minutes (−0.032, p < 0.001), whereas Model B's did not change significantly (p = 0.66) [49].

**Model A without drug features.** Removing the drug features (propofol and remifentanil concentrations and the propofol TCI flag) reduced Model A's AUROC to 0.767, 0.747 and 0.732 (reduction +0.099 [0.091–0.107], +0.088 [0.079–0.095] and +0.066 [0.059–0.073]; all p < 0.001). This is the same size as the Model A vs Model B difference. The model without drug features was indistinguishable from Model B (ΔAUROC −0.004 [−0.007 to −0.001], p = 0.028; −0.002 [−0.005 to 0.000], p = 0.086; +0.001 [−0.002 to 0.003], p = 0.57), so preoperative hemoglobin, the only other extra feature, added nothing [63].

## Operating points

At the cross-fitted Youden threshold (Table 3) [48]:
- **Model A at 5 minutes:** sensitivity 0.747, specificity 0.825, PPV 0.070, NPV 0.995. It alarmed 11.1 times per hour of eligible monitoring and anticipated 82.8% of events, with a median lead time of 4.3 minutes.
- **Model B at 5 minutes:** anticipated a similar share of events (79.2%), but with 17.1 alarms per hour.
- **Across all horizons:** Model B generated 38–66% more alarms than Model A for a similar event-level sensitivity (for example, 21.7 vs 13.1 alarms per hour at 10 minutes). PPVs were low (0.04–0.13) and NPVs high (0.977–0.995), consistent with the low prevalence.

Decision curves are shown in Supplementary Figure S2.

## Subgroups

Discrimination was consistently lower in ASA III patients and in the youngest patients, for all three models and at every horizon (Supplementary Table S6) [49].
- **Model A by ASA class (I, II, III):** AUROC 0.873, 0.872 and 0.826 at 5 minutes, and 0.809, 0.802 and 0.763 at 15 minutes.
- **Model A by age (<40, 40–65, 65–80, ≥80 years):** 0.833, 0.870, 0.870 and 0.851 at 5 minutes, and 0.746, 0.801, 0.807 and 0.792 at 15 minutes.
- **Calibration in ASA III:** Model A overestimated risk slightly (observed/expected ratio 0.84–0.87; calibration slope 0.80–0.83).
- **Model B:** the pattern was similar, with lower AUROC also in patients aged 80 years or older (0.744, 0.708 and 0.696).
- **ASA IV–V (24 cases):** too small to interpret (95% CIs as wide as 0.52–0.85).

## What the models predict: the time course of the anesthetic

**Phase-stratified performance (Table 4)** [49]:
- Within phases, discrimination was substantially lower than pooled: Model A at 5 minutes had within-phase AUROCs of 0.737 (early), 0.676 (mid) and 0.759 (late), against a pooled AUROC of 0.866.
- Model A anticipated 91% of late events but only 36% of early and 44% of mid events.
- Within phases, Model B's AUROC ranged from 0.43 to 0.63 and the baseline's from 0.42 to 0.66 (Table 4).

**Reference models** [51, 52]:
- **Elapsed time alone** reached AUROCs of 0.669, 0.675 and 0.676.
- **Elapsed time plus drug features** (the anesthetic course) reached 0.842, 0.810 and 0.772.
- **Model A exceeded the anesthetic-course model** by +0.024 (0.019–0.029), +0.025 (0.019–0.030) and +0.027 (0.022–0.032) (all p < 0.001).
- **Model B exceeded elapsed time alone** by +0.101 (0.092–0.110), +0.075 (0.068–0.082) and +0.056 (0.050–0.062) (all p < 0.001).
- **Removing elapsed time** reduced AUROC by 0.017 (0.013–0.021), 0.031 (0.026–0.036) and 0.041 (0.035–0.048) for Model A, and by 0.047 (0.041–0.055), 0.065 (0.057–0.073) and 0.066 (0.058–0.075) for Model B (all p < 0.001), to 0.849, 0.804 and 0.757 (A) and 0.723, 0.684 and 0.666 (B) [63].

**Surgery period** [50]:
- Restricting windows and events to the surgical period left 3,217, 5,067 and 6,220 positive labelled windows (0.74%, 1.20% and 1.51%; before the completeness rule), 24–32% of the anesthesia-period count on the same basis (9,967, 19,444 and 26,157) [46, 50].
- Model A's AUROC was 0.798 (0.782–0.814), 0.733 (0.715–0.752) and 0.698 (0.677–0.720), and Model B's 0.714, 0.679 and 0.664.
- Model A still exceeded Model B (ΔAUROC +0.084, +0.054 and +0.034; all p < 0.001).
- Within surgery, 85% of events occurred in the late phase. Mid-phase events (11%) were poorly anticipated (AUROC 0.60; 48% caught).
- Within surgery, elapsed time alone reached 0.641, 0.636 and 0.630, and the anesthetic course 0.786, 0.730 and 0.681. Model A's increment over the anesthetic course was small (+0.011 [0.000–0.022], p = 0.046; +0.003 [−0.012 to 0.017], p = 0.66; +0.017 [0.001–0.036], p = 0.04). Model B's increment over elapsed time remained significant (+0.073 [0.057–0.089], +0.043 [0.027–0.059], +0.034 [0.017–0.050]; all p < 0.001) [52].

**Feature importance.** SHAP analysis was consistent with these results [54, 59]:
- In Model A, the drug features (propofol concentration ranked first at 5 minutes) and elapsed time (first at 10 and 15 minutes) dominated.
- In Model B, elapsed time ranked first at every horizon. It was followed by the MAP slope, ETCO2 slope and HR variance at 5 minutes, and by HR variance first at 10 and 15 minutes.
- In Model B, the contribution of the MAP slope decreased with horizon (mean |SHAP| 0.380, 0.203 and 0.123).
- Nonlinear HR features contributed little (mean |SHAP| ≤ 0.09).
- Rankings were consistent across horizons (Spearman ρ 0.76–0.96) and outer folds (mean ρ 0.92–0.98; all p < 0.0001).

## Label filtering and other sensitivity analyses

**BIS artifact filtering** [44, 45, 52]:
- **How much BIS ≥ 60 time each filter removed:**
  - EMG > 40 dB removed 4.1% of all valid BIS seconds but 41.1% of BIS ≥ 60 seconds;
  - EMG > 35 dB removed 49.6% of BIS ≥ 60 seconds;
  - SQI < 50 removed 2.5% and 5.6%, respectively.
- **Events detected (3,066 cases before case-level exclusion):** 4,545 events in 2,570 cases with SQI < 50, versus 1,806 in 1,099 cases (EMG > 40 dB) and 1,448 in 883 cases (EMG > 35 dB).
- **Effect on discrimination:** models trained and evaluated on EMG-filtered labels had AUROCs 0.061–0.111 lower than on SQI-filtered labels (all p < 0.001; Supplementary Table S5).
- **SQI filtering made no practical difference:** results with no filtering (|ΔAUROC| ≤ 0.002) or SQI < 15 (≤ 0.004; nominally significant only for Model B at 5 minutes, p = 0.008) were practically identical to the primary analysis. AUPRC was 0.004–0.010 lower with the primary labels than with unfiltered labels (significant in five of six comparisons), consistent with the slightly higher positive rate without filtering (1.95% vs 1.90% at 5 minutes) [46, 52]. SQI < 15 removed only 0.07% of valid BIS seconds and 0.2% of BIS ≥ 60 time [44].

**Window completeness.** Requiring ≥ 90% or ≥ 50% completeness instead of ≥ 70% changed AUROC by at most 0.002 (nominally significant only for Model B at 15 minutes) [52].

**Gap filling.** In a 200-case sample, 99.96% of gaps were 10 seconds or shorter, so the gap-filling limit could affect only about 0.02% of gaps [41].

**Matched negative sampling.** Against 5,304 negatives from 2,659 cases selected at least 20 minutes from any event, AUROCs were [53]:
- Model A: 0.891, 0.861 and 0.825;
- Model B: 0.815, 0.796 and 0.781;
- baseline: 0.645, 0.593 and 0.569.

Model A remained better than Model B (+0.076, +0.065 and +0.044; all p = 0.001). Even so, elapsed time remained the strongest separator between positive and negative windows (standardised mean difference ≈ +0.90).

**Repeat patients.** Removing the 98 cases of 43 patients with surgeries in different folds changed AUROC by +0.0005 to +0.003 [58].

**Effective sample size** [55]:
- Clustering within cases gave design effects of 4.7, 8.7 and 12.2 (intraclass correlation 0.016–0.048).
- Effective sample sizes were about 106,800, 57,300 and 40,400 windows and 1,840–1,980 positives, or 80–104 effective positives per candidate feature.
- Case-level bootstrap standard errors of AUROC were 2.0–3.2 times those from window-level resampling (square root of the empirical design effects 4.0–10.5).

## Computational cost

On a laptop processor (Intel Core i7-1280P), computing features and a calibrated prediction for one window took [56]:
- **All cores:** mean 139 ms (SD 32; 95th percentile 197 ms) for Model A and 137 ms (p95 193 ms) for Model B.
- **One thread on a single core:** 132 ms (p95 180 ms) and 130 ms (p95 178 ms).
- DFA and sample entropy accounted for about 90% of this time.

A prediction is due every 60 seconds.

---

## Tables and figures (to be built from the results files)

**Table 1.** Characteristics of the 2,673 included cases, overall and by occurrence of at least one event (`table1.md`) [57].

**Table 2.** Discrimination and calibration by model and horizon.

| Horizon | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE | Calibration intercept | Calibration slope |
|---|---|---|---|---|---|---|---|
| 5 min | A | 0.866 (0.858–0.874) | 0.176 (0.161–0.193) | 0.0153 (0.0148–0.0158) | 0.0010 | −0.064 | 1.02 |
| | B | 0.770 (0.761–0.780) | 0.088 (0.080–0.097) | 0.0163 (0.0158–0.0169) | 0.0006 | −0.036 | 1.02 |
| | Baseline | 0.641 (0.631–0.652) | 0.032 (0.030–0.034) | 0.0168 (0.0163–0.0174) | 0.0000 | 0.000 | 0.94 |
| 10 min | A | 0.835 (0.826–0.844) | 0.239 (0.220–0.260) | 0.0293 (0.0283–0.0302) | 0.0022 | −0.051 | 1.03 |
| | B | 0.750 (0.740–0.760) | 0.118 (0.109–0.127) | 0.0320 (0.0310–0.0330) | 0.0015 | −0.029 | 1.05 |
| | Baseline | 0.588 (0.578–0.597) | 0.048 (0.046–0.051) | 0.0333 (0.0322–0.0343) | 0.0000 | 0.001 | 0.92 |
| 15 min | A | 0.798 (0.790–0.807) | 0.231 (0.213–0.251) | 0.0406 (0.0393–0.0418) | 0.0016 | −0.030 | 1.01 |
| | B | 0.732 (0.722–0.742) | 0.129 (0.121–0.138) | 0.0436 (0.0422–0.0449) | 0.0009 | −0.017 | 1.02 |
| | Baseline | 0.561 (0.552–0.570) | 0.060 (0.057–0.062) | 0.0453 (0.0439–0.0467) | 0.0002 | 0.004 | 0.90 |

*Prevalence 1.72%, 3.46% and 4.77% at 5, 10 and 15 minutes; the Brier score of a model predicting only the prevalence would be 0.0169, 0.0334 and 0.0454. 95% CIs from 1,000 case-level bootstrap resamples (`sqi_primary_w70_diagnostic_metrics.csv`) [48].*

**Table 3.** Operating characteristics at the cross-fitted Youden threshold [48].

| Model, horizon | Sensitivity | Specificity | PPV | NPV | Alarms/h | Events anticipated | Median lead (min) |
|---|---|---|---|---|---|---|---|
| A, 5 min | 0.747 | 0.825 | 0.070 | 0.995 | 11.1 | 82.8% | 4.3 |
| A, 10 min | 0.713 | 0.799 | 0.113 | 0.987 | 13.1 | 84.0% | 9.2 |
| A, 15 min | 0.686 | 0.761 | 0.126 | 0.980 | 15.6 | 82.1% | 14.2 |
| B, 5 min | 0.675 | 0.723 | 0.041 | 0.992 | 17.1 | 79.2% | 4.2 |
| B, 10 min | 0.715 | 0.651 | 0.069 | 0.985 | 21.7 | 83.9% | 9.3 |
| B, 15 min | 0.690 | 0.656 | 0.091 | 0.977 | 21.6 | 80.5% | 14.3 |

*Lead time is capped by the horizon by construction.*

**Table 4.** The time course of the anesthetic: pooled vs within-phase discrimination, reference models and the surgery period (AUROC; 5 / 10 / 15 min) [49–52].

| Analysis | 5 min | 10 min | 15 min |
|---|---|---|---|
| Model A, pooled | 0.866 | 0.835 | 0.798 |
| Model A, within phase (early / mid / late) | 0.737 / 0.676 / 0.759 | 0.691 / 0.631 / 0.698 | 0.647 / 0.571 / 0.632 |
| Model B, within phase (early / mid / late) | 0.630 / 0.517 / 0.622 | 0.583 / 0.488 / 0.549 | 0.532 / 0.428 / 0.504 |
| Model A, events anticipated (early / mid / late) | 36% / 44% / 91% | 34% / 53% / 92% | 25% / 58% / 91% |
| Elapsed time only | 0.669 | 0.675 | 0.676 |
| Elapsed time + drug features | 0.842 | 0.810 | 0.772 |
| Model A without drug features | 0.767 | 0.747 | 0.732 |
| Model A without elapsed time | 0.849 | 0.804 | 0.757 |
| Model B without elapsed time | 0.723 | 0.684 | 0.666 |
| Model B | 0.770 | 0.750 | 0.732 |
| Surgery period: Model A / Model B | 0.798 / 0.714 | 0.733 / 0.679 | 0.698 / 0.664 |
| Surgery period: elapsed time only | 0.641 | 0.636 | 0.630 |
| Surgery period: elapsed time + drug features | 0.786 | 0.730 | 0.681 |

*Calibrated pooled AUROC from `run_comparisons.csv` [52, 63].*

**Figure 1.** Cohort flow diagram (`cohort_flow.png`, redrawn for the journal).
**Figure 2.** Reliability diagrams by model and horizon.
**Figure 3.** Timing of events within the anesthetic and within-phase performance *(suggested; built from `sqi_primary_w70_phase_stratified.csv`)*.
**Figure 4.** SHAP summary plots, Models A and B, 5 minutes.

**Supplement:** S1 deviations from protocol; S2 artifact bounds; S3 feature audit table; S4 retained vs excluded cases; S5 sensitivity analyses (Tests 3, 4, 6); S6 subgroups; S7 matched sampling; S8 repeat patients; S9 effective sample size; S10 latency; S11 surgery-period full results; decision curves.