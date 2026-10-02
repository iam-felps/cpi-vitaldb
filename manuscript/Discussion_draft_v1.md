# Discussion (draft v1 — journal-neutral)

*Drafting notes are in italics and square brackets; remove before submission. [REF] marks a citation still to be inserted. Every number is taken from Results draft v1. Framing follows decision-log entry 60 and must be agreed with co-authors before this section is finalised. Length of the text below, excluding notes: about 2,000 words; most journals will need it shortened to about 1,200–1,500.*

## Principal findings

In 2,673 VitalDB cases, gradient-boosted models forecast sustained BIS ≥ 60 episodes 5–15 minutes ahead:
- **Model A (with drug information):** AUROC 0.866–0.798.
- **Model B (without drug information):** AUROC 0.770–0.732.
- **Calibration:** both models were well calibrated, with calibration slopes 1.01–1.05 and expected calibration error of at most 0.0022. In this cohort, the continuous index can therefore be read as a probability, although calibration was weaker in ASA III patients (slope 0.80–0.83).

The more important finding is what these models predict. Most of their discrimination reflects the time course of the anesthetic rather than anticipation of individual episodes:
- **Timing and drugs alone came close to the full model.** Elapsed time alone reached an AUROC of about 0.67, and elapsed time plus drug concentrations 0.84 at 5 minutes, within 0.024–0.027 of the full Model A.
- **Most events came late.** Almost three in five events (58%) began in the final fifth of the anesthetic.
- **Within each phase, discrimination fell well below the pooled figure.** For Model A at 5 minutes, within-phase AUROCs were 0.68–0.76 against a pooled 0.866.
- **Mid-case episodes were rare and poorly anticipated.** Within the surgical period, the 11% of events that began mid-case were poorly anticipated (AUROC 0.60).

## What the index anticipates, and what it does not

BIS ≥ 60 episodes in this cohort were concentrated at the end of the anesthetic:
- **Across the whole anesthetic:** 58% of events began in its final fifth, and only 4.4% in the middle phase.
- **Within surgery:** 85% of events still occurred in the late phase.

This pattern is consistent with deliberate reduction of anesthetic delivery during closure and emergence, when rising BIS is often the intended trajectory rather than a failure of depth. *[REF: emergence practice.]* A model that learns when in the anesthetic the risk is high, and how far drug concentrations have fallen, will therefore achieve high pooled discrimination without necessarily detecting unexpected lightening.

The reference models make this explicit:
- **Timing alone carries substantial signal.** Elapsed time alone discriminated with AUROC 0.67.
- **Drug concentrations carry most of the rest.** Adding them raised this to 0.84 at 5 minutes (0.81 and 0.77 at 10 and 15 minutes).
- **Everything else added little on top.** All of Model A's remaining features (vital signs, nonlinear HR features, age, BMI, ASA class and hemoglobin) added only 0.02–0.03, and within the surgical period at most 0.017.
- **SHAP agreed.** Drug concentrations and elapsed time dominated Model A's feature rankings.

The clinically most relevant target is an episode during the maintenance phase. Discrimination was lowest there: within-phase AUROC 0.676 for Model A at 5 minutes, and 0.60 within surgery, where about half of mid-phase events were anticipated. Early events were anticipated even less often (36%), largely because few were preceded by an eligible window. The small number of mid-phase events (173 of 3,930 within anesthesia) also limits how precisely this can be estimated.

The most defensible description of the CPI in its current form is therefore a calibrated estimate of the risk that BIS will rise, driven mainly by the anesthetic time course. It is not an early-warning signal for unexpected awareness.

This distinction matters beyond this study. Pooled AUROC in rolling-window designs mixes discrimination between patients and between phases of the same anesthetic, and here pooled AUROC exceeded within-phase AUROC by 0.11–0.23 for Model A and by up to 0.30 for Model B. Studies of intraoperative forecasting that report only pooled discrimination may overstate how well their models anticipate clinically unexpected events. Phase-stratified results, reference models based on timing alone, and analysis periods that exclude intended emergence should be reported routinely. *[Place this paragraph in the methodological section instead, depending on length.]*

## BIS labels and EMG filtering

The second message concerns label quality:
- **EMG filtering removed BIS elevations disproportionately.** Excluding BIS values recorded with EMG > 40 dB removed 4.1% of all BIS data but 41.1% of BIS ≥ 60 time; a 35 dB threshold removed 49.6%.
- **Discrimination was lower with EMG-filtered labels.** Models trained and evaluated on them discriminated worse by 0.06–0.11 AUROC.
- **SQI filtering had little effect.** SQI < 50 removed 5.6% of BIS ≥ 60 time. Compared with unfiltered labels, AUROC changed by at most 0.002 and AUPRC by at most 0.010.

Frontalis EMG activity rises with arousal in patients without deep neuromuscular blockade. *[REF: frontalis EMG and arousal.]* EMG-based filtering therefore likely deletes much of the physiological state the labels are meant to capture. The lower discrimination with EMG-filtered labels is consistent with less accurate labels rather than a weaker model.

There is no reference standard here, and the alternative explanation cannot be excluded: EMG contamination can itself raise BIS, so some of the removed time may have been artifact. Even so, a filter that removes 41–50% of the outcome while touching 4–6% of the data cannot be regarded as neutral.

Any study that labels BIS elevations after EMG-based filtering, such as an EMG > 35 dB cutoff during emergence [REF Kim et al. 2022], may therefore be working with a biased subset of events. *[Literature pass: check how often EMG filtering is used for BIS labels before claiming it is common.]* We recommend the manufacturer's signal quality index as the primary validity criterion and reporting how much outcome time any filter removes. SQI filtering is not fully neutral either: its share of BIS ≥ 60 time removed (5.60%) was 2.3 times its share of all BIS data removed (2.46%), a residual trade-off inherent to BIS-derived labels.

## Model A versus Model B

The primary comparison favoured Model A at every horizon (ΔAUROC +0.067 to +0.096), and the ablation analysis shows exactly why:
- **The drug features account for the whole difference.** Removing them reduced Model A's AUROC by the same amount.
- **Nothing else in Model A added anything.** The resulting model was practically indistinguishable from Model B (ΔAUROC −0.004 to +0.001), so preoperative hemoglobin contributed nothing.
- **The difference is therefore information about the anesthetic, not physiology.** The TCI-calculated effect-site concentrations (5-minute means) tell the model how much drug the patient is receiving at that moment; the models had no explicit trend feature for them. The propofol TCI indicator also encodes anesthetic technique (TIVA vs volatile anesthesia), so the advantage may partly reflect technique-specific patterns of emergence. Because the ablation removed both together, the two cannot be separated.

Qian et al. found no consistent benefit from adding age, sex, BMI, ASA class, comorbidities or anesthetic technique to HR-based features on this database [REF]. Our only Model A-specific static feature, preoperative hemoglobin, likewise added nothing. What carried the whole advantage of Model A was drug-delivery information: time-varying effect-site concentrations together with a real-time TCI indicator. Their static technique covariate may not capture this.

In AnesNet, a model without BIS channels, using only ABP, ECG, plethysmography and ETCO2, forecast excessive depth (BIS < 40) far less well than BIS-inclusive models [REF Qi et al. 2026]. Our results point the same way: information about the anesthetic state (BIS there, drug concentrations here) carries most of the performance, and routine hemodynamic channels alone reach only moderate discrimination. Their outcome is physiologically distinct from ours, so the comparison is qualitative.

## Feasibility of the pragmatic model

Model B addresses the setting that motivated this study: monitoring without BIS or TCI pumps. It is feasible in the technical sense:
- **It adds real information over the clock.** Its vital-sign and clinical features added 0.06–0.10 AUROC over elapsed time alone, and 0.03–0.07 within the surgical period.
- **It is well calibrated.**
- **It is cheap to run.** A prediction took under 0.2 seconds (95th percentile) on a single core of a standard laptop processor, far below the 60-second interval between predictions.
- **The dominant signals were plausible.** Apart from elapsed time, they were short-term MAP and ETCO2 slopes and HR variance, plausibly reflecting hemodynamic and ventilatory responses to lightening. SHAP rankings show importance, not direction. The MAP slope mattered most at the 5-minute horizon. *[REF: hemodynamic response to lightening.]*

But its absolute performance is moderate:
- **Discrimination:** AUROC 0.73–0.77 pooled, and 0.43–0.63 within phases.
- **Alarm burden:** at the Youden threshold, 17–22 alarms per hour with PPV of 0.04–0.09.

Two further limits apply to the low-resource argument:
- **The nonlinear HR features cost more than they return.** DFA and sample entropy accounted for about 90% of computation while contributing little to prediction, so a simpler model is likely to perform similarly at a fraction of the cost. This should be tested rather than assumed.
- **Blood pressure was measured continuously and invasively.** Model B relies on continuous invasive MAP, whereas in most low-resource settings blood pressure is measured non-invasively and intermittently.

We therefore regard Model B as a proof of technical feasibility, not as evidence that it would work in low-resource practice.

## Comparison with previous work

*[Expand with citations after the literature pass. Use only what the cited papers report.]*
- **Qian et al. [REF]:** on the same database, they forecast BIS > 60 from ECG-derived HR dynamics using matched negative sampling. Our matched-sampling analysis was designed as an internal robustness check, not as a replication, and gave higher AUROCs than our primary analysis (Model A 0.825–0.891). Even there, elapsed time remained the strongest separator of positive and negative windows. This suggests that designs sampling negatives far from events test an easier task, and that timing effects are not removed by matching.
- **Wang et al. (HypoBridCast) [REF] and Chaari et al. [REF]:** *[comparison of design, artifact handling and class-separation diagnostics.]*
- **Hypotension Prediction Index [REF]:** the concept of a continuous, probabilistic, forward-looking index is shared, but the signal and the outcome differ.

## Strengths and limitations

**Strengths:**
- A pre-specified protocol with all deviations reported (Supplementary Table S1).
- Nested case-level cross-validation with class weighting and nested calibration.
- Case-level resampling for all confidence intervals and tests, justified by design effects of about 5–12.
- Phase-stratified, surgery-period and reference-model analyses that test what the models predict.
- Verification steps that exposed and corrected several data errors before analysis: track misalignment, unreliable track indices, and look-ahead in the drug features.

**Limitations:**
- **Single centre, internal validation only.** The dataset comes from one high-resource academic centre, and the models have not been validated externally.
- **Selected population.** Requiring continuous invasive MAP and BIS selected longer, higher-risk operations (median surgical duration 160 vs 80 minutes in excluded eligible cases). Breast and colorectal surgery were underrepresented. The models apply to arterial-line-monitored patients only.
- **BIS is a surrogate.** It is an imperfect surrogate of consciousness, and a BIS ≥ 60 episode is not equivalent to awareness.
- **No negative control.** The planned vasopressor and fluid-bolus negative control could not be performed, because VitalDB records only total doses without administration times.
- **Induction not covered.** The first eligible prediction window fell a median of 15 minutes after anesthesia start, so induction was largely not covered.
- **Limited effective sample size.** Effective events per candidate predictor (80–104) exceed conventional regression thresholds but are below some recommendations for flexible machine-learning methods [REF van der Ploeg et al. 2014]. Stable performance and feature rankings across folds (Spearman 0.92–0.98) mitigate this concern.
- **Weaker subgroups.** Performance was lower in ASA III and in patients younger than 40, and the ASA IV–V subgroup was too small to assess.
- **Hardware.** Latency was measured on one laptop, not on low-end hardware.
- **Operating threshold.** The Youden operating point produces a high alarm burden (11–22 alarms per hour); clinically usable thresholds and alarm-persistence rules were not studied.

## Implications and future work

**For research:**
- Forecasting models of intraoperative events should report within-phase performance and timing-only reference models alongside pooled metrics.
- BIS-based labels should be filtered by signal quality rather than EMG, with the amount of outcome time removed reported.

**For the CPI:** the next steps are:
1. external validation in other centres;
2. a version using non-invasive, intermittent blood pressure;
3. a version restricted to the maintenance phase, where unexpected episodes matter most;
4. operating points chosen for acceptable alarm rates.

## Conclusions

A calibrated index forecast sustained BIS elevations 5–15 minutes ahead with good pooled discrimination. However, most of that discrimination came from the time course of the anesthetic and from drug concentrations, and mid-case episodes were poorly anticipated. EMG-based filtering of BIS labels removed a large share of BIS elevations and was associated with lower model performance. Without drug data, forecasting remained feasible in real time on a single laptop core, but with moderate accuracy. These results argue for reporting timing-adjusted performance in intraoperative forecasting and for external validation before any clinical use.