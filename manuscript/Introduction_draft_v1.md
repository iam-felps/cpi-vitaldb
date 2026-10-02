# Introduction (draft v1 — journal-neutral)

*Drafting notes are in italics and square brackets; remove before submission. [REF] marks a citation still to be inserted. Framing follows decision-log entry 60 to maintain consistency with the Methods, Results, and Discussion.*

## Background and rationale

Maintaining an adequate depth of general anesthesia is fundamental to preventing intraoperative awareness, autonomic hyperreactivity, and unwanted patient movement, while avoiding excessive depth associated with delayed recovery, postoperative delirium, and adverse hemodynamic events [REF]. Processed electroencephalogram (EEG) indices, such as the Bispectral Index (BIS), are widely used to monitor hypnotic depth, with values between 40 and 60 generally considered the target range during surgical maintenance [REF].

Despite continuous physiological monitoring, transient or sustained rises in BIS above target thresholds ($\ge 60$) remain common during surgery. These episodes may reflect surgical stimulation exceeding anesthetic depth, physiological arousal, or deliberate reduction of drug delivery toward the end of a case [REF]. While real-time BIS displays reflect the immediate hypnotic state, anticipating impending rises in BIS several minutes in advance could provide anesthesiologists with a actionable window to titrate anesthetic delivery, administer analgesics, or prepare for patient emergence.

Recent advances in machine learning and physiological signal processing have prompted growing interest in real-time predictive modeling within the operating room [REF]. Several intraoperative forecasting tools—ranging from models predicting intraoperative hypotension to algorithms forecasting physiological depth—have demonstrated high discrimination using pooled receiver operating characteristic (ROC) curves [REF]. However, predicting hypnotic state transitions presents distinct clinical and methodological challenges:

1. **Temporal confounding:** Anesthetic delivery follows a structured, time-dependent trajectory from induction through maintenance to emergence. High pooled discrimination in rolling-window models may largely reflect the global time course of the procedure or scheduled drug changes rather than genuine anticipation of unexpected, clinically silent lightening events during maintenance [REF].
2. **Label contamination:** Machine-learning models rely on accurate ground-truth labels. In processed EEG signals like BIS, electromyographic (EMG) activity from frontalis muscle contraction can distort index values. While EMG activity rises during arousal, traditional artifact filters that discard data during elevated EMG activity may inadvertently delete the precise physiological events the models are intended to learn [REF].
3. **Data availability in pragmatic settings:** Sophisticated predictive models often depend on continuous, high-resolution data from target-controlled infusion (TCI) pumps or advanced physiological monitors [REF]. In resource-constrained settings or routine non-TCI environments, models must rely solely on non-invasive or standard vital signs, raising questions about their technical feasibility and diagnostic trade-offs.

## Objectives

To address these challenges, we evaluated the feasibility, predictive performance, and underlying signal drivers of machine-learning models designed to forecast sustained BIS elevations ($\ge 60$ for $\ge 3$ minutes) across short-to-medium horizons (5, 10, and 15 minutes ahead) using a large, high-resolution perioperative database (VitalDB). 

Specifically, this study aimed to:
1. Develop and calibrate two primary prediction models: a benchmark model incorporating TCI drug delivery concentrations alongside vital signs (**Model A**), and a pragmatic model using routine vital signs and patient demographics alone (**Model B**).
2. Quantify what these models actually anticipate by dissecting performance across distinct surgical phases (early, mid, and late/emergence) and comparing them against simple time-only and drug-only reference models.
3. Evaluate the impact of EMG-based signal filtering versus signal quality index (SQI) filtering on label fidelity and model performance.
4. Assess the computational feasibility and diagnostic utility of operating a pragmatic, non-TCI predictive model in real time.