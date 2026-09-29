# CPI — predicting inadequate depth of anesthesia (BIS ≥ 60) from VitalDB

Machine-learning study forecasting sustained BIS ≥ 60 episodes 5, 10 and 15 minutes ahead from
routine intraoperative monitoring (Model A: with TCI drug concentrations; Model B: without),
with nested case-level cross-validation and calibrated probabilities.

- Protocol as executed: `docs/CPI_Protocol_v7.md`; implementation guide: `docs/CPI_Implementation_Guide_v4.md`
- Differences from the pre-specified protocol v6: `docs/CPI_Protocol_Deviations_v6_to_v7.md`
- Every analytical decision, with numbers: `docs/decision_log.md` (entries cited below)
- Shared constants (paths, seed 2311, horizons, feature lists): `src/cpi_config.py`

## Environment

Python 3.13 (Windows); key packages: numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, nolds 0.6.3, pyarrow, matplotlib, scipy, psutil,
vitaldb. Run every command from the project root:

```powershell
& ".venv\Scripts\python.exe" "src\<script>.py" [options]
```

## Run order

Steps marked *network* need internet access to VitalDB; everything else reads local files.
Case counts are those of the final analysis.

### Phase 2–3 — data and cohort (6,388 → 3,066 cases)

| Step | Script | Output |
|---|---|---|
| 1 | `phase3_cohort_v2.py` *network, first run only* | clinical snapshot `data/raw/vitaldb_clinical_snapshot.csv`; `results/phase3_step4_caseids.csv` (5,825) |
| 2 | `phase3_recheck_tracks.py` *network* | `results/phase3_track_recheck.csv` (per-case track list; resumable) |
| 3 | `phase3_cohort_v2.py` again | `results/phase3_final_cohort_caseids.csv` (3,066). Never overwrites an existing list. Case 3609 is excluded because its track list could not be downloaded (entry 57). |
| 4 | `phase2_download_tracks.py` *network* | raw tracks `data/raw/{caseid}.parquet` |
| 5 | `phase4_download_sqi.py` *network* | raw BIS SQI |

### Phase 4 — signal processing and cohort lock (3,066 → 2,673 cases)

| Step | Script | Output |
|---|---|---|
| 6 | `phase4_step1_reshape_plausibility.py` | 1-s grid per case, plausibility bounds (entry 14) → `data/interim/{caseid}.parquet` |
| 7 | `phase4_step3_locf.py` | LOCF ≤ 15 s on vital signs (entry 16) |
| 8 | `phase4_step2d_v2_aligned.py` | all BIS variants rebuilt from raw on the common grid (SQI ≥ 50 primary; SQI ≥ 15; EMG ≤ 40/35 dB; unfiltered), with alignment self-check |
| 9 | `phase4_step2e_event_detection_v3.py` | sustained BIS ≥ 60 events (≥ 180 s) per variant → `results/phase4_events_{variant}.csv` |
| 10 | `phase4_step4_case_missingness.py` | case-level missingness, 95th-percentile exclusion → `results/phase4_final_cohort_caseids.csv` (2,673) |

### Phase 5–6 — features and labels

| Step | Script | Output |
|---|---|---|
| 11 | `phase5_feature_engineering_v2.py` | `data/processed/phase5_features_v2.parquet` (5-min windows, 60-s stride; DFA with OLS fit and validity rule; elapsed time from anesthesia start; clinical variables). Reuses v1 sample-entropy values if `phase5_features_full.parquet` exists, otherwise computes them (slower). Entry 45 |
| 12 | `phase5b_apply_dfa_validity.py` | no-op on a fresh run (the rule is built into step 11); kept for the record |
| 13 | `phase5c_realtime_tci.py` | real-time drug features, no look-ahead (entry 47); idempotent |
| 14 | `phase6_labeling_v2.py` | `data/processed/phase6_labels_v2.parquet` (anesthesia period, primary) |
| 15 | `phase6_labeling_v2.py --period surgery` | `..._surgery.parquet` (sensitivity analysis, entry 49) |

### Phase 7 — models (nested CV, class weighting, 12 GBDT candidates, nested calibration)

Output per run: `results/phase7_v2/{tag}_{model}_{horizon}_predictions.parquet`, `_config.csv`,
and `{tag}_log.txt`, whose first line records the exact options used. The primary run creates
`outer_folds.csv`, which every other run reuses.

| Run (tag) | Command options |
|---|---|
| Primary `sqi_primary_w70` | *(none)* — Models A, B, baseline × 5/10/15 min |
| Test 3 `emg40_w70`, `emg35_w70`, `unfiltered_w70` | `--variant emg40` (etc.) `--models A B --fixed-config-from sqi_primary_w70` |
| Test 4 `sqi15_w70` | `--variant sqi15 --models A B --fixed-config-from sqi_primary_w70` |
| Test 6 `sqi_primary_w90`, `sqi_primary_w50` | `--min-window-frac 0.9` / `0.5` `--models A B --fixed-config-from sqi_primary_w70` |
| Surgery period `sqi_primary_w70_surgery` | `--period surgery --fixed-config-from sqi_primary_w70` |
| Ablation (drug features removed) | `--models A --drop-features tiva_propofol PPF20_CE_mean RFTN20_CE_mean --fixed-config-from sqi_primary_w70` |
| Reference: timing only | `--models A --only-features elapsed_time_s --fixed-config-from sqi_primary_w70` (and with `--period surgery`) |
| Reference: anesthetic course | `--models A --only-features elapsed_time_s PPF20_CE_mean RFTN20_CE_mean tiva_propofol --fixed-config-from sqi_primary_w70` (and with `--period surgery`) |
| Elapsed-time ablation | `--models A B --drop-features elapsed_time_s --fixed-config-from sqi_primary_w70` |

Check each run's log header for the exact options before re-running.

### Phase 9 — evaluation (all from saved predictions unless noted)

| Script | What it produces (`results/phase9_v2/`) | Entry |
|---|---|---|
| `phase9_diagnostic_metrics_v2.py --tag sqi_primary_w70` (and `_surgery`) | AUROC/AUPRC, sensitivity, specificity, PPV, NPV at cross-fitted Youden thresholds; event-level sensitivity; alarms/h; reliability; decision curves | |
| `phase9_tests_subgroups_v2.py --tag ...` | Hypothesis tests 1–2, subgroups, phase stratification (patient-level paired bootstrap) | |
| `phase9_compare_runs_v2.py` | Tests 3, 4, 6, reference-model and ablation comparisons (`--sets` runs only the named sets; other rows are kept) | 52, 63 |
| `phase9_matched_sampling_v2.py` | matched negative-sampling secondary analysis | 53 |
| `phase9_shap_v2.py` | out-of-fold TreeSHAP (refits each fold's selected model) | 54 |
| `phase9_shap_consistency_v2.py` | Spearman correlations with p-values and Jaccard indices from `shap_importance.csv` | |
| `phase9_effective_n_v2.py` | clustering-adjusted effective sample size | 55 |
| `phase9_latency_benchmark_v2.py --tier workstation` and `--tier single` | per-window latency (refits fold-0 models) | 56 |
| `phase9_table1_attrition_v2.py` | cohort flow diagram, Table 1, repeat-patient count | 57 |
| `phase9_repeat_patients_v2.py` | repeat-patient sensitivity analysis | 58 |
| `phase9_selection_bias_v2.py` | retained vs excluded cases (selection-bias check) | 61 |

## Diagnostics (read-only checks, cited in the decision log)

`check_track_alignment.py`, `check_filter_overlap.py`, `diag_dfa_mismatch.py`,
`phase2_sanity_check.py`, `phase2_bis_coverage_check.py`, `phase3_cohort_check.py`

## Archive

`src/archive/` holds superseded scripts (v1 pipeline, earlier Phase 4 BIS steps, earlier Phase
7–10 analyses). They are kept for traceability of earlier decision-log entries and are not part
of the pipeline; their relative paths may no longer resolve.