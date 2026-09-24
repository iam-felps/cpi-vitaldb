# Feature Audit Table — CPI Project (Phase 5, Section 8)

Every feature entered into Model A and/or Model B, with source track, window definition, and leakage confirmation. Feature set was fixed a priori before Phase 7 model training; no data-dependent pruning was applied at any point (per protocol, decision log entry re: Phase 5 step 4).

| Feature | Source track | Window (relative to t) | Model A | Model B | Real-time available? | Leakage check |
|---|---|---|---|---|---|---|
| HR_mean | Solar8000/HR | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window; confirmed no look-ahead |
| HR_slope | Solar8000/HR | trailing 5 min ending at t | ✅ | ✅ | Yes | Closed-form linear slope over trailing window only |
| HR_var | Solar8000/HR | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ART_MBP_mean | Solar8000/ART_MBP | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ART_MBP_slope | Solar8000/ART_MBP | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ART_MBP_var | Solar8000/ART_MBP | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| PLETH_SPO2_mean | Solar8000/PLETH_SPO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| PLETH_SPO2_slope | Solar8000/PLETH_SPO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| PLETH_SPO2_var | Solar8000/PLETH_SPO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ETCO2_mean | Solar8000/ETCO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ETCO2_slope | Solar8000/ETCO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| ETCO2_var | Solar8000/ETCO2 | trailing 5 min ending at t | ✅ | ✅ | Yes | Strictly trailing window |
| HR_dfa_alpha1 | Solar8000/HR (derived) | trailing 5 min ending at t | ✅ | ✅ | Yes | Nonlinear/fractal feature; box sizes 4–16s (Peng et al. 1995, adapted to 1Hz sampling — decision log entry 23); computed only on the trailing window, no look-ahead |
| HR_dfa_alpha2 | Solar8000/HR (derived) | trailing 5 min ending at t | ✅ | ✅ | Yes | Box sizes 16–64s; same trailing-window guarantee as alpha1 |
| HR_sampen | Solar8000/HR (derived) | trailing 5 min ending at t | ✅ | ✅ | Yes | m=2, r=0.2×window SD (Richman & Moorman 2000); per-window SD, no cross-window leakage |
| elapsed_time_s | Case metadata (casestart) | instantaneous at t | ✅ | ✅ | Yes | Only permitted temporal feature; NOT derived from case duration or time-to-end |
| age | Clinical metadata | static, known pre-case | ✅ | ✅ | Yes | Known before case starts; no leakage risk |
| bmi | Clinical metadata | static, known pre-case | ✅ | ✅ | Yes | Known before case starts |
| asa | Clinical metadata | static, known pre-case | ✅ | ✅ | Yes | Known before case starts |
| PPF20_CE_mean | Orchestra/PPF20_CE | trailing 5 min ending at t | ✅ | ❌ | Yes (TCI-derived; Model A only, per pragmatic/low-resource design) | Strictly trailing window |
| RFTN20_CE_mean | Orchestra/RFTN20_CE | trailing 5 min ending at t | ✅ | ❌ | Yes (Model A only) | Strictly trailing window |
| preop_hb | Clinical metadata (lab) | static, known pre-case | ✅ | ❌ | Yes (Model A only) | Known before case starts |

## Explicitly excluded (confirmed, not included in either model)

| Candidate feature | Reason excluded |
|---|---|
| Total case duration | Direct leakage risk — would encode time-to-case-end |
| Time-to-case-end (any form) | Direct leakage risk |
| preop_hct (hematocrit) | Not available in VitalDB clinical data (no `preop_hct` field exists) — decision log entry 24 |
| BIS, BIS/EMG, BIS/SQI values | Used only for target/label construction and artifact filtering (Phase 4); never used as a predictive feature in either model |
| Data-dependent pruned/dropped nonlinear features | None — all candidate nonlinear HR features (DFA α1, α2, sample entropy) were retained a priori in both models; no post hoc correlation-based or importance-based pruning was applied at any stage (protocol requirement) |

## Verification performed

- **No duration-derived feature**: confirmed by direct inspection of the fixed `MODEL_A_FEATURES` / `MODEL_B_FEATURES` lists used throughout Phases 5–9 (see `src/phase7_cv_pipeline_full.py` and subsequent scripts) — no feature computed from `caseend`, total case length, or any quantity requiring knowledge of when the case ends.
- **All rolling-window statistics use strictly trailing windows** ending at t (verified in the Phase 5 `fast_rolling_stats` / `nonlinear_hr_features` implementation — window slice is always `arr[end-WINDOW_S:end]`, never extending past `end`).
- **Feature set fixed a priori**: the same `MODEL_A_FEATURES` / `MODEL_B_FEATURES` lists were used unchanged from Phase 7 (initial training) through Phase 9 (all sensitivity analyses) — no feature was added, removed, or reselected based on any interim result.