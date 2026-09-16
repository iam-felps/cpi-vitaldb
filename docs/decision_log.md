# Decision Log — CPI Project

## Locked design decisions (from protocol v2)

1. **Outcome scope:** broader inadequate-depth framing (sustained BIS≥60 anywhere in the case), with phase stratification (early/mid/late) as a pre-specified sensitivity analysis — not a narrowed late-case-only emergence window.

2. **Matched negative sampling:** included as a secondary design, but purely as an internal robustness/construct-validity check within the paper. Not framed as a head-to-head comparison against Qian et al.

3. **Nonlinear HR features (DFA, sample entropy):** included in both Model A and Model B.

## Phase 0 decisions

4. **Python version:** 3.13 selected (Windows environment). 3.14 ruled out — CatBoost has no working install path on 3.14 as of testing (confirmed open compatibility issue, no published wheels). 3.13 confirmed compatible with CatBoost (supported since release 1.2.8) and expected compatible with the rest of the stack (LightGBM, miceforest, neurokit2, nolds, vitaldb) based on pre-install research; actual `pip install` served as final confirmation — all libraries installed successfully.

5. **Random seed:** 2311, applied globally wherever randomness is used (splits, class-weighting in place of SMOTE, bootstrap resampling).

6. **imbalanced-learn:** installed per the original Phase 0 library list, but not used in the pipeline — SMOTE was replaced with class-weighting (decision documented separately: linear interpolation between nonlinear HR dynamics features generates physiologically impossible synthetic trajectories).