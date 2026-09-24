"""
Single source of truth for constants shared across CPI scripts (guide Phase 0 step 6).
Import with:  from cpi_config import *
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INTERIM_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CLINICAL_SNAPSHOT = os.path.join(RAW_DIR, 'vitaldb_clinical_snapshot.csv')

RANDOM_SEED = 2311

# VitalDB anonymisation: casestart = 0 s, stored as 2100-01-01 00:00:00 UTC in raw 'dt'
VITALDB_EPOCH = 4102444800.0

WINDOW_S = 300
STRIDE_S = 60
HORIZONS_S = {'5min': 300, '10min': 600, '15min': 900}
LOCF_MAX_GAP_S = 15

CORE_TRACKS = ['HR', 'ART_MBP', 'PLETH_SPO2', 'ETCO2']
# Window completeness: features are computed for windows >= 50% present and the
# per-track completeness is stored; the primary analysis keeps windows where every
# core track is >= 70% present (protocol: <=30% missing). Test 6 uses 90/70/50.
FEATURE_MIN_WINDOW_FRAC = 0.5
PRIMARY_MIN_WINDOW_FRAC = 0.7
TEST6_MIN_WINDOW_FRACS = {'10pct_missing': 0.9, '30pct_missing': 0.7, '50pct_missing': 0.5}

HEMO_FEATURES = [f'{t}_{s}' for t in CORE_TRACKS for s in ('mean', 'slope', 'var')]
NONLINEAR_HR_FEATURES = ['HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen']
CLINICAL_FEATURES = ['age', 'bmi', 'asa']

MODEL_B_FEATURES = HEMO_FEATURES + NONLINEAR_HR_FEATURES + ['elapsed_time_s'] + CLINICAL_FEATURES
MODEL_A_FEATURES = MODEL_B_FEATURES + ['PPF20_CE_mean', 'RFTN20_CE_mean', 'tiva_propofol', 'preop_hb']
BASELINE_FEATURES = [f'{t}_{s}' for t in CORE_TRACKS for s in ('last', 'slope')]

# MICE (baseline/lab variables only, protocol 3.3); all other NaN handled natively by the GBDT
MICE_VARIABLES = ['age', 'bmi', 'asa', 'preop_hb']