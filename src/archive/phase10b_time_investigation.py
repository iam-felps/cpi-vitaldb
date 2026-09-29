"""
Phase 10b — Investigating elapsed_time_s's dominant SHAP importance.

Three checks, addressing the concern that the model may simply be detecting
"late in case = intentional emergence" rather than learning genuine
physiological signal:

1. Event-clustering histogram: where do actual positive events fall in
   relative case-time? Direct, unconfounded evidence of whether events
   cluster near case-end.
2. SHAP dependence plot for elapsed_time_s, all 6 model/horizon combinations
   (raw per-row SHAP values this time, not just the aggregate).
3. Ablation test: refit each combination WITHOUT elapsed_time_s (same
   5-fold outer CV, same saved Phase 7 hyperparameters otherwise), compare
   AUROC against the original (with elapsed_time_s) result. A small drop
   means other features are doing equivalent work; a large drop means the
   model leans heavily on elapsed time specifically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score
import ast
import os
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

RANDOM_SEED = 2311
N_OUTER_FOLDS = 5

MODEL_B_FEATURES = [
    'HR_mean', 'HR_slope', 'HR_var',
    'ART_MBP_mean', 'ART_MBP_slope', 'ART_MBP_var',
    'PLETH_SPO2_mean', 'PLETH_SPO2_slope', 'PLETH_SPO2_var',
    'ETCO2_mean', 'ETCO2_slope', 'ETCO2_var',
    'HR_dfa_alpha1', 'HR_dfa_alpha2', 'HR_sampen',
    'elapsed_time_s', 'age', 'bmi', 'asa',
]
MODEL_A_FEATURES = MODEL_B_FEATURES + ['PPF20_CE_mean', 'RFTN20_CE_mean', 'preop_hb']
MODELS = {'ModelA': MODEL_A_FEATURES, 'ModelB': MODEL_B_FEATURES}
HORIZONS = ['5min', '10min', '15min']


def mice_impute(train_df, test_df, features):
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
    train_arr = imputer.fit_transform(train_df[features])
    test_arr = imputer.transform(test_df[features])
    return pd.DataFrame(train_arr, columns=features), pd.DataFrame(test_arr, columns=features)


# ============ 1. Event-clustering histogram ============
def event_clustering_histogram():
    print("=== 1. Event-clustering histogram (relative onset time) ===")
    events = pd.read_csv(os.path.join(RESULTS_DIR, 'phase4_events_sqi_primary.csv'))
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase5_features_full.parquet'))
    case_max_t = features.groupby('caseid')['t'].max().to_dict()

    events['start'] = pd.to_timedelta(events['start']).dt.total_seconds()
    events['rel_onset'] = events.apply(
        lambda row: row['start'] / case_max_t.get(row['caseid'], np.nan) if case_max_t.get(row['caseid'], 0) else np.nan,
        axis=1
    )
    events = events.dropna(subset=['rel_onset'])
    events = events[(events['rel_onset'] >= 0) & (events['rel_onset'] <= 1)]

    print(f"Events with valid relative onset: {len(events)}")
    print(events['rel_onset'].describe())
    print(f"\nFraction of events with relative onset > 0.8 (late phase): {(events['rel_onset'] > 0.8).mean()*100:.1f}%")
    print(f"Fraction with relative onset < 0.4 (early phase): {(events['rel_onset'] < 0.4).mean()*100:.1f}%")
    print(f"Fraction in mid (0.4-0.8): {((events['rel_onset'] >= 0.4) & (events['rel_onset'] <= 0.8)).mean()*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(events['rel_onset'], bins=50, edgecolor='black')
    ax.set_xlabel('Relative onset time within case (0=start, 1=end)')
    ax.set_ylabel('Number of events')
    ax.set_title('Distribution of BIS>=60 Event Onset Times (relative to case duration)')
    plt.tight_layout()
    outpath = os.path.join(RESULTS_DIR, 'phase10b_event_clustering_histogram.png')
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")
    events[['caseid', 'rel_onset']].to_csv(os.path.join(RESULTS_DIR, 'phase10b_event_relative_onsets.csv'), index=False)


# ============ 2. SHAP dependence plots for elapsed_time_s, all 6 combos ============
def shap_dependence_plots(df, phase7_results):
    print("\n=== 2. SHAP dependence plots: elapsed_time_s, all 6 combinations ===")
    for model_name, features in MODELS.items():
        for horizon in HORIZONS:
            print(f"  {model_name}/{horizon}...")
            label_col = f'label_{horizon}'
            sub = df.dropna(subset=[label_col]).dropna(subset=features, how='all').reset_index(drop=True)

            params_row = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon) & (phase7_results['fold'] == 0)
            ]
            best_params = ast.literal_eval(params_row['best_params'].values[0])

            imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=10)
            X_imp = pd.DataFrame(imputer.fit_transform(sub[features]), columns=features)
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features)
            y = sub[label_col].astype(int).values

            model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
            model.fit(X_scaled, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            elapsed_idx = features.index('elapsed_time_s')
            elapsed_raw = sub['elapsed_time_s'].values  # use raw (unscaled) seconds for x-axis readability
            elapsed_shap = shap_values[:, elapsed_idx]

            # Downsample for plot clarity given large N
            rng = np.random.RandomState(RANDOM_SEED)
            n_plot = min(20000, len(elapsed_raw))
            idx = rng.choice(len(elapsed_raw), size=n_plot, replace=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(elapsed_raw[idx] / 60, elapsed_shap[idx], s=2, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_xlabel('Elapsed time since induction (minutes)')
            ax.set_ylabel('SHAP value (impact on model output)')
            ax.set_title(f'SHAP Dependence: elapsed_time_s — {model_name}, {horizon} horizon')
            plt.tight_layout()
            outpath = os.path.join(RESULTS_DIR, f'phase10b_shap_dependence_{model_name}_{horizon}.png')
            plt.savefig(outpath)
            plt.close(fig)
            print(f"    Saved: {outpath}")


# ============ 3. Ablation test: without elapsed_time_s ============
def ablation_test(df, phase7_results):
    print("\n=== 3. Ablation test: refit WITHOUT elapsed_time_s ===")
    results = []
    for model_name, features in MODELS.items():
        ablated_features = [f for f in features if f != 'elapsed_time_s']
        for horizon in HORIZONS:
            label_col = f'label_{horizon}'
            sub = df.dropna(subset=[label_col]).dropna(subset=ablated_features, how='all').reset_index(drop=True)
            X = sub[ablated_features].copy()
            y = sub[label_col].astype(int).values
            groups = sub['caseid'].values

            fold_params = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon)
            ].sort_values('fold')['best_params'].apply(ast.literal_eval).tolist()

            outer_cv = GroupKFold(n_splits=N_OUTER_FOLDS)
            fold_aurocs = []
            fold_start_all = time.time()

            for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
                best_params = fold_params[fold_i]
                X_train_raw = X.iloc[train_idx].reset_index(drop=True)
                X_test_raw = X.iloc[test_idx].reset_index(drop=True)
                y_train, y_test = y[train_idx], y[test_idx]

                X_train_imp, X_test_imp = mice_impute(X_train_raw, X_test_raw, ablated_features)
                scaler = RobustScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=ablated_features)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=ablated_features)

                sm = SMOTE(random_state=RANDOM_SEED)
                X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
                model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1)
                model.fit(X_train_res, y_train_res)
                preds = model.predict_proba(X_test_scaled)[:, 1]
                fold_aurocs.append(roc_auc_score(y_test, preds))

            mean_auroc_ablated = np.mean(fold_aurocs)

            # Compare against original (with elapsed_time_s) result from Phase 7
            original_auroc = phase7_results[
                (phase7_results['model'] == model_name) & (phase7_results['horizon'] == horizon)
            ]['auroc'].mean()

            drop = original_auroc - mean_auroc_ablated
            print(f"  {model_name}/{horizon}: AUROC with elapsed_time={original_auroc:.4f}, "
                  f"without={mean_auroc_ablated:.4f}, drop={drop:.4f} (time={time.time()-fold_start_all:.0f}s)")
            results.append({'model': model_name, 'horizon': horizon,
                             'auroc_with_elapsed_time': original_auroc,
                             'auroc_without_elapsed_time': mean_auroc_ablated,
                             'auroc_drop': drop})

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'phase10b_ablation_elapsed_time.csv'), index=False)


if __name__ == '__main__':
    total_start = time.time()
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'phase6_labeled_features.parquet'))
    phase7_results = pd.read_csv(os.path.join(RESULTS_DIR, 'phase7_all_results.csv'))

    event_clustering_histogram()
    shap_dependence_plots(df, phase7_results)
    ablation_test(df, phase7_results)

    print(f"\n\nTotal time: {(time.time()-total_start)/60:.1f} min")