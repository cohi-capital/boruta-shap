import pandas as pd

from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 421
N_JOBS = 8

RFC_PARAMS = {
    'params_class_weight': 'balanced_subsample',
    'params_max_depth': 147,
    'params_max_features': 90,
    'params_max_samples': 0.6932948590457279,
    'params_min_samples_leaf': 26,
    'params_min_samples_split': 11,
    'params_n_estimators': 8,  # 0,
    'params_pos_threshold': 0.9434533270075801
}

df = pd.read_parquet('/Users/ali/Downloads/training_data_round_1.parquet')
print(len(df))
# df.reset_index(drop=True, inplace=True)
df = df.sample(axis=0, frac=0.1)
print(len(df))

X = df.drop(columns=['y_classification', 'ror_next_24h_max', 'time_close'])  # .sample(axis=1, frac=0.02)
y = df['y_classification']


rfc = RandomForestClassifier(
    n_estimators=RFC_PARAMS['params_n_estimators'],
    max_features=RFC_PARAMS['params_max_features'],
    max_depth=RFC_PARAMS['params_max_depth'],
    min_samples_leaf=RFC_PARAMS['params_min_samples_leaf'],
    min_samples_split=RFC_PARAMS['params_min_samples_split'],
    class_weight=RFC_PARAMS['params_class_weight'],
    max_samples=RFC_PARAMS['params_max_samples'],
    n_jobs=N_JOBS,
    verbose=1
)

feature_selector = BorutaShap(model=rfc, importance_measure='shap', classification=True, percentile=90)
feature_selector.fit(
    X=X,
    y=y,
    n_trials=100,
    random_state=RANDOM_STATE,
    feature_perturbation="tree_path_dependent",
    n_jobs=N_JOBS,
    sample_pct=2.5
)

feature_selector.results_to_csv('feature_importance_tree_path_dependent_percentile_90_pvalue_005')
