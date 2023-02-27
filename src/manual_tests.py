import pandas as pd

from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier


df = pd.read_parquet('/Users/ali/Documents/code/feature_candidates_first_filter.parquet')
df.reset_index(drop=True, inplace=True)

df2 = df.sample(axis=0, frac=0.03)
X = df2.drop(columns=['y_classification', 'ror_next_24h_max', 'time_close']).sample(axis=1, frac=0.02)
y = df2['y_classification']

rfc = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1)
Feature_Selector = BorutaShap(model=rfc, importance_measure='shap', classification=True)#, percentile=80, pvalue=0.1)

Feature_Selector.fit(X=X, y=y, n_trials=100, random_state=0, sample=True, train_or_test='train')
