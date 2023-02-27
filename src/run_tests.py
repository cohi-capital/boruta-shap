import numpy as np
import pandas as pd

from BorutaShap import BorutaShap
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def load_data(data_type='classification'):
    """
    Load Example datasets for the user to try out the package
    """
    data_type = data_type.lower()

    if data_type == 'classification':
        cancer = load_breast_cancer()
        X = pd.DataFrame(
            np.c_[cancer['data'], cancer['target']],
            columns=np.append(cancer['feature_names'], ['target'])
        )
        y = X.pop('target')

    elif data_type == 'regression':
        housing = fetch_california_housing()
        X = pd.DataFrame(
            np.c_[housing['data'], housing['target']],
            columns=np.append(housing['feature_names'], ['target'])
        )
        y = X.pop('target')

    else:
        raise ValueError("No data_type was specified, use either 'classification' or 'regression'")

    return X, y


def test_models(data_type, models):
    X, y = load_data(data_type=data_type)

    for key, value in models.items():
        print('Testing: ' + str(key))
        # no model selected default is Random Forest, if classification is False it is a Regression problem
        feature_selector = BorutaShap(model=value,
                                      importance_measure='shap',
                                      classification=True)

        feature_selector.fit(X=X, y=y, n_trials=5, random_state=0, train_or_test = 'train')

        # Returns Boxplot of features disaplay False or True to see the plots for automation False
        feature_selector.plot(X_size=12, figsize=(12,8),
                     y_scale='log', which_features='all', display=False)


if __name__ == "__main__":
    tree_classifiers = {'tree-classifier':DecisionTreeClassifier(), 'forest-classifier':RandomForestClassifier(),
                        'xgboost-classifier':XGBClassifier(),'lightgbm-classifier':LGBMClassifier(),
                        'catboost-classifier':CatBoostClassifier()}

    tree_regressors = {'tree-regressor':DecisionTreeRegressor(), 'forest-regressor':RandomForestRegressor(),
                       'xgboost-regressor':XGBRegressor(),'lightgbm-regressor':LGBMRegressor(),
                       'catboost-regressor':CatBoostRegressor()}

    test_models('regression', tree_regressors)
    test_models('classification', tree_classifiers)
