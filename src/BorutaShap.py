from datetime import datetime
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from scipy.stats import binom_test, ks_2samp
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
import numpy as np
from numpy.random import choice
import seaborn as sns
import fasttreeshap

import warnings
warnings.filterwarnings("ignore")


class BorutaShap:
    """
    BorutaShap is a wrapper feature selection method built on the foundations of both the SHAP and Boruta algorithms.
    """
    def __init__(self, model=None, importance_measure='Shap', classification=True, percentile=100, pvalue=0.05):
        """
        Parameters
        ----------
        model: Model Object
            If no model specified then a base Random Forest will be returned otherwise the specified model will
            be returned.

        importance_measure: String
            Which importance measure too use either Shap or Gini/Gain

        classification: Boolean
            if true then the problem is either a binary or multiclass problem otherwise if false then it is regression

        percentile: Int
            An integer ranging from 0-100 it changes the value of the max shadow importance values. Thus, lowering its
            value would make the algorithm more lenient.

        pvalue: float
            A float used as a significance level again if the p-value is increased the algorithm will be more lenient
            making it smaller would make it more strict also by making the model more strict could impact runtime making
            it slower. As it will be less likely to reject and accept features.
        """
        self.importance_measure = importance_measure.lower()
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.model = model

        # All the features introduced later on:

        # Populated in fit
        self.starting_X = None
        self.X = None
        self.y = None
        self.sample_weight = None
        self.n_trials = None
        self.random_state = None
        self.ncols = None
        self.columns = None
        self.all_columns = None
        self.rejected_columns = []
        self.accepted_columns = []
        self.sample_pct = None
        self.normalize = None
        self.approximate = None
        self.feature_perturbation = None
        self.check_additivity = None
        self.features_to_remove = []
        self.hits = None
        self.order = None
        self.preds = None
        self.X_feature_import = None
        self.shadow_feature_import = None
        self.n_jobs = None

        # Populated in create_importance_history
        self.history_shadow = None
        self.history_x = None
        self.history_hits = None

        # Populated in create_shadow_features
        self.X_shadow = None
        self.X_boruta = None
        self.X_categorical = None

        # Populated in explain
        self.shap_values = None

        # Populated in calculate_rejected_accepted_tentative
        self.rejected = []
        self.accepted = []
        self.tentative = []

        self.check_model()
        # Record model type
        self.model_type = str(type(self.model)).lower()

    def fit(self, X, y, sample_weight=None, n_trials=20, random_state=0, sample_pct=None,
            check_additivity=False, normalize=True, verbose=True, approximate=False,
            feature_perturbation="tree_path_dependent", n_jobs=-1, fuckidy=None):
        """
        The main body of the program this method it computes the following

        1. Extend the information system by adding copies of all variables (the information system
        is always extended by at least 5 shadow attributes, even if the number of attributes in
        the original set is lower than 5).

        2. Shuffle the added attributes to remove their correlations with the response.

        3. Run a random forest classifier on the extended information system and gather the
        Z scores computed.

        4. Find the maximum Z score among shadow attributes (MZSA), and then assign a hit to
        every attribute that scored better than MZSA.

        5. For each attribute with undetermined importance perform a two-sided test of equality
        with the MZSA.

        6. Deem the attributes which have importance significantly lower than MZSA as ‘unimportant’
        and permanently remove them from the information system.

        7. Deem the attributes which have importance significantly higher than MZSA as ‘important’.

        8. Remove all shadow attributes.

        9. Repeat the procedure until the importance is assigned for all the attributes, or the
        algorithm has reached the previously set limit of the random forest runs.

        10. Stores results.

        Parameters
        ----------
        X: Dataframe
            A pandas dataframe of the features.

        y: Series/ndarray
            A pandas series or numpy ndarray of the target

        sample_weight: Series/ndarray
            A pandas series or numpy ndarray of the sample weight of the observations (optional)

        n_trials: int
            The number of times to run Boruta to determine feature importance

        random_state: int
            A random state for reproducibility of results

        sample_pct: Int
            An integer ranging from 0-100 that determines how much of the data is sampled while calculating SHAP values.

        check_additivity: Boolean
            Whether to check the additivity property during SHAP calculations

        normalize: boolean
            If true the importance values will be normalized using the z-score formula

        verbose: Boolean
            A flag indicator to print out all the rejected or accepted features.

        approximate: Boolean
            Whether to do exact of approximate SHAP calculations

        feature_perturbation: string
            Parameter to SHAP. The two options are 'tree_path_dependent' and 'interventional'. These refer to the
            methods of calculating probabilities in SHAP.'tree_path_dependent' is truer to data while 'interventional'
            is truer to model. 'tree_path_dependent' is the faster options. Thanks to the fasttreeshap package,
            we can now parallelize the job as well.

        n_jobs: int
            Number of cores to to run interventional fastreesshap on.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        np.random.seed(random_state)
        self.starting_X = X.copy()
        self.X = X.copy()
        self.y = y.copy()
        self.sample_weight = sample_weight.copy()
        self.n_trials = n_trials
        self.random_state = random_state
        self.ncols = self.X.shape[1]
        self.columns = self.X.columns.to_numpy()
        self.all_columns = self.X.columns.to_numpy()
        self.create_shadow_features()
        self.rejected_columns = []
        self.accepted_columns = []

        self.check_X()
        self.check_missing_values()
        self.sample_pct = sample_pct

        self.normalize = normalize
        self.approximate = approximate
        self.feature_perturbation = feature_perturbation
        self.check_additivity = check_additivity

        self.n_jobs = n_jobs

        self.features_to_remove = []
        self.hits = np.zeros(self.ncols)
        self.order = self.create_mapping_between_cols_and_indices()
        self.create_importance_history()

        if self.sample_pct is not None:
            self.preds = self.isolation_forest(self.X, self.sample_weight)

        self.train_model()

        for trial in trange(self.n_trials):
            num_features_dropped = self.remove_features_if_rejected()

            # Early stopping
            if self.X.shape[1] == 0:
                break

            # Only retrain if any columns have been dropped. Otherwise skip retraining since it is costly.
            if num_features_dropped:
                self.columns = self.X.columns.to_numpy()
                self.create_shadow_features()
                self.train_model()

            self.X_feature_import, self.shadow_feature_import = self.feature_importance()
            self.update_importance_history()
            hits = self.calculate_hits()
            self.hits += hits
            self.history_hits = np.vstack((self.history_hits, self.hits))
            self.test_features(iteration=trial+1)
            
            # Short circuit if there are no tentative columns left.
            self.calculate_rejected_accepted_tentative(verbose=False)
            if not self.tentative:
                print("No tentative columns left. Terminating.")
                break

            print(f"\n{datetime.now()} | Finished loop {trial+1} of {n_trials}.\n")

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative(verbose=verbose)

    def check_model(self):
        """
        Checks that a model object has been passed as a parameter when initializing the BorutaShap class.

        Returns
        -------
        Model Object
            If no model specified then a base Random Forest will be returned otherwise the specified model will
            be returned.

        Raises
        ------
        AttributeError
             If the model object does not have the required attributes.
        """
        check_fit = hasattr(self.model, 'fit')
        check_predict_proba = hasattr(self.model, 'predict')

        try:
            check_feature_importance = hasattr(self.model, 'feature_importances_')
        except:
            check_feature_importance = True

        if self.model is None:
            if self.classification:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestRegressor()

        elif check_fit is False and check_predict_proba is False:
            raise AttributeError('Model must contain both the fit() and predict() methods')

        elif check_feature_importance is False and self.importance_measure == 'gini':
            raise AttributeError('Model must contain the feature_importances_ method to use Gini try Shap instead')

        else:
            pass

    def check_X(self):
        """
        Checks that the data passed to the BorutaShap instance is a pandas Dataframe

        Returns
        -------
        Dataframe

        Raises
        ------
        AttributeError
             If the data is not of the expected type.
        """

        if isinstance(self.X, pd.DataFrame) is False:
            raise AttributeError('X must be a pandas Dataframe')

        else:
            pass

    def missing_values_y(self):
        """
        Checks for missing values in target variable.

        Returns
        -------
        Boolean

        Raises
        ------
        AttributeError
             If data is not in the expected format.

        """
        if isinstance(self.y, pd.Series):
            return self.y.isnull().values.any()

        elif isinstance(self.y, np.ndarray):
            return np.isnan(self.y).any()

        else:
            raise AttributeError('Y must be a pandas Dataframe or a numpy array')

    def check_missing_values(self):
        """
        Checks for missing values in the data.

        Returns
        -------
        Boolean

        Raises
        ------
        AttributeError
             If there are missing values present.

        """
        X_missing = self.X.isnull().any().any()
        y_missing = self.missing_values_y()

        models_to_check = ('xgb', 'catboost', 'lgbm', 'lightgbm')

        model_name = str(type(self.model)).lower()
        if X_missing or y_missing:
            if any([x in model_name for x in models_to_check]):
                print('Warning there are missing values in your data !')
            else:
                raise ValueError('There are missing values in your Data')
        else:
            pass

    def train_model(self):
        """
        Trains Model also checks to see if the model is an instance of catboost as it needs extra parameters
        also the try except is for models with a verbose statement

        Parameters
        ----------
        X: Dataframe
            A pandas dataframe of the features.

        y: Series/ndarray
            A pandas series or numpy ndarray of the target

        Returns
        ----------
        fitted model object
        """
        if 'catboost' in self.model_type:
            self.model.fit(self.X_boruta, self.y, sample_weight=self.sample_weight,
                           cat_features=self.X_categorical, verbose=False)
        else:
            try:
                self.model.fit(self.X_boruta, self.y, sample_weight=self.sample_weight, verbose=False)
            except:
                self.model.fit(self.X_boruta, self.y, sample_weight=self.sample_weight)

    def calculate_rejected_accepted_tentative(self, verbose):
        """
        Figures out which features have been either accepted rejected or tentative

        Returns
        -------
        3 lists

        """
        self.rejected = list(set(self.flatten_list(self.rejected_columns))-set(self.flatten_list(self.accepted_columns)))
        self.accepted = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.all_columns) - set(self.rejected + self.accepted))

        if verbose:
            print(str(len(self.accepted)) + ' attributes confirmed important: ' + str(self.accepted))
            print(str(len(self.rejected)) + ' attributes confirmed unimportant: ' + str(self.rejected))
            print(str(len(self.tentative)) + ' tentative attributes remains: ' + str(self.tentative))

    def create_importance_history(self):
        """
        Creates a dataframe object to store historical feature importance scores.
        """
        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)
        self.history_hits = np.zeros(self.ncols)

    def update_importance_history(self):
        """
        At each iteration update the dataframe object that stores the historical feature importance scores.
        """
        padded_history_shadow = np.full(self.ncols, np.NaN)
        padded_history_x = np.full(self.ncols, np.NaN)

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_history_shadow[map_index] = self.shadow_feature_import[index]
            padded_history_x[map_index] = self.X_feature_import[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))

    def store_feature_importance(self):
        """
        Reshapes the columns in the historical feature importance scores object also adds the mean, median, max, min
        shadow feature scores.

        Returns
        -------
        Dataframe
        """
        self.history_x = pd.DataFrame(
            data=self.history_x,
            columns=self.all_columns
        )

        self.history_x['Max_Shadow'] = [max(i) for i in self.history_shadow]
        self.history_x['Min_Shadow'] = [min(i) for i in self.history_shadow]
        self.history_x['Mean_Shadow'] = [np.nanmean(i) for i in self.history_shadow]
        self.history_x['Median_Shadow'] = [np.nanmedian(i) for i in self.history_shadow]

    def results_to_csv(self, filename='feature_importance'):
        """
        Saves the historical feature importance scores to csv.

        Parameters
        ----------
        filename : string
            Used as the name for the output file.

        Returns
        -------
        Comma delimited file
        """
        features = pd.DataFrame(
            data={
                'Features': self.history_x.iloc[1:].columns.values,
                'Average Feature Importance': self.history_x.iloc[1:].mean(axis=0).values,
                'Standard Deviation Importance': self.history_x.iloc[1:].std(axis=0).values
            }
        )

        decision_mapper = self.create_mapping_of_features_to_attribute(
            maps=['Tentative', 'Rejected', 'Accepted', 'Shadow']
        )
        features['Decision'] = features['Features'].map(decision_mapper)
        features = features.sort_values(by='Average Feature Importance',ascending=False)

        features.to_csv(filename + '.csv', index=False)

    def remove_features_if_rejected(self):
        """
        At each iteration if a feature has been rejected by the algorithm remove it from the process.
        """
        features_dropped = 0
        if len(self.features_to_remove) != 0:
            print(f"Dropping features: {self.features_to_remove}")
            try:
                self.X.drop(columns=list(self.features_to_remove), inplace=True)
                features_dropped = len(self.features_to_remove)
                self.features_to_remove = []
            except Exception as e:
                print(f"Failed to drop features {self.features_to_remove} because of exception: {e}")
        return features_dropped

    @staticmethod
    def flatten_list(array):
        return [item for sublist in array for item in sublist]

    def create_mapping_between_cols_and_indices(self):
        return dict(zip(self.X.columns.to_list(), np.arange(self.X.shape[1])))

    def calculate_hits(self):
        """
        If a features importance is greater than the maximum importance value of all the random shadow
        features then we assign it a hit.
        """
        shadow_threshold = np.percentile(self.shadow_feature_import,
                                         self.percentile)

        padded_hits = np.zeros(self.ncols)
        hits = self.X_feature_import > shadow_threshold

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_hits[map_index] += hits[index]

        return padded_hits

    def create_shadow_features(self):
        """
        Creates the random shadow features by shuffling the existing columns.

        Returns:
            Dataframe with random permutations of the original columns.
        """
        self.X_shadow = self.X.apply(np.random.permutation)

        # If the model is 'catboost', convert columns of type 'object' to type 'category' and make note of them.
        if 'catboost' in self.model_type and isinstance(self.X_shadow, pd.DataFrame):
            obj_cols = self.X_shadow.select_dtypes("object").columns.tolist()
            if not obj_cols:
                pass
            else:
                self.X_shadow[obj_cols] = self.X_shadow[obj_cols].astype("category")

        self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
        self.X_boruta = pd.concat([self.X, self.X_shadow], axis=1)

        # 'catboost' models need to know which columns are categorical. Note their names.
        if 'catboost' in self.model_type:
            col_types = self.X_boruta.dtypes
            self.X_categorical = list(col_types[(col_types == 'category') | (col_types == 'object')].index)

    @staticmethod
    def calculate_z_score(array):
        """
        Calculates the Z-score of an array

        Parameters:
        ----------
            array: array_like

        Returns:
        ----------
            normalised array
        """
        mean_value = np.mean(array)
        std_value = np.std(array)
        return [(element-mean_value)/std_value for element in array]

    def feature_importance(self):
        """
        Calculates the feature importance scores of the model

        Returns:
            array of normalized feature importance scores for both the shadow and original features.

        Raise
        ----------
            ValueError:
                If no Importance measure was specified
        """
        if self.importance_measure == 'shap':
            self.explain()
            vals = self.shap_values
            if self.normalize:
                vals = self.calculate_z_score(vals)
            x_feature_import = vals[:len(self.X.columns)]
            shadow_feature_import = vals[len(self.X.columns):]

        elif self.importance_measure == 'gini':
            feature_importances_ = np.abs(self.model.feature_importances_)
            if self.normalize:
                feature_importances_ = self.calculate_z_score(feature_importances_)
            x_feature_import = feature_importances_[:len(self.X.columns)]
            shadow_feature_import = feature_importances_[len(self.X.columns):]

        else:
            raise ValueError('No Importance_measure was specified select one of (shap, gini)')

        return x_feature_import, shadow_feature_import

    @staticmethod
    def isolation_forest(X, sample_weight):
        """
        Fits an isolation forest to the dataset and gives an anomaly score to every sample
        """
        clf = IsolationForest().fit(X, sample_weight=sample_weight)
        preds = clf.score_samples(X)
        return preds

    def get_split_size_list(self, length):
        """
        Provides indices to split dataframe into intervals specified by sample_pct
        """
        pct_step_size = round(self.sample_pct / 100 * length)
        return np.arange(pct_step_size, length, pct_step_size)

    def find_sample(self):
        """
        Finds a sample by comparing the distributions of the anomaly scores between the sample and the original
        distribution using the KS-test. Starts off at sample_pct and tries to find a significant sample a 100 times
        before going up to the next size increment by the same till
        a significant sample is found.
        """
        max_tries_per_size = 100
        p_value_threshold = 0.95

        size_steps = self.get_split_size_list(self.X.shape[0])
        for size in size_steps:
            for i in range(max_tries_per_size):
                sample_indices = choice(np.arange(self.preds.size), size=size, replace=False)
                sample = np.take(self.preds, sample_indices)
                ks_p_value = ks_2samp(self.preds, sample).pvalue
                if ks_p_value > p_value_threshold:
                    return self.X_boruta.iloc[sample_indices]
            print(f"Could not find a significant sample in {max_tries_per_size} attempts for size {size}. "
                  f"Increasing size.")

        print("Significant sample not found. Returning the entire dataset.")
        return self.X_boruta

    def explain(self):
        """
        The shap package has numerous variants of explainers which use different assumptions depending on the model
        type this function allows the user to choose explainer.

        Returns:
            shap values

        Raise
        ----------
            ValueError:
                if no model type has been specified tree as default
        """
        if self.feature_perturbation == "interventional":
            explainer = fasttreeshap.TreeExplainer(
                self.model,
                data=self.X_boruta,
                feature_perturbation=self.feature_perturbation,
                algorithm="v2",
                approximate=self.approximate,
                n_jobs=self.n_jobs
            )
        else:
            explainer = fasttreeshap.TreeExplainer(
                self.model,
                feature_perturbation=self.feature_perturbation,
                algorithm="v2",
                approximate=self.approximate,
                n_jobs=self.n_jobs
            )

        shap_values = explainer.shap_values(
            self.find_sample() if self.sample_pct is not None else self.X_boruta,
            approximate=self.approximate,
            check_additivity=self.check_additivity
        )

        if self.classification:
            # for some reason shap returns values wrapped in a list of length 1
            self.shap_values = np.array(shap_values)

            if isinstance(self.shap_values, list):
                class_inds = range(len(self.shap_values))
                shap_imp = np.zeros(self.shap_values[0].shape[1])
                for ind in class_inds:
                    shap_imp += np.abs(self.shap_values[ind]).mean(0)
                self.shap_values /= len(self.shap_values)

            elif len(self.shap_values.shape) == 3:
                self.shap_values = np.abs(self.shap_values).sum(axis=0)
                self.shap_values = self.shap_values.mean(0)

            else:
                self.shap_values = np.abs(self.shap_values).mean(0)

        else:
            self.shap_values = shap_values
            self.shap_values = np.abs(self.shap_values).mean(0)

    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        """
        Perform a test that the probability of success is p.
        This is an exact, two-sided test of the null hypothesis
        that the probability of success in a Bernoulli experiment is p
        """
        return [binom_test(x, n=n, p=p, alternative=alternative) for x in array]

    @staticmethod
    def symmetric_difference_between_two_arrays(array_one, array_two):
        set_one = set(array_one)
        set_two = set(array_two)
        return np.array(list(set_one.symmetric_difference(set_two)))

    @staticmethod
    def find_index_of_true_in_array(array):
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        used to counteract the problem of multiple comparisons.
        """
        pvals = np.array(pvals)

        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass

        alphac_bon = alpha / float(n_tests)
        reject = pvals <= alphac_bon
        pvals_corrected = pvals * float(n_tests)
        return reject, pvals_corrected

    def test_features(self, iteration):
        """
        For each feature with an undetermined importance perform a two-sided test of equality
        with the maximum shadow value to determine if it is statistically better

        Parameters
        ----------
        hits: an array which holds the history of the number times
              this feature was better than the maximum shadow

        Returns:
            Two arrays of the names of the accepted and rejected columns at that instance
        """

        acceptance_p_values = self.binomial_H0_test(
            self.hits,
            n=iteration,
            p=0.5,
            alternative='greater'
        )

        rejection_p_values = self.binomial_H0_test(
            self.hits,
            n=iteration,
            p=0.5,
            alternative='less'
        )

        # [1] as function returns a tuple
        modified_acceptance_p_values = self.bonferoni_corrections(
            acceptance_p_values,
            alpha=0.05,
            n_tests=len(self.columns)
        )[1]

        modified_rejection_p_values = self.bonferoni_corrections(
            rejection_p_values,
            alpha=0.05,
            n_tests=len(self.columns)
        )[1]

        # Take the inverse as we want true to keep features
        rejected_columns = np.array(modified_rejection_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]

        self.features_to_remove = rejected_features

        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)
        
        if len(rejected_features) > 0:
            print(f"Rejecting {len(rejected_features)} features: {rejected_features}")
        if len(accepted_features) > 0:
            print(f"Accepting {len(accepted_features)} features: {accepted_features}")

    def tentative_rough_fix(self):
        """
        Sometimes no matter how many iterations are run a feature may neither be rejected or
        accepted. This method is used in this case to make a decision on a tentative feature
        by comparing its median importance value with the median max shadow value.

        Returns:
            Two arrays of the names of the final decision of the accepted and rejected columns.

        """
        median_tentative_values = self.history_x[self.tentative].median(axis=0).values
        median_max_shadow = self.history_x['Max_Shadow'].median(axis=0)

        filtered = median_tentative_values > median_max_shadow

        self.tentative = np.array(self.tentative)
        newly_accepted = self.tentative[filtered]

        if len(newly_accepted) < 1:
            newly_rejected = self.tentative

        else:
            newly_rejected = self.symmetric_difference_between_two_arrays(newly_accepted, self.tentative)

        print(str(len(newly_accepted)) + ' tentative features are now accepted: ' + str(newly_accepted))
        print(str(len(newly_rejected)) + ' tentative features are now rejected: ' + str(newly_rejected))

        self.rejected = self.rejected + newly_rejected.tolist()
        self.accepted = self.accepted + newly_accepted.tolist()

    def subset(self, tentative=False):
        """
        Returns the subset of desired features
        """
        if tentative:
            return self.starting_X[self.accepted + self.tentative.tolist()]
        else:
            return self.starting_X[self.accepted]

    @staticmethod
    def create_list(array, color):
        colors = [color for x in range(len(array))]
        return colors

    @staticmethod
    def filter_data(data, column, value):
        data = data.copy()
        return data.loc[(data[column] == value) | (data[column] == 'Shadow')]

    @staticmethod
    def check_if_input_field_which_features_is_correct(my_string):
        my_string = str(my_string).lower()
        if my_string in ['tentative', 'rejected', 'accepted', 'all']:
            pass
        else:
            raise ValueError(
                my_string +
                " is not a valid value did you mean to type 'all', 'tentative', 'accepted' or 'rejected' ?"
            )

    def plot(self, X_rotation=90, X_size=8, figsize=(12, 8),
             y_scale='log', which_features='all', display=True):

        """
        creates a boxplot of the feature importances

        Parameters
        ----------
        X_rotation: int
            Controls the orientation angle of the tick labels on the X-axis

        X_size: int
            Controls the font size of the tick labels

        figsize: tuple of ints
            Defines the size of the plot

        y_scale: string
            Log transform of the y axis scale as hard to see the plot as it is normally dominated by two or three
            features.

        which_features: string
            Despite efforts if the number of columns is large the plot becomes cluttered so this parameter allows you to
            select subsets of the features like the accepted, rejected or tentative features default is all.

        display: Boolean
        controls if the output is displayed or not, set to false when running test scripts
        """
        # data from wide to long
        data = self.history_x.iloc[1:]
        data['index'] = data.index
        data = pd.melt(data, id_vars='index', var_name='Methods')

        decision_mapper = self.create_mapping_of_features_to_attribute(
            maps=['Tentative', 'Rejected', 'Accepted', 'Shadow']
        )
        data['Decision'] = data['Methods'].map(decision_mapper)
        data.drop(['index'], axis=1, inplace=True)

        options = {
            'accepted': self.filter_data(data, 'Decision', 'Accepted'),
            'tentative': self.filter_data(data, 'Decision', 'Tentative'),
            'rejected': self.filter_data(data, 'Decision', 'Rejected'),
            'all': data
        }

        self.check_if_input_field_which_features_is_correct(which_features)
        data = options[which_features.lower()]

        self.box_plot(data=data,
                      X_rotation=X_rotation,
                      X_size=X_size,
                      y_scale=y_scale,
                      figsize=figsize)
        if display:
            plt.show()
        else:
            plt.close()

    def box_plot(self, data, X_rotation, X_size, y_scale, figsize):
        if y_scale == 'log':
            minimum = data['value'].min()
            if minimum <= 0:
                data['value'] += abs(minimum) + 0.01

        order = data.groupby(by=["Methods"])["value"].mean().sort_values(ascending=False).index
        my_palette = self.create_mapping_of_features_to_attribute(
            maps=[
                'yellow',
                'red',
                'green',
                'blue'
            ]
        )

        # Use a color palette
        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            x=data["Methods"],
            y=data["value"],
            order=order,
            palette=my_palette
        )

        if y_scale == 'log':
            ax.set(yscale="log")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=X_rotation, size=X_size)
        ax.set_title('Feature Importance')
        ax.set_ylabel('Z-Score')
        ax.set_xlabel('Features')

    def create_mapping_of_features_to_attribute(self, maps=None):
        if maps is None:
            maps = []

        rejected = list(self.rejected)
        tentative = list(self.tentative)
        accepted = list(self.accepted)
        shadow = ['Max_Shadow', 'Median_Shadow', 'Min_Shadow', 'Mean_Shadow']

        tentative_map = self.create_list(tentative, maps[0])
        rejected_map = self.create_list(rejected, maps[1])
        accepted_map = self.create_list(accepted, maps[2])
        shadow_map = self.create_list(shadow, maps[3])

        values = tentative_map + rejected_map + accepted_map + shadow_map
        keys = tentative + rejected + accepted + shadow

        return self.to_dictionary(keys, values)

    @staticmethod
    def to_dictionary(list_one, list_two):
        return dict(zip(list_one, list_two))


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
