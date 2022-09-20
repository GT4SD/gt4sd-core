# -*- coding:utf-8 -*-
"""
Prediction.py

Package for IBM Molecule Generation Experience

MIT License

Copyright (c) 2022 International Business Machines Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .FeatureExtraction import FeatureSet, MergedFeatureSet
from .Utility import update_data_mask, get_subclasses

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from collections import Counter
from enum import IntEnum
import copy

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def print_regression_model():
    """Print all the available regression models.
    """
    classes = get_subclasses(RegressionModel)
    print('Available regression models:')
    for index, cls in enumerate(classes):
        print('{0}: {1}'.format(index, cls.__name__))


def get_regression_model():
    """Get all the available regression models.

    Returns:
        dict: a mapping of class name and class object
    """
    class_map = dict()
    classes = get_subclasses(RegressionModel)
    for cls in classes:
        class_map[cls.__name__] = cls
    return class_map


# -----------------------------------------------------------------------------
# ModelSnapShot: a snapshot of a regression model stored in moldata
# -----------------------------------------------------------------------------


class ModelSnapShot(object):
    """Snapshot of regression model after fitting to molecule data. In solving the reverse problem,
    estimated feature values and generated candidate molecules are stored associated with the snapshot.

    Attributes:
        id (str): id of snapshot of a regression model
        model (RegressionModel): a regression model object
        estimator (object): a copy of an estimator object
    """

    def __init__(self, model):
        """Constructor of ModelSnapShot.

        Args:
            model (RegressionModel): a regression model object
        """
        self.id = model.get_id()
        self.model = copy.copy(model)
        self.estimator = copy.deepcopy(model.estimator)
        self.scaler = copy.deepcopy(model.scaler)

    def get_id(self):
        """Get id of a snapshot.

        Returns:
            str: id of a snapshot
        """
        return self.id

    def get_model(self):
        """Get a regression model restoring parameters when stored in a snapshot.

        Returns:
            RegressionModel: restored regression model
        """
        self.model.estimator = self.estimator
        self.model.scaler = self.scaler
        return self.model


# -----------------------------------------------------------------------------
# RegressionModel: a regression model for feature vectors of moldata
# -----------------------------------------------------------------------------


class RegressionModel(object):
    """Base class of regression model.

    Attributes:
        estimator (object): an estimator object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        score (float): R^2 score of fitting
        cv_score (tuple): R^2 score of cross validation (mean, std)
        mse (float): mean square error of fitting
        selection_mask (list): a list of masks for selected features
        selection_threshold (float): a threshold of feature selection
        status (Status): a status of regression model
    """

    class Status(IntEnum):
        INI = 0
        FIT = 1
        OPT = 2
        SEL = 3
        OPTSEL = 4

        @staticmethod
        def to_string(status, threshold):
            if status == RegressionModel.Status.INI:
                return 'ini'
            elif status == RegressionModel.Status.FIT:
                return 'fit'
            elif status == RegressionModel.Status.OPT:
                return 'opt'
            elif status == RegressionModel.Status.SEL:
                if threshold is None:
                    return 'select()'
                else:
                    return 'select({0})'.format(threshold)
            elif status == RegressionModel.Status.OPTSEL:
                if threshold is None:
                    return 'opt:select()'.format(threshold)
                else:
                    return 'opt:select({0})'.format(threshold)
            else:
                return ''

    def __init__(self, estimator, moldata, target_property, features, scaler):
        """Constructor of RegressionModel.

        Args:
            estimator (object): an estimator object
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            scaler (object): scaling for standardizing data
        """
        # check feature set
        if not (isinstance(features, MergedFeatureSet)):
            raise TypeError('features should be MergedFeatureSet')
        self.estimator = estimator
        self.scaler = scaler
        self.moldata = moldata
        self.target_property = target_property
        self.features = features
        self.params = dict()
        self.prediction_std = 0
        self.score = 0
        self.cv_score = (0, 0)
        self.mse = 0
        self.selection_mask = None
        self.selection_threshold = 0
        self.status = self.Status.INI
        # check target property value and molecules
        target_df = self.moldata.get_property_vector()
        target_df[self.target_property] = target_df[self.target_property].astype(float)
        target = target_df[self.target_property].values
        self.target_mask = list(map(lambda x: not np.isnan(x), target))
        self.target_mask = update_data_mask(self.target_mask, moldata.get_mols_mask())

    def get_id_base(self):
        """Get base id string. Actual id string is obtained by adding parameters to the base id string.

        Returns:
            str: base id string
        """
        return self.estimator.__class__.__name__

    def get_id(self):
        """Get id string.

        Returns:
            str: id string
        """
        id = self.get_id_base()
        if len(self.params) > 0:
            id += ':'
            for key in sorted(self.params.keys()):
                id += '{0}={1} '.format(key, self.params[key])
        return id.rstrip()+':'+self.Status.to_string(self.status, self.selection_threshold)

    def is_linear_model(self):
        """Get if the model is linear regression

        Returns:
            bool: true if linear regression model
        """
        return False

    def get_moldata(self):
        """Get MolData of a regression model

        Returns:
            MolData: moldata object
        """
        return self.moldata

    def get_target_property(self):
        """Get target property name.

        Returns:
            str: target property name
        """
        return self.target_property

    def get_vector_size(self):
        """Get the size of a feature vector.

        Returns:
            int: the size
        """
        if self.selection_mask is not None:
            return sum(self.selection_mask)
        else:
            return self.features.get_vector_size()

    def get_features(self):
        """Get a feature set for the regression.

        Returns:
            FeatureSet: a feature set
        """
        return self.features

    def get_feature_list(self):
        """Get a list of feature of a feature set.

        Returns:
            list: a list of features
        """
        return self.features.get_feature_list()

    def is_feature_selected(self):
        """Check if features are already selected.

        Returns:
            bool: true if features are selected
        """
        return self.status >= self.Status.SEL

    def get_prediction_std(self):
        """Get a standard deviation of the prediction.

        Returns:
            float: a standard deviation
        """
        return self.prediction_std

    def set_prediction_std(self, target, estimate):
        """Compute and set a standard deviation of the prediction.

        Args:
            target (array): array of target values of the regression
            estimate (array): array of estimated values of the regression
        """
        self.prediction_std = np.std(estimate - target)

    def get_score(self):
        """Get R^2 score of fitting.

        Returns:
            float: score
        """
        return self.score

    def get_cv_score(self):
        """Get R^2 score by cross validation.

        Returns:
            array: array of cv scores
        """
        return self.cv_score

    def get_cv_score_mean(self):
        """Get mean of R^2 score by cross validation.

        Returns:
            float: mean of cv score
        """
        return np.mean(self.cv_score)

    def get_cv_score_std(self):
        """Get std of R^2 score by cross validation.

        Returns:
            tuple: (mean, std)
        """
        return np.std(self.cv_score)

    def set_mse(self, target, estimate):
        """Set Mean Square Error of prediction

        Args:
            target (array): array of target values of the regression
            estimate (array): array of estimated values of the regression
        """
        self.mse = mean_squared_error(target, estimate)

    def get_mse(self):
        """Get Mean Square Error of prediction

        Returns:
            float: mse
        """
        return self.mse

    def get_rmse(self):
        """Get Root Mean Square Error of precision

        Returns:
            float: rmse
        """
        return np.sqrt(self.mse)

    def get_selection_mask(self):
        """Get a list of masks for selected features.

        Returns:
            list: a list of masks for selected features
        """
        if self.status < self.Status.SEL:
            logger.warning('feature selection is not yet done')
            return None
        return self.selection_mask

    def get_selected_feature_headers(self):
        """Get a list of selected feature header

        Returns:
            list: a list of feature headers
        """
        if self.status < self.Status.SEL:
            logger.warning('feature selection is not yet done')
            return None
        fid_list = self.features.get_header_list()
        if self.selection_mask is None:
            return fid_list
        else:
            return [fid for fid, sel in zip(fid_list, self.selection_mask) if sel]

    def get_params(self):
        """Get parameters of an estimator.

        Returns:
            dict: parameters of an estimator
        """
        return self.params

    def get_data(self, dataframe=None):
        """Get feature vectors to apply regression.

        Args:
            dataframe (DataFrame, optional): dataframe of feature vector. Defaults to None.

        Returns:
            array: a matrix of molecules and feature vectors
        """
        if dataframe is None:
            dataframe = self.moldata.get_feature_vector(self.features.id).astype(float)
            return dataframe.values[self.target_mask]
        else:
            return dataframe.values

    def get_selected_data(self, dataframe=None):
        """Get feature vector of selected features to apply regression.

        Args:
            dataframe (DataFrame, optional): dataframe of feature vector. Defaults to None.

        Returns:
            array: a matrix of molecules and selected feature vectors
        """
        if self.status < self.Status.SEL:
            logger.error('feature selection is not yet done')
            return None
        if dataframe is None:
            dataframe = self.moldata.get_feature_vector(self.features.id)
            return dataframe.values[self.target_mask][:, self.selection_mask]
        else:
            return dataframe.values[:, self.selection_mask]

    def get_target(self):
        """Get target values of regression.

        Returns:
            array: vector of target values
        """
        target_df = self.moldata.get_property_vector()
        target_df[self.target_property] = target_df[self.target_property].astype(float)
        target = target_df[self.target_property].values
        return target[self.target_mask]

    def register_model(self):
        """Register current snapshot of the regression model to moldata
        """
        self.moldata.add_regression_model(self)

    def predict_single_val(self, vector):
        """Get an estimate for a single feature vector.

        Args:
            vector: a feature vector

        Returns:
            float: an estimate
        """
        raise NotImplementedError('RegressionModel:predict_single_val()')

    def predict_val(self, data):
        """Get estimates for a matrix of molecules and a feature vector.

        Args:
            data: a matrix of molecules and a feature vector

        Returns:
            array: a vector of estimates
        """
        raise NotImplementedError('RegressionModel:predict_val()')

    def predict(self, dataframe=None):
        """Get estimates for molecule data in moldata.

        Args:
            dataframe (DataFrame, optional): dataframe of a feature vector. Defaults to None.

        Returns:
            DataFrame: a dataframe of estimates
        """
        raise NotImplementedError('RegressionModel:predict()')

    def fit(self, data=None):
        """Fit the model to feature vectors, and get estimates. If data is not provided, data from
        moldata is used. standard deviation of the estimates is set.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model (default None)

        Returns:
            array: a vector of estimates
         """
        raise NotImplementedError('RegressionModel:fit()')

    def cross_validation(self, data=None):
        """Fit the model to feature vectors, and estimate the accuracy by cross validation. If data is
        not provided, data from moldata is used. standard deviation of the estimate is set.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model (default None)

        Returns:
            array: array of scores of cross validation
        """
        raise NotImplementedError('RegressionModel:cross_validation()')

    def param_optimization(self, data=None):
        """Optimize hyperparameters of the estimator by grid search (GridSearchCV).
        If data is not provided, data from moldata is used. Accuracy of the obtained estimator
        is evaluated by cross validation.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model (default None)

        Returns:
            GridSearchCV: grid search object
         """
        raise NotImplementedError('RegressionModel:param_optimization()')

    def feature_selection(self):
        """Select important features for the prediction based on LASSO penalty (SelectFromModel).
        After the feature selection, the accuracy of the estimator is evaluated by cross validation.

        Returns:
            list: a list of selected features
        """
        raise NotImplementedError('RegressionModel:param_optimization()')


class SklearnRegressionModel(RegressionModel):
    """Base class of regression models from SciKitLearn python library.
    """

    hyper_params = None
    """str: dictionary of hyper parameters"""

    def __init__(self, estimator, moldata, target_property, features, scaler=True):
        """Constructor of SklearnRegressionModel.

        Args:
            estimator (object): a SciKitLearn estimator object
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            scaler (bool): flag of scaling for standardizing data
        """
        super().__init__(estimator, moldata, target_property, features,
                         StandardScaler() if scaler else None)

    def get_id(self):
        """Get id string.

        Returns:
            str: id string
        """
        id = self.get_id_base()
        params = self.get_params()
        if len(params) > 0:
            id += ':'
            for key in sorted(params.keys()):
                id += '{0}={1} '.format(key, params[key])
        return id.rstrip()+':'+self.Status.to_string(self.status, self.selection_threshold)

    def get_params(self):
        """Get parameters (defined in hyper_params) from an estimator object.

        Returns:
            dict: parameters of an estimator
        """
        if self.hyper_params is None:
            return self.estimator.get_params()
        else:
            params = self.estimator.get_params()
            for p in self.hyper_params:
                self.params[p] = params[p]
            return self.params

    def set_params(self, **kwargs):
        """Set parameters to an estimator object.

        Args:
            kwargs: key word args for an estimator
        """
        self.estimator.set_params(**kwargs)

    def get_coef(self):
        """Get coefficient of features by the regression.

        Returns:
            array: coefficient of features
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        if self.scaler is None:
            return self.estimator.coef_
        elif self.scaler.scale_ is None:
            return self.estimator.coef_
        else:
            return self.estimator.coef_ / self.scaler.scale_

    def get_shift(self):
        """Get amount of shifts by scaler

        Returns:
            array: amount of shifts
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        shift = 0
        if self.estimator.fit_intercept:
            shift = - self.estimator.intercept_
        if self.scaler is None:
            return shift
        elif self.scaler.mean_ is None:
            return shift
        else:
            if self.scaler.scale_ is None:
                return shift + np.inner(self.scaler.mean_, self.estimator.coef_)
            else:
                return shift + np.inner(self.scaler.mean_, self.estimator.coef_ / self.scaler.scale_)

    def predict_single_val(self, vector):
        """Get an estimate for a single feature vector.

        Note:
            a feature vector should include only selected features.

        Args:
            vector: a feature vector

        Returns:
            float: an estimate
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        data = np.array(vector).reshape(1, len(vector))
        if self.scaler is not None:
            scale_data = self.scaler.transform(data.astype(float))
        else:
            scale_data = data
        estimate = self.estimator.predict(scale_data)
        return estimate[0]

    def predict_val(self, data):
        """Get estimates for a matrix of molecules and a feature vector.

        Note:
            a feature vector should include only selected features.

        Args:
            data (array): a matrix of molecules and a feature vector

        Returns:
            array: a vector of estimates
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        if self.scaler is not None:
            scale_data = self.scaler.transform(data.astype(float))
        else:
            scale_data = data
        estimate = self.estimator.predict(scale_data)
        return estimate

    def get_r2_score(self, data, target):
        """Get R^2 score of the estimation

        Args:
            data (array): a matrix of molecules and a feature vector
            target (array): a vector of molecules and a target property value

        Returns:
            float: R^2 score
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        if self.scaler is not None:
            scale_data = self.scaler.transform(data.astype(float))
        else:
            scale_data = data
        score = self.estimator.score(scale_data, target)
        return score

    def predict(self, dataframe=None):
        """Get estimates for molecule data in moldata.

        Note:
            If dataframe of a feature vector is given, selected features are selected from the dataframe.

        Args:
            dataframe (DataFrame): dataframe of a feature vector. Defaults to None.

        Returns:
            DataFrame: a dataframe of estimates
        """
        if self.status < self.Status.FIT:
            logger.error('model fitting is not yet done')
            return None
        if dataframe is None:
            dataframe = self.moldata.get_feature_vector(self.features.id)
        if self.is_feature_selected():
            data = self.get_selected_data(dataframe)
        else:
            data = self.get_data(dataframe)
        if self.scaler is not None:
            scale_data = self.scaler.transform(data.astype(float))
        else:
            scale_data = data
        estimate = self.estimator.predict(scale_data)
        estimate_df = pd.DataFrame(data=estimate, index=dataframe.index,
                                   columns=["'{0}':{1}".format(self.target_property, self.get_id())])
        return estimate_df

    def fit(self, data=None, verbose=True):
        """Fit the model to feature vectors, and get estimates. If data is not provided, data from
        moldata is used. standard deviation of the estimates is set.

        Args:
            data (array, optional): a matrix of molecules and a feature vector to fit model. Defaults to None.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            array: a vector of estimates
         """
        self.cross_validation(data=data, verbose=verbose)
        return

    def cross_validation(self, data=None, n_splits=3, shuffle=True, verbose=True):
        """Fit the model to feature vectors, and estimate the accuracy by cross validation. If data is
        not provided, data from moldata is used. standard deviation of the estimate is set.

        Args:
            data (array, optional): a matrix of molecules and a feature vector to fit model. Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            array: array of scores of cross validation
        """
        if data is None:
            data = self.get_data()
        target = self.get_target()
        if self.scaler is not None:
            self.scaler.fit(data)
            scale_data = self.scaler.transform(data)
        else:
            scale_data = data
        if verbose:
            print('regression model cross validation target=\'{0}\': data_size={1}: model:{2} n_splits={3} shuffle={4}'
                  .format(self.target_property, len(data), self.get_id_base(), n_splits, shuffle))
        kf = RepeatedKFold(n_splits=n_splits)
        self.cv_score = cross_val_score(self.estimator, scale_data, target, cv=kf, scoring='r2')
        self.estimator.fit(scale_data, target)
        self.score = self.estimator.score(scale_data, target)
        estimate = self.estimator.predict(scale_data)
        self.set_prediction_std(target, estimate)
        self.set_mse(target, estimate)
        # update status
        self.status = self.Status.FIT
        self.selection_mask = None
        return self.cv_score

    def param_optimization(self, data=None, param_grid=None, n_splits=3, shuffle=True, verbose=True):
        """Optimize hyperparameters of the estimator.
        If data is not provided, data from moldata is used. Accuracy of the obtained estimator
        is evaluated by cross validation.

        Args:
            data (array, optional): a matrix of molecules and a feature vector to fit model. Defaults to None.
            param_grid (dict, optional): a map of a hyperparameter to the values for grid search. Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.
        """
        if data is None:
            data = self.get_data()
        if self.selection_mask is not None and self.status >= self.Status.SEL:
            selection = FeatureSelector(self.selection_mask)
            data = selection.transform(data)
        target = self.get_target()
        if self.scaler is not None:
            self.scaler.fit(data)
            scale_data = self.scaler.transform(data)
        else:
            scale_data = data
        if verbose:
            print('regression model parameter optimization target=\'{0}\': data_size={1}: model:{2} n_splits={3} shuffle={4}'
                  .format(self.target_property, len(data), self.get_id_base(), n_splits, shuffle))
        if param_grid is None:
            # use default parameter grid
            param_grid = self.param_grid
        # backup status
        status = self.status
        selection_mask = self.selection_mask
        # search parameters
        self.search_optimized_parameters(scale_data, target, param_grid, n_splits, shuffle)
        if verbose:
            print('optimized parameters: {0}'.format(self.get_params()))
        # fit model with optimized parameters
        self.cross_validation(data=data, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        # restore status
        self.status = status
        self.selection_mask = selection_mask
        if verbose:
            print('R^2 score={0:.2f} cv_score={1:.2f} (+/- {2:.2f})'.
                  format(self.get_score(), self.get_cv_score_mean(), self.get_cv_score_std()))
        # update status
        if self.selection_mask is not None and self.status >= self.Status.SEL:
            self.status = self.Status.OPTSEL
        else:
            self.status = self.Status.OPT
            self.selection_mask = None

    def search_optimized_parameters(self, data, target, param_grid, n_splits, shuffle):
        """Optimize hyperparameters of the estimator by grid search (GridSearchCV).

        Args:
            data (array): a matrix of molecules and a feature vector to fit model.
            target (array): a vector of target property value.
            param_grid (dict): a map of a hyperparameter to the values for grid search.
            n_splits (int): the number of split of partitions for cross validation.
            shuffle (bool): if true, data is shuffled in splitting into partitions.
        """
        # grid search
        kf = RepeatedKFold(n_splits=n_splits)
        search = GridSearchCV(self.estimator, param_grid, cv=kf).fit(data, target)
        self.estimator.set_params(**search.best_params_)

    def plot_estimate(self, file=None, df_data=None, df_target=None, df_category=None):
        """Plot estimation result for a target property.

        Args:
            file (str, optional): a file to save the figure
            df_data (array, optional): a matrix of molecules and a feature vector. Defaults to None.
            df_target (array, optional): a vector of molecules and a target property. Defaults to None.
            df_category (array, optional): a vector of molecules and a category. Defaults to None.
        """
        if self.is_feature_selected():
            data = self.get_selected_data(df_data)
        else:
            data = self.get_data(df_data)
        if df_target is None:
            target = self.get_target()
        else:
            target = df_target.values
        estimate = self.predict_val(data)
        # plot target and estimated value
        fig = plt.figure()
        plt.title("'{0}' by {1}\nR^2 score={2:.2f} cv_score={3:.2f} (+/- {4:.2f}) data={5}".
                  format(self.target_property, self.get_id(),
                         self.get_score(), self.get_cv_score_mean(), self.get_cv_score_std(),
                         len(data)))
        if df_category is not None:
            if df_data is None:
                category = df_category.astype('str').values[self.target_mask]
            else:
                category = df_category.astype('str').values
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            markers = ['o', '^', 's', 'D', '+', 'x']
            values = Counter()
            for val in category:
                values[val] += 1
            for index, (val, count) in enumerate(sorted(values.items(), key=lambda x: -x[1])):
                cat_str = ' ' if val == 'nan' else val
                cat_mask = category == val
                cat_score = r2_score(target[cat_mask], estimate[cat_mask])
                plt.plot(target[cat_mask], estimate[cat_mask],
                         colors[index % len(colors)]+markers[(int(index/len(colors))) % len(markers)],
                         label='{0}: {1:.2f} ({2})'.format(cat_str, cat_score, len(target[cat_mask])))
            plt.legend()
        else:
            plt.plot(target, estimate, 'ro')
        plt.xlabel("property '{0}'".format(self.target_property))
        plt.ylabel("estimate")
        data_min = min(np.min(estimate), np.min(target))
        data_max = max(np.max(estimate), np.max(target))
        data_diff = data_max - data_min
        data_min = data_min - data_diff*0.1
        data_max = data_max + data_diff*0.1
        plt.xlim([data_min, data_max])
        plt.ylim([data_min, data_max])
        x = np.linspace(data_min, data_max, 10)
        y = x
        plt.plot(x, y)
        if file is None:
            plt.show()
        else:
            fig.savefig(file)


class SklearnLinearRegressionModel(SklearnRegressionModel):
    """Base class for linear_model of SciKitLearn
    """

    hyper_params = None
    """str: dictionary of hyper parameters"""

    def __init__(self, estimator, moldata, target_property, features, scaler=True):
        """Constructor of SklearnLinearRegressionModel.

        Args:
            estimator (object): a SciKitLearn estimator object
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            scaler (bool): flag of scaling for standardizing data
        """
        super().__init__(estimator, moldata, target_property, features, scaler=scaler)

    def is_linear_model(self):
        """Get if the model is linear regression

        Returns:
            bool: true if linear regression model
        """
        return True

    def feature_selection(self, threshold=None, n_splits=3, shuffle=True, verbose=True):
        """Select important features for the prediction based on LASSO penalty (SelectFromModel).
        After the feature selection, the accuracy of the estimator is evaluated by cross validation.

        Args:
            threshold (str, float, optional): threshold of coefficient for eliminating meaningless features.
                Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            list: a list of selection mask
        """
        data = self.get_data()
        target = self.get_target()
        if self.scaler is not None:
            self.scaler.fit(data)
            scale_data = self.scaler.transform(data)
        else:
            scale_data = data
        if verbose:
            print('feature selection target=\'{0}\': data_size={1}: model:{2} threshold={3}'.
                  format(self.target_property, len(data), self.get_id(), threshold))
        # backup status
        status = self.status
        selection_mask = self.selection_mask
        # feature selection
        self.selection_threshold = threshold
        selection = SelectFromModel(self.estimator, threshold=threshold).fit(scale_data, target)
        data_new = selection.transform(data)
        if verbose:
            print('feature size:{0} -> {1}'.format(data.shape[1], data_new.shape[1]))
        if data_new.shape[1] > 0:
            self.cross_validation(data_new, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        else:
            self.cross_validation(data, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        # restore status
        self.status = status
        self.selection_mask = selection_mask
        if verbose:
            print('R^2 score={0:.2f} cv_score={1:.2f} (+/- {2:.2f})'.
                  format(self.get_score(), self.get_cv_score_mean(), self.get_cv_score_std()))
        if self.status == self.Status.OPT:
            self.status = self.Status.OPTSEL
        else:
            self.status = self.Status.SEL
        if data_new.shape[1] > 0:
            self.selection_mask = selection.get_support()
        else:
            self.selection_mask = [True] * data.shape[1]
        return self.get_selection_mask()


class LinearRegressionModel(SklearnLinearRegressionModel):
    """Linear (SciKitLearn) linear regression model.

    Attributes:
        estimator (Linear): Linear object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        selection_mask (list): a list of masks for selected features
        status (str): a status of regression model
    """

    hyper_params = []
    """dict: list of hyper parameters"""

    param_grid = {}
    """dict: default grid of grid search hyper parameter optimization"""

    def __init__(self, moldata, target_property, features, scaler=True, **kwargs):
        """Constructor of LinearRegressionModel.

        Args:
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            scaler (bool, optional): flag of scaling for standardizing data. Defaults to True.
            **kwargs (dict, optional): other key word arguments for LinearRegression
        """
        super().__init__(LinearRegression(*kwargs), moldata, target_property, features, scaler=scaler)


class RidgeRegressionModel(SklearnLinearRegressionModel):
    """Ridge (SciKitLearn) linear regression model.

    Attributes:
        estimator (Ridge): Ridge object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        selection_mask (list): a list of masks for selected features
        status (str): a status of regression model
    """

    hyper_params = ['alpha']
    """dict: list of hyper parameters"""

    param_grid = {'alpha': np.logspace(-4, 3, 8)}
    """dict: default grid of grid search hyper parameter optimization"""

    def __init__(self, moldata, target_property, features, alpha=1.0, scaler=True, **kwargs):
        """Constructor of RidgeRegressionModel.

        Args:
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            alpha (float, optional): regularization strength. must be a positive float. Defaults to 1.0.
            scaler (bool, optional): flag of scaling for standardizing data. Default to True
            **kwargs (dict, optional): other key word arguments for Ridge
        """
        super().__init__(Ridge(alpha=alpha, **kwargs), moldata, target_property, features, scaler=scaler)

    def search_optimized_parameters(self, data, target, param_grid, n_splits, shuffle):
        """Optimize hyperparameters of the estimator by LassoCV.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model.
            target (array): a vector of target property value.
            param_grid (dict): a map of a hyperparameter to the values for grid search.
            n_splits (int): the number of split of partitions for cross validation.
            shuffle (bool): if true, data is shuffled in splitting into partitions.
        """
        kf = RepeatedKFold(n_splits=n_splits)
        params = self.estimator.get_params()
        search = RidgeCV(alphas=param_grid['alpha'], cv=kf,
                         fit_intercept=params['fit_intercept'],
                         normalize=params['normalize'])
        search.fit(data, target)
        self.estimator.set_params(alpha=search.alpha_)


class LassoRegressionModel(SklearnLinearRegressionModel):
    """Lasso (SciKitLearn) linear regression model.

    Attributes:
        estimator (Lasso): Lasso object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        selection_mask (list): a list of masks for selected features
        status (str): a status of regression model
    """

    hyper_params = ['alpha']
    """dict: list of hyper parameters"""

    param_grid = {'alpha': np.logspace(-6, 1, 8)}
    """dict: default grid of grid search hyper parameter optimization"""

    def __init__(self, moldata, target_property, features, alpha=1.0, scaler=True, **kwargs):
        """Constructor of LassoRegressionModel.

        Args:
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            alpha (float, optional): regularization strength. must be a positive float. Defaults to 1.0.
            scaler (bool, optional): flag of scaling for standardizing data. Defaults to True.
            **kwargs (dict, optional): other key word arguments for Lasso
        """
        super().__init__(Lasso(alpha=alpha, **kwargs), moldata, target_property, features, scaler=scaler)

    def search_optimized_parameters(self, data, target, param_grid, n_splits, shuffle):
        """Optimize hyperparameters of the estimator by LassoCV.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model.
            target (array): a vector of target property value.
            param_grid (dict): a map of a hyperparameter to the values for grid search.
            n_splits (int): the number of split of partitions for cross validation.
            shuffle (bool): if true, data is shuffled in splitting into partitions.
        """
        #
        kf = RepeatedKFold(n_splits=n_splits)
        params = self.estimator.get_params()
        search = LassoCV(alphas=param_grid['alpha'], cv=kf,
                         fit_intercept=params['fit_intercept'],
                         normalize=params['normalize'],
                         precompute=params['precompute'])
        search.fit(data, target)
        self.estimator.set_params(alpha=search.alpha_)


class ElasticNetRegressionModel(SklearnLinearRegressionModel):
    """ElasticNet (SciKitLearn) linear regression model.

    Attributes:
        estimator (Lasso): Lasso object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        selection_mask (list): a list of masks for selected features
        status (str): a status of regression model
    """

    hyper_params = ['alpha', 'l1_ratio']
    """dict: list of hyper parameters"""

    param_grid = {'alpha': np.logspace(-6, 2, 9), 'l1_ratio': np.linspace(0.0, 1.0, 6)}
    """dict: default grid of grid search hyper parameter optimization"""

    def __init__(self, moldata, target_property, features, alpha=1.0, l1_ratio=0.5,
                 scaler=True, **kwargs):
        """Constructor of ElasticNetRegressionModel.

        Args:
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            alpha (float, optional): regularization strength. must be a positive float. Defaults to 1.0.
            l1_ratio (float, optional): ratio l1 regularization. Defaults to 0.5
            scaler (bool, optional): flag of scaling for standardizing data. Defaults to True
            **kwargs (dict, optional): other key word arguments for Lasso
        """
        super().__init__(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs),
                         moldata, target_property, features, scaler=scaler)

    def search_optimized_parameters(self, data, target, param_grid, n_splits, shuffle):
        """Optimize hyperparameters of the estimator by ElasticNetCV.

        Args:
            data (array): a matrix of molecules and a feature vector to fit model.
            target (array): a vector of target property value.
            param_grid (dict): a map of a hyperparameter to the values for grid search.
            n_splits (int): the number of split of partitions for cross validation.
            shuffle (bool): if true, data is shuffled in splitting into partitions.
        """
        #
        kf = RepeatedKFold(n_splits=n_splits)
        params = self.estimator.get_params()
        search = ElasticNetCV(l1_ratio=param_grid['l1_ratio'], alphas=param_grid['alpha'], cv=kf,
                              fit_intercept=params['fit_intercept'],
                              normalize=params['normalize'],
                              precompute=params['precompute'])
        search.fit(data, target)
        self.estimator.set_params(alpha=search.alpha_, l1_ratio=search.l1_ratio_)


class RandomForestRegressionModel(SklearnRegressionModel):
    """RandomForestRegressor (SciKitLearn) regression model.
     This is a meta estimator that fits a number of classifying decision trees on various sub-samples of the database.

    Attributes:
        estimator (RandomForestRegressor): RandomForestRegressor object
        scaler (object): scaling for standardizing data
        moldata (MolData): a molecule data management object
        target_property (str): name of target property of regression
        features (MergedFeatureSet): a feature set for regression
        params (dict): parameters of an estimator
        prediction_std (float): standard deviation of the prediction
        selection_mask (list): a list of masks for selected features
        status (str): a status of regression model
    """

    hyper_params = ['min_samples_split']
    """dict: list of pyper parameters"""

    param_grid = {'min_samples_split': np.arange(2, 4, dtype=int)}
    """dict: default grid of grid search hyper parameter optimization"""

    def __init__(self, moldata, target_property, features, scaler=True, **kwargs):
        """Constructor of RandomForestRegressionModel.

        Args:
            moldata (MolData): a molecule data management object
            target_property (str): name of target property of regression
            features (MergedFeatureSet): a feature set for regression
            scaler (bool, optional): flag of scaling for standardizing data. Defaults to True
            **kwargs (dict, optional): other key word arguments for KernelRidge
        """
        super().__init__(RandomForestRegressor(*kwargs),
                         moldata, target_property, features, scaler=scaler)

    def feature_selection(self, threshold=None, n_splits=3, shuffle=True, verbose=True):
        """Select important features for the prediction based on LASSO penalty (SelectFromModel).
        After the feature selection, the accuracy of the estimator is evaluated by cross validation.

        Args:
            threshold (str, float, optional): threshold of coefficient for eliminating meaningless features.
                Defaults to None.
            n_splits (int, optional): the number of split of partitions for cross validation. Defaults to 3.
            shuffle (bool, optional): if true, data is shuffled in splitting into partitions. Defaults to True.
            verbose (bool, optional): flag of verbose message. Defaults to True.

        Returns:
            list: a list of selection mask
        """
        data = self.get_data()
        if self.scaler is not None:
            self.scaler.fit(data)
        if verbose:
            print('feature selection target=\'{0}\': data_size={1}: model:{2} threshold={3}'.
                  format(self.target_property, len(data), self.get_id(), threshold))
        # backup status
        status = self.status
        selection_mask = self.selection_mask
        # feature selection
        self.selection_threshold = threshold
        self.cross_validation(data, n_splits=n_splits, shuffle=shuffle, verbose=False)
        if self.selection_threshold is None:
            thresh = 0
        else:
            thresh = self.selection_threshold
        new_selection_mask = self.estimator.feature_importances_ > thresh
        data_new = data[:, new_selection_mask]
        if verbose:
            print('feature size:{0} -> {1}'.format(data.shape[1], data_new.shape[1]))
        if data_new.shape[1] < data.shape[1]:
            self.cross_validation(data_new, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        else:
            self.cross_validation(data, n_splits=n_splits, shuffle=shuffle, verbose=verbose)
        # restore status
        self.status = status
        self.selection_mask = selection_mask
        if verbose:
            print('R^2 score={0:.2f} cv_score={1:.2f} (+/- {2:.2f})'.
                  format(self.get_score(), self.get_cv_score_mean(), self.get_cv_score_std()))
        if self.status == self.Status.OPT:
            self.status = self.Status.OPTSEL
        else:
            self.status = self.Status.SEL
        self.selection_mask = new_selection_mask
        return self.get_selection_mask()


class FeatureSelector:
    """Class for managing feature selection. 

    Attributes:
        mask (list): a list of masks for selected features
    """
    def __init__(self, mask):
        """Constructor of FeatureSelector.

        Args:
            mask (list): a list of masks for selected features
        """
        self.mask = mask

    def transform(self, X):
        """Transform data to include the elements of the selected features

        Args:
            X (array): data we want to transform

        Returns:
            array: transformed data
        """
        return X[:, self.mask]

    def get_support(self):
        """Get the current set of selected features

        Returns:
            list: a list of masks for selected features
        """
        return self.mask

    def set_mask(self, loc, flag):
        """Masking/unmasking a feature

        Args:
            loc (int): location of the feature
            flag (int): a mask of feature at the location
        """
        self.mask[loc] = flag

    def set_masks(self, pairs):
        """Masking/unmasking multiple features

        Args:
            pairs (list): a list whose pair is a location and a flag of the feature
        """
        for p in pairs:
            self.set_mask(p[0], p[1])

    def flip_mask(self, loc):
        """Flipping a mask of one feature

        Args:
            loc (int): location of the mask
        """
        self.mask[loc] = not self.mask[loc]

    def flip_masks(self, locs):
        """Flipping masks of features

        Args:
            locs (list): feature locations
        """
        for loc in locs:
            self.flip_mask(loc)
