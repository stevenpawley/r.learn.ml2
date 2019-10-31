#!/usr/bin/env python
# -- coding: utf-8 --

"""The module rlearn_utils contains functinons to assist
with passing pre-defined scikit learn classifiers
and other utilities for loading/saving training data."""

from __future__ import absolute_import

import grass.script as gs
import numpy as np


def option_to_list(x, dtype=None):
    """
    Parses a multiple choice option from into a list
    
    Args
    ----
    x : str
        String with comma-separated values
    
    dtype : func, optional
        Optionally pass a function to coerce with list elements into a
        specific type, e.g. int
    
    Returns
    -------
    x : list
        List of parsed values
    """

    if x.strip() == '':
        x = None
    elif ',' in x is False:
        x = [x]
    else:
        x = x.split(',')
    
    if dtype:
        x = [dtype(i) for i in x]
    
    if x is not None:
        
        if len(x) == 1 and x[0] == -1:
            x = None
    
    return x


def model_classifiers(estimator, random_state, n_jobs, p, weights=None):
    """
    Provides the classifiers and parameters using by the module

    Args
    ----
    estimator (string): Name of scikit learn estimator
    random_state (float): Seed to use in randomized components
    n_jobs (integer): Number of processing cores to use
    p (dict): Classifier setttings (keys) and values
    weights (string): None, or 'balanced' to add class_weights

    Returns
    -------
    clf (object): Scikit-learn classifier object
    mode (string): Flag to indicate whether classifier performs classification
        or regression
    """
    try:
        from sklearn.experimental import enable_hist_gradient_boosting
    except ImportError:
        pass
    
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor)
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor)
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    # convert balanced boolean to scikit learn method
    if weights is True:
        weights = 'balanced'
    else: weights = None

    classifiers = {
        'SVC': SVC(C=p['C'],
                   class_weight=weights,
                   probability=True,
                   random_state=random_state),
        'SVR': SVR(C=p['C'],
                   epsilon=p['epsilon']),
        'LogisticRegression':
            LogisticRegression(C=p['C'],
                               class_weight=weights,
                               solver='liblinear',
                               random_state=random_state,
                               multi_class='auto',
                               n_jobs=1,
                               fit_intercept=True),
        'LinearRegression':
            LinearRegression(n_jobs=n_jobs,
                             fit_intercept=True),
        'DecisionTreeClassifier':
            DecisionTreeClassifier(max_depth=p['max_depth'],
                                   max_features=p['max_features'],
                                   min_samples_split=p['min_samples_split'],
                                   min_samples_leaf=p['min_samples_leaf'],
                                   class_weight=weights,
                                   random_state=random_state),
        'DecisionTreeRegressor':
            DecisionTreeRegressor(max_features=p['max_features'],
                                  min_samples_split=p['min_samples_split'],
                                  min_samples_leaf=p['min_samples_leaf'],
                                  random_state=random_state),
        'RandomForestClassifier':
            RandomForestClassifier(n_estimators=p['n_estimators'],
                                   max_features=p['max_features'],
                                   min_samples_split=p['min_samples_split'],
                                   min_samples_leaf=p['min_samples_leaf'],
                                   class_weight=weights,
                                   random_state=random_state,
                                   n_jobs=n_jobs,
                                   oob_score=True),
        'RandomForestRegressor':
            RandomForestRegressor(n_estimators=p['n_estimators'],
                                  max_features=p['max_features'],
                                  min_samples_split=p['min_samples_split'],
                                  min_samples_leaf=p['min_samples_leaf'],
                                  random_state=random_state,
                                  n_jobs=n_jobs,
                                  oob_score=True),
        'ExtraTreesClassifier':
            ExtraTreesClassifier(n_estimators=p['n_estimators'],
                                 max_features=p['max_features'],
                                 min_samples_split=p['min_samples_split'],
                                 min_samples_leaf=p['min_samples_leaf'],
                                 class_weight=weights,
                                 random_state=random_state,
                                 n_jobs=n_jobs,
                                 bootstrap=True,
                                 oob_score=True),
        'ExtraTreesRegressor':
            ExtraTreesRegressor(n_estimators=p['n_estimators'],
                                max_features=p['max_features'],
                                min_samples_split=p['min_samples_split'],
                                min_samples_leaf=p['min_samples_leaf'],
                                random_state=random_state,
                                bootstrap=True,
                                n_jobs=n_jobs,
                                oob_score=True),
        'GradientBoostingClassifier':
            GradientBoostingClassifier(learning_rate=p['learning_rate'],
                                       n_estimators=p['n_estimators'],
                                       max_depth=p['max_depth'],
                                       min_samples_split=p['min_samples_split'],
                                       min_samples_leaf=p['min_samples_leaf'],
                                       subsample=p['subsample'],
                                       max_features=p['max_features'],
                                       random_state=random_state),
        'GradientBoostingRegressor':
            GradientBoostingRegressor(learning_rate=p['learning_rate'],
                                      n_estimators=p['n_estimators'],
                                      max_depth=p['max_depth'],
                                      min_samples_split=p['min_samples_split'],
                                      min_samples_leaf=p['min_samples_leaf'],
                                      subsample=p['subsample'],
                                      max_features=p['max_features'],
                                      random_state=random_state),
        'HistGradientBoostingClassifier':
            GradientBoostingClassifier(learning_rate=p['learning_rate'],
                                       n_estimators=p['n_estimators'],
                                       max_depth=p['max_depth'],
                                       min_samples_split=p['min_samples_split'],
                                       min_samples_leaf=p['min_samples_leaf'],
                                       subsample=p['subsample'],
                                       max_features=p['max_features'],
                                       random_state=random_state),
        'HistGradientBoostingRegressor':
            GradientBoostingRegressor(learning_rate=p['learning_rate'],
                                      n_estimators=p['n_estimators'],
                                      max_depth=p['max_depth'],
                                      min_samples_split=p['min_samples_split'],
                                      min_samples_leaf=p['min_samples_leaf'],
                                      subsample=p['subsample'],
                                      max_features=p['max_features'],
                                      random_state=random_state),
        'GaussianNB': GaussianNB(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=p['n_neighbors'],
                                                     weights=p['weights'],
                                                     n_jobs=n_jobs),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=p['n_neighbors'],
                                                   weights=p['weights'],
                                                   n_jobs=n_jobs)
    }

    # define classifier
    try:
        clf = classifiers[estimator]
    except:
        gs.fatal('HistGradienBoostingClassifier and ' 
           'HistGradientBoostingRegressor only available on '
           'scikit-learn version >= 0.21.3')

    # classification or regression
    if estimator == 'LogisticRegression' \
        or estimator == 'DecisionTreeClassifier' \
        or estimator == 'RandomForestClassifier' \
        or estimator == 'ExtraTreesClassifier' \
        or estimator == 'GradientBoostingClassifier' \
        or estimator == 'HistGradientBoostingClassifier' \
        or estimator == 'GaussianNB' \
        or estimator == 'LinearDiscriminantAnalysis' \
        or estimator == 'QuadraticDiscriminantAnalysis' \
        or estimator == 'SVC' \
        or estimator == 'KNeighborsClassifier':
        mode = 'classification'
    else:
        mode = 'regression'

    return (clf, mode)


def scoring_metrics(mode):
    """
    Simple helper function to return a suite of scoring methods depending on
    if classification or regression is required
    
    Args
    ----
    mode : str
        'classification' or 'regression'
    
    Returns
    -------
    scoring : list
        List of sklearn scoring metrics
    search_scorer : func, or str
        Scoring metric to use for hyperparameter tuning
    """
    
    from sklearn import metrics
    from sklearn.metrics import make_scorer
    
    if mode == 'classification':
        scoring = {
            'accuracy': metrics.accuracy_score,
            'balanced_accuracy': metrics.balanced_accuracy_score,
            'matthews_correlation_coefficient': metrics.matthews_corrcoef,
            'kappa': metrics.cohen_kappa_score
            }

        search_scorer = make_scorer(metrics.matthews_corrcoef)
    
    else:
        scoring = {
            'r2' : metrics.r2_score,
            'explained_variance': metrics.explained_variance_score,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error
            }
        
        search_scorer = make_scorer(metrics.r2_score)
    
    return(scoring, search_scorer)



def save_training_data(X, y, cat, groups, file):
    """
    Saves any extracted training data to a csv file.
    
    Training data is saved in the following format:
        col (0..n) : feature data
        col (n) : response data
        col (n+1): grass cat value
        col (n+2): group idx

    Args
    ----
    X (2d numpy array): Numpy array containing predictor values
    y (1d numpy array): Numpy array containing labels
    cat (1d numpy array): Numpy array of GRASS key column
    groups (1d numpy array): Numpy array of group labels
    file (string): Path to a csv file to save data to
    """

    # if there are no group labels, create a nan filled array
    if groups is None:
        groups = np.empty((y.shape[0]))
        groups[:] = np.nan

    training_data = np.column_stack([X, y, cat, groups])
    np.savetxt(file, training_data, delimiter=',')


def load_training_data(file):
    """
    Loads training data and labels from a csv file

    Args
    ----
    file (string): Path to a csv file to save data to

    Returns
    -------
    X (2d numpy array): Numpy array containing predictor values
    y (1d numpy array): Numpy array containing labels
    cat (1d numpy array): Numpy array of GRASS key column
    groups (1d numpy array): Numpy array of group labels, or None
    """

    training_data = np.loadtxt(file, delimiter=',')
    n_cols = training_data.shape[1]
    last_Xcol = n_cols-3

    # group_ids stored in last column
    groups = training_data[:, -1]

    if bool(np.isnan(groups).all()) is True:
        groups = None
        
    # cat stored in 2nd last column
    cat = training_data[:, -2]
    
    # response stored in 3rd last column
    y = training_data[:, -3]
    X = training_data[:, 0:last_Xcol]

    return (X, y, cat, groups)
