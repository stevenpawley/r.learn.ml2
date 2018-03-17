#!/usr/bin/env python
# -- coding: utf-8 --

"""The module rlearn_crossval contains functions to perform
model validation and permutation feature importances."""

from __future__ import absolute_import
import numpy as np
from numpy.random import RandomState


def specificity_score(y_true, y_pred):
    """Calculate specificity score for a binary classification

    Parameters
    ----------
    y_true : 1d array-like
        True values of class labels
    y_pred : 1d array-like
        Predicted class labels

    Returns
    -------
    specificity : float
        Specificity score"""

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0][0])
    fp = float(cm[0][1])

    return tn/(tn+fp)


def permutation_importance(estimator, X, y, groups=None, cv=3, n_permutations=25,
                           scoring='accuracy', fit_params=None, n_jobs=-1,
                           random_state=None):
    """Perform permutation-based feature importance during cross-validation

    Parameters
    ----------
    estimator : Scikit-learn estimator
        Estimator is required to have been fitted to a training partition
    X : 2d array-like
        2d numpy array containing data from a test partition in
        (n_observations, n_features) shape
    y : 1d array-like
        1d numpy array containing class labels or response values from a
        test partition
    groups : 1d array-like
        1d numpy array of len(y) containing group labels.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    n_permutations : int
        Number of random permutations to apply
    scorer : Scikit-learn metric function
        Metric to use for scoring of permutation importance
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    n_jobs : int
        Number of processing cores
    random_state : float or int
        Seed to pass to the numpy random.seed

    Returns
    -------
    scores : 2d array-like
        2d numpy array of scores for each predictor following permutation
    
    Notes
    -----
    
    Procedure:
    1. Pass fitted estimator and test partition X y
    2. Assess AUC on the test partition (bestauc)
    3. Permute each variable and assess the difference between bestauc and
       the messed-up variable
    4. Repeat (3) for many random permutations
    5. Average the repeats"""

    from sklearn.externals.joblib import Parallel, delayed
    from sklearn.metrics.scorer import check_scoring
    from sklearn.model_selection._split import check_cv
    from sklearn.base import is_classifier, clone

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv.random_state = random_state
    scorer = check_scoring(estimator, scoring)

    # calculate cross-validation scores without permutation
    scores = Parallel(n_jobs=n_jobs)(
        delayed(__fit_and_permute)(clone(estimator), X, y, groups, train, test,
                scorer, n_permutations, random_state, fit_params)
        for train, test in cv.split(X, y, groups))
    
    # average the repeats
    scores = np.asarray(scores)
    scores_mean = scores.mean(axis=0)
    scores_std = scores.std(axis=0)

    return (scores_mean, scores_std)


def __fit_and_permute(estimator, X, y, groups, train, test, scorer, n_permutations, random_state, fit_params):
    """Fit classifiers/regressors in parallel

    Parameters
    ----------
    estimator : Scikit-learn estimator
        Estimator
    X : 2d array-like
        2d numpy array containing data from a training partition in
        (n_observations, n_features) shape
    y : 1d array-like
        1d numpy array containing class labels or response values from a
        training partition
    groups : 1d array-like
        1d numpy array of len(y) containing group labels
    train_indices : 1d array-like
        1d numpy array of indices to use for training partition
    sample_weight : 1d array-like
         1d numpy array of len(y) containing weights to use during fitting"""
    
    rstate = RandomState(random_state)
    
    # create training and test folds
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    # adjust length of groups
    try:
        groups_train = groups[train]
    except:
        groups_train = None

    # adjust length of fit params
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    # scoring on original data
    try:
        estimator.fit(X_train, y_train, groups=groups_train, **fit_params)
    except:
        estimator.fit(X_train, y_train, **fit_params)
    
    best_score = scorer(estimator, X_test, y_test)
    
    # scoring on permuted data
    permuted_scores = np.zeros((n_permutations, X_test.shape[1]))
    for n in range(n_permutations):
        for i in range(X_test.shape[1]):
            Xscram = np.copy(X_test)
            Xscram[:, i] = rstate.choice(X_test[:, i], X_test.shape[0])
            permuted_scores[n, i] = scorer(estimator, Xscram, y_test)
    
    # average the repeats
    permuted_scores = permuted_scores.mean(axis=0)
    
    # calculate difference from original and permuted scores
    diff = best_score - permuted_scores

    return diff
