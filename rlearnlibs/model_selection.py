#!/usr/bin/env python
# -- coding: utf-8 --

from __future__ import absolute_import

from copy import deepcopy
import numpy as np


def cross_val(estimator, X, y, idx, cv, n_jobs, fit_params):
    """
    Cross validation function
    
    Parameters
    ----------
    estimator : scikit-learn estimator
    X : ndarray
        2d array of training data
    y : ndarray
        1d array of response data
    idx : ndarray
        1d array of unique identifiers (i.e. GRASS cat attribute)
    cv : model_selection function
    
    Returns
    -------
    preds : dict
    """
    from joblib import Parallel, delayed
    
    estimator = deepcopy(estimator)
        
    preds = {
        'y_pred': np.zeros((0, )),
        'y_true': np.zeros((0, )),
        'cat': np.zeros((0, )),
        'fold': np.zeros((0, ))
        }
    
    res = Parallel(n_jobs=n_jobs)(
        delayed(_cross_val_fit)(estimator, X, y, fold, train, test, fit_params)
        for fold, (train, test) in enumerate(cv.split(X, y)))
    
    for fold, y_pred in enumerate(res):
        preds['y_pred'] = np.concatenate((preds['y_pred'], y_pred))
        preds['fold'] = np.concatenate((preds['fold'], np.repeat(fold, y_pred.shape[0])))

    preds['y_true'] = y
    preds['cat'] = idx
    
    return preds


def _cross_val_fit(estimator, X, y, fold, train_idx, test_idx, fit_params):
    """Fit function used for cross validation
    """
    
    estimator = deepcopy(estimator)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]
    
    estimator.fit(X_train, y_train, **fit_params)
    y_pred = estimator.predict(X_test)

    return y_pred
