#!/usr/bin/env python
# -- coding: utf-8 --
#
############################################################################
# MODULE:        r.learn.ml
# AUTHOR:        Steven Pawley
# PURPOSE:       Supervised classification and regression of GRASS rasters
#                using the python scikit-learn package
#
# COPYRIGHT: (c) 2017 Steven Pawley, and the GRASS Development Team
#                This program is free software under the GNU General Public
#                for details.
#
#############################################################################
# July, 2017. Jaan Janno, Mait Lang. Bugfixes concerning crossvalidation failure
# when class numeric ID-s were not continous increasing +1 each.
# Bugfix for processing index list of nominal layers.

#%module
#% description: Perform cross-validation of a model created using r.learn.train
#% keyword: raster
#% keyword: classification
#% keyword: regression
#% keyword: machine learning
#% keyword: scikit-learn
#% keyword: cross-validation
#%end

#%option G_OPT_F_INPUT
#% key: load_model
#% label: Load model from file
#% description: File representing pickled scikit-learn estimator model
#% required: yes
#% guisection: Required
#%end

#%option
#% key: cv
#% type: integer
#% description: Number of cross-validation folds
#% answer: 1
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: preds_file
#% label: Save cross-validation predictions to csv
#% required: no
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: classif_file
#% label: Save classification report to csv
#% required: no
#% guisection: Cross validation
#%end

#%option
#% key: random_state
#% type: integer
#% description: Seed to use for random state
#% answer: 1
#% guisection: Optional
#%end

#%option
#% key: n_jobs
#% type: integer
#% description: Number of cores for multiprocessing, -2 is n_cores-1
#% answer: 1
#% guisection: Optional
#%end

from __future__ import absolute_import, print_function

import atexit
import os
import sys
import warnings
from copy import deepcopy

import grass.script as gs
import numpy as np
from grass.script.utils import get_lib_path
import joblib

path = get_lib_path(modname='r.learn.ml')

if path is None:
    gs.fatal('Not able to find the r.learn library directory')
sys.path.append(path)

from utils import scoring_metrics


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def main():
    try:
        import sklearn
        from sklearn.model_selection import (StratifiedKFold, GroupKFold, 
                                             KFold, cross_val_predict)
        from sklearn import metrics
        from sklearn.metrics import classification_report

        if sklearn.__version__ < '0.21':
            gs.fatal("Scikit learn 0.21 or newer is required")
        
    except ImportError:
        gs.fatal("Scikit learn 0.21 or newer is not installed")

    try:
        import pandas as pd
        
    except ImportError:
        gs.fatal("Pandas is not installed ")

    # parser options
    model_load = options['load_model']
    cv = int(options['cv'])
    preds_file = options['preds_file']
    classif_file = options['classif_file']
    fimp_file = options['fimp_file']
    random_state = int(options['random_state'])
    n_jobs = int(options['n_jobs'])

    estimator, X, y, cat, group_id, mode, fit_params = joblib.load(model_load)
    scoring, search_scorer = scoring_metrics(mode)

    # define the outer search resampling method
    if cv > 1:
        if group_id is None and mode == 'classification':
            outer = StratifiedKFold(n_splits=cv, random_state=random_state)
        elif group_id is None and mode == 'regression':
            outer = KFold(n_splits=cv, random_state=random_state)
        else:
            outer = GroupKFold(n_splits=cv)

    # cross-validation    
    if (mode == 'classification' and 
        cv > np.histogram(y, bins=np.unique(y))[0].min()):
        gs.message(os.linesep)
        gs.fatal('Number of cv folds is greater than number of '
                    'samples in some classes')
        
    gs.message(os.linesep)
    gs.message("Cross validation global performance measures......:")

    if (mode == 'classification' and len(np.unique(y)) == 2 and
        all([0, 1] == np.unique(y))):    
        scoring['roc_auc'] = metrics.roc_auc_score    
    
    
    preds = cross_val_predict(estimator, X, y, group_id, cv=outer,
                                n_jobs=n_jobs, fit_params=fit_params)
    
    test_idx = [test for train, test in outer.split(X, y)]        
    n_fold = np.zeros((0, ))
    
    for fold in range(outer.get_n_splits()):
        n_fold = np.hstack(
            (n_fold, np.repeat(fold, test_idx[fold].shape[0])))
    
    preds = {
        'y_pred': preds,
        'y_true': y,
        'cat': cat,
        'fold': n_fold
    }
    
    preds = pd.DataFrame(data=preds, 
                         columns=['y_pred', 'y_true', 'cat', 'fold'])        
    gs.message(os.linesep)
    gs.message('Global cross validation scores...')
    gs.message(os.linesep)
    gs.message(('Metric \t Mean \t Error'))    

    for name, func in scoring.items():
        score_mean = (
            preds.groupby('fold').
            apply(lambda x: func(x['y_true'], x['y_pred'])).
            mean())

        score_std = (
            preds.groupby('fold').
            apply(lambda x: func(x['y_true'], x['y_pred'])).
            std())
        
        gs.message(name + 
                   '\t' + str(score_mean.round(3)) + 
                   '\t' + str(score_std.round(3)))

    if mode == 'classification':    
        gs.message(os.linesep)
        gs.message('Cross validation class performance measures......:')
        
        report_str = classification_report(
            y_true=preds['y_true'], 
            y_pred=preds['y_pred'], 
            output_dict=False)
        
        report = classification_report(
            y_true=preds['y_true'], 
            y_pred=preds['y_pred'], 
            output_dict=True)
        report = pd.DataFrame(report)
        
        gs.message(report_str)
        
        if classif_file != '':
            report.to_csv(classif_file, mode='w', index=True)
            
    # write cross-validation predictions to csv file
    if preds_file != '':
        preds.to_csv(preds_file, mode='w', index=False)
        text_file = open(preds_file + 't', "w")
        text_file.write('"Real", "Real", "integer", "integer"')
        text_file.close()


if __name__ == "__main__":
    options, flags = gs.parser()
    main()
