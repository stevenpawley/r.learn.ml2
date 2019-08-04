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
#% description: Supervised classification and regression of GRASS rasters using the python scikit-learn package
#% keyword: raster
#% keyword: classification
#% keyword: regression
#% keyword: machine learning
#% keyword: scikit-learn
#%end

#%option G_OPT_I_GROUP
#% key: group
#% label: Group of raster layers to be classified
#% description: GRASS imagery group of raster maps representing feature variables to be used in the machine learning model
#% required: yes
#% multiple: no
#%end

#%option G_OPT_R_INPUT
#% key: training_map
#% label: Labelled pixels
#% description: Raster map with labelled pixels for training
#% required: no
#% guisection: Required
#%end

#%option G_OPT_V_INPUT
#% key: training_points
#% label: Vectorfile with training samples
#% description: Vector points map where each point is used as training sample. Handling of missing values in training data can be choosen later.
#% required: no
#% guisection: Required
#%end

#%option G_OPT_DB_COLUMN
#% key: field
#% label: Response attribute column
#% description: Name of attribute column in training_points table containing response values
#% required: no
#% guisection: Required
#%end

#%option G_OPT_F_OUTPUT
#% key: save_model
#% label: Save model to file (for compression use e.g. '.gz' extension)
#% required: no
#% guisection: Required
#%end

#%option string
#% key: model_name
#% label: model_name
#% description: Supervised learning model to use
#% answer: RandomForestClassifier
#% options: LogisticRegression,LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis,KNeighborsClassifier,GaussianNB,DecisionTreeClassifier,DecisionTreeRegressor,RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingClassifier,GradientBoostingRegressor,SVC,EarthClassifier,EarthRegressor
#% guisection: Estimator settings
#% required: no
#%end

#%option
#% key: c
#% type: double
#% label: Inverse of regularization strength
#% description: Inverse of regularization strength (LogisticRegression and SVC)
#% answer: 1.0
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: max_features
#% type: integer
#% label: Number of features available during node splitting; zero uses estimator defaults
#% description: Number of features available during node splitting (tree-based classifiers and regressors)
#% answer:0
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: max_depth
#% type: integer
#% label: Maximum tree depth; zero uses estimator defaults
#% description: Maximum tree depth for tree-based method; zero uses estimator defaults (full-growing for Decision trees and Randomforest, 3 for GBM)
#% answer:0
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: min_samples_split
#% type: integer
#% label: The minimum number of samples required for node splitting
#% description: The minimum number of samples required for node splitting in tree-based estimators
#% answer: 2
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: min_samples_leaf
#% type: integer
#% label: The minimum number of samples required to form a leaf node
#% description: The minimum number of samples required to form a leaf node in tree-based estimators
#% answer: 1
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: n_estimators
#% type: integer
#% label: Number of estimators
#% description: Number of estimators (trees) in ensemble tree-based estimators
#% answer: 100
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: learning_rate
#% type: double
#% label: learning rate
#% description: learning rate (also known as shrinkage) for gradient boosting methods
#% answer: 0.1
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: subsample
#% type: double
#% label: The fraction of samples to be used for fitting
#% description: The fraction of samples to be used for fitting, controls stochastic behaviour of gradient boosting methods
#% answer: 1.0
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: max_degree
#% type: integer
#% label: The maximum degree of terms in forward pass
#% description: The maximum degree of terms in forward pass for Py-earth
#% answer: 1
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option
#% key: n_neighbors
#% type: integer
#% label: Number of neighbors to use
#% description: Number of neighbors to use
#% answer: 5
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option string
#% key: weights
#% label: weight function
#% description: Distance weight function for k-nearest neighbours model prediction
#% answer: uniform
#% options: uniform,distance
#% multiple: yes
#% guisection: Estimator settings
#%end

#%option string
#% key: grid_search
#% label: Resampling method to use for hyperparameter optimization
#% description: Resampling method to use for hyperparameter optimization
#% options: cross-validation,holdout
#% answer: cross-validation
#% multiple: no
#% guisection: Estimator settings
#%end

#%option G_OPT_R_INPUT
#% key: category_maps
#% required: no
#% multiple: yes
#% label: Names of categorical rasters within the imagery group
#% description: Names of categorical rasters within the imagery group that will be one-hot encoded. Leave empty if none.
#% guisection: Optional
#%end

#%option G_OPT_R_INPUT
#% key: group_raster
#% label: Custom group ids for training samples from GRASS raster
#% description: GRASS raster containing group ids for training samples. Samples with the same group id will not be split between training and test cross-validation folds
#% required: no
#% guisection: Cross validation
#%end

#%option
#% key: cv
#% type: integer
#% description: Number of cross-validation folds
#% answer: 1
#% guisection: Cross validation
#%end

#%flag
#% key: t
#% description: Perform hyperparameter tuning only
#% guisection: Cross validation
#%end

#%flag
#% key: f
#% label: Estimate permutation-based feature importances
#% description: Estimate feature importance using a permutation-based method
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: errors_file
#% label: Save cross-validation global accuracy results to csv
#% required: no
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: preds_file
#% label: Save cross-validation predictions to csv
#% required: no
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: fimp_file
#% label: Save feature importances to csv
#% required: no
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: param_file
#% label: Save hyperparameter search scores to csv
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
#% answer: -2
#% guisection: Optional
#%end

#%flag
#% key: s
#% label: Standardization preprocessing
#% description: Standardize feature variables (convert values the get zero mean and unit variance).
#% guisection: Optional
#%end

#%flag
#% key: b
#% description: Balance training data using class weights
#% guisection: Optional
#%end

#%option G_OPT_F_OUTPUT
#% key: save_training
#% label: Save training data to csv
#% required: no
#% guisection: Optional
#%end

#%option G_OPT_F_INPUT
#% key: load_training
#% label: Load training data from csv
#% required: no
#% guisection: Optional
#%end

#%rules
#% required: training_map,training_points,load_training
#% exclusive: training_map,training_points,load_training
#% exclusive: load_training,save_training
#%end

from __future__ import absolute_import, print_function

import atexit
import os
import sys
import warnings
import numpy as np
from copy import deepcopy

import grass.script as gs
from grass.script.utils import get_lib_path

path = get_lib_path(modname='r.learn.ml')

if path is None:
    gs.fatal('Not able to find the r.learn library directory')
sys.path.append(path)

from utils import (
    model_classifiers, load_training_data, save_training_data, option_to_list,
    scoring_metrics)
from raster import RasterStack


tmp_rast = []

def cleanup():
    for rast in tmp_rast:
        gs.run_command(
            "g.remove", name=rast, type='raster', flags='f', quiet=True)

def warn(*args, **kwargs):
    pass

warnings.warn = warn


def main():
    try:
        import sklearn

        if sklearn.__version__ < '0.20':
            gs.fatal("Scikit learn 0.20 or newer is required")
        
    except ImportError:
        gs.fatal("Scikit learn 0.20 or newer is not installed")


    try:
        import pandas as pd
        
    except ImportError:
        gs.fatal("Pandas is not installed ")

    # -------------------------------------------------------------------------
    # Parser options
    # -------------------------------------------------------------------------

    # required gui section
    group = options['group']
    training_map = options['training_map']
    training_points = options['training_points']
    field = options['field']
    model_save = options['save_model']

    # estimator gui section
    model_name = options['model_name']
    grid_search = options['grid_search']
    hyperparams = {
        'C': options['c'],
        'min_samples_split': options['min_samples_split'],
        'min_samples_leaf': options['min_samples_leaf'],
        'n_estimators': options['n_estimators'],
        'learning_rate': options['learning_rate'],
        'subsample': options['subsample'],
        'max_depth': options['max_depth'],
        'max_features': options['max_features'],
        'max_degree': options['max_degree'],
        'n_neighbors': options['n_neighbors'],
        'weights': options['weights']
        }

    # cross validation
    cv = int(options['cv'])
    group_raster = options['group_raster']
    tune_only = flags['t']
    importances = flags['f']
    errors_file = options['errors_file']
    preds_file = options['preds_file']
    fimp_file = options['fimp_file']
    param_file = options['param_file']

    # general options
    norm_data = flags['s']
    category_maps = option_to_list(options['category_maps'])
    random_state = int(options['random_state'])
    load_training = options['load_training']
    save_training = options['save_training']
    n_jobs = int(options['n_jobs'])
    balance = flags['b']

    # -------------------------------------------------------------------------
    # Make dicts for hyperparameters, datatypes and parameters for tuning
    # -------------------------------------------------------------------------

    hyperparams_type = dict.fromkeys(hyperparams, int)
    hyperparams_type['C'] = float
    hyperparams_type['learning_rate'] = float
    hyperparams_type['subsample'] = float
    hyperparams_type['weights'] = str
    param_grid = deepcopy(hyperparams_type)
    param_grid = dict.fromkeys(param_grid, None)

    for key, val in hyperparams.items():
        # split any comma separated strings and add them to the param_grid
        if ',' in val:
            
            # add all vals to param_grid
            param_grid[key] = [hyperparams_type[key](i)
                for i in val.split(',')]
            
            # use first param for default
            hyperparams[key] = [hyperparams_type[key](i)
                for i in val.split(',')][0]
        
        # else convert the single strings to int or float
        else:
            hyperparams[key] = hyperparams_type[key](val)

    if hyperparams['max_depth'] == 0: hyperparams['max_depth'] = None
    if hyperparams['max_features'] == 0: hyperparams['max_features'] = 'auto'
    param_grid = {k: v for k, v in param_grid.items() if v is not None}

    # retrieve sklearn estimator object and parameters
    estimator, mode = model_classifiers(
        model_name, random_state, n_jobs, hyperparams, balance)

    # remove dict keys that are incompatible for the selected estimator
    estimator_params = estimator.get_params()
    param_grid = {
        key: value for key, value in param_grid.items()
        if key in estimator_params
        }
    
    scoring, search_scorer = scoring_metrics(mode)

    # -------------------------------------------------------------------------
    # Error checking of input options
    # -------------------------------------------------------------------------

    # feature importances selected by no cross-validation scheme used
    if importances is True and cv == 1:
        gs.fatal('Feature importances require cross-validation cv > 1')

    # check for field attribute if training_points are used
    if training_points != '' and field == '':
        gs.fatal('No attribute column specified for training points')

    # check that cv > 1 if hyperparameter tuning is selected
    if (any(param_grid) is True and cv == 1 and
        grid_search == 'cross-validation'):
        gs.fatal(
            'Hyperparameter search using cross validation requires cv > 1')

    # check the cross-validation occurs if feature importances is True
    if importances is True and tune_only is True:
        gs.fatal(
            'Permutation feature importances require cross validation')

    if importances is True and cv == 1:
        gs.fatal(
            'Permutation feature importances require cv > 1')

    # -------------------------------------------------------------------------
    # Define RasterStack
    # -------------------------------------------------------------------------

    # fetch individual raster names from group
    maplist = (gs.read_command("i.group", group=group, flags="g").
               split(os.linesep)[:-1])
    
    # create RasterStack
    stack = RasterStack(rasters=maplist)
    
    if category_maps is not None:
        stack.categorical = category_maps

    # -------------------------------------------------------------------------
    # Extract training data
    # -------------------------------------------------------------------------

    # Sample training data and group id
    if load_training != '':
        X, y, group_id, sample_coords = load_training_data(load_training)
    else:
        gs.message('Extracting training data')

        # append spatial clumps or group raster to the predictors
        if group_raster != '':
            stack.append(group_raster)

        # extract training data
        if training_map != '':
            X, y, sample_coords = stack.extract_pixels(training_map)
        elif training_points != '':
            X, y, sample_coords = stack.extract_points(training_points, field)
        
        y = y.flatten()  # reshape to 1 dimension

        # take group id from last column and remove from predictors
        if group_raster != '':
            group_id = X[:, -1]
            X = np.delete(X, -1, axis=1)
            stack.drop(group_raster)
        else:
            group_id = None

        # check for labelled pixels and training data
        if y.shape[0] == 0 or X.shape[0] == 0:
            gs.fatal('No training pixels or pixels in imagery group '
                     '...check computational region')

        # shuffle data
        from sklearn.utils import shuffle
        
        if group_id is None:
            X, y, sample_coords = shuffle(
                X, y, sample_coords, random_state=random_state)
        else:
            X, y, sample_coords, group_id = shuffle(
                X, y, sample_coords, group_id, random_state=random_state)

        # optionally save extracted data to .csv file
        if save_training != '':
            save_training_data(
                X, y, group_id, sample_coords, save_training)

    # -------------------------------------------------------------------------
    # Define the inner search resampling method
    # -------------------------------------------------------------------------

    from sklearn.model_selection import (
        GridSearchCV, StratifiedKFold, GroupKFold, KFold, ShuffleSplit,
        GroupShuffleSplit)

    # define inner resampling using cross-validation method
    if any(param_grid) is True and grid_search == 'cross-validation':
        
        if group_id is None and mode == 'classification':
            inner = StratifiedKFold(n_splits=cv, random_state=random_state)
        
        elif group_id is None and mode == 'regression':
            inner = KFold(n_splits=cv, random_state=random_state)
        
        else:
            inner = GroupKFold(n_splits=cv)

    # define inner resampling using the holdout method
    elif any(param_grid) is True and grid_search == 'holdout':
        
        if group_id is None:
            inner = ShuffleSplit(
                n_splits=1, test_size=0.33, random_state=random_state)
        
        else:
            inner = GroupShuffleSplit(
                n_splits=1, test_size=0.33, random_state=random_state)
    else:
        inner = None

    # -------------------------------------------------------------------------
    # Define the outer search resampling method
    # -------------------------------------------------------------------------
    if cv > 1:
        
        if group_id is None and mode == 'classification':
            outer = StratifiedKFold(n_splits=cv, random_state=random_state)
            
        elif group_id is None and mode == 'regression':
            outer = KFold(n_splits=cv, random_state=random_state)
            
        else:
            outer = GroupKFold(n_splits=cv)

    # -------------------------------------------------------------------------
    # Define sample weights for estimators that require weights in fit method
    # -------------------------------------------------------------------------

    # estimators that take sample_weights
    if balance is True and mode == 'classification' and model_name in (
            'GradientBoostingClassifier', 'GaussianNB'):
        
        from sklearn.utils import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced', classes=(y), y=y)
        
    else:
        class_weights = None

    # -------------------------------------------------------------------------
    # Define the preprocessing pipeline
    # -------------------------------------------------------------------------
    
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # standardization
    if norm_data is True and category_maps is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        trans = ColumnTransformer(
            remainder='passthrough',
            transformers=[
                ('scaling', scaler,
                 np.setxor1d(range(stack.count),
                             stack.categorical).astype('int'))])
        
    # onehot encoding
    if category_maps is not None:
        from sklearn.preprocessing import OneHotEncoder            
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        trans.transformers.append(('onehot', enc, stack.categorical))

    # combine transformers
    if norm_data is True or category_maps is not None:       
        estimator = Pipeline([('preprocessing', trans),
                              ('estimator', estimator)])

    # -------------------------------------------------------------------------
    # Create the hyperparameter grid search method
    # -------------------------------------------------------------------------

    # check if dict contains and keys - perform GridSearchCV
    if any(param_grid) is True:

        # if Pipeline then change param_grid keys to named_step
        if isinstance(estimator, Pipeline):
            for key in param_grid.keys():
                newkey = 'estimator__' + key
                param_grid[newkey] = param_grid.pop(key)

        # create grid search method
        estimator = GridSearchCV(
            estimator=estimator, param_grid=param_grid,
            scoring=search_scorer, n_jobs=n_jobs, cv=inner)

    # -------------------------------------------------------------------------
    # Estimator training
    # -------------------------------------------------------------------------

    gs.message(os.linesep)
    gs.message(('Fitting model using ' + model_name))

    # fitting ensuring that all options are passed
    if (model_name in ('GradientBoostingClassifier', 'GausianNB')
        and balance is True):
        if isinstance(estimator, Pipeline):
            fit_params = {'estimator__sample_weight': class_weights}
        else:
            fit_params = {'sample_weight': class_weights}
    else:
        fit_params = {}

    if isinstance(inner, (GroupKFold, GroupShuffleSplit)):
        estimator.fit(X, y, groups=group_id, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)

    # message best hyperparameter setup and optionally save using pandas
    if any(param_grid) is True:
        gs.message(os.linesep)
        gs.message('Best parameters:')
        gs.message(str(estimator.best_params_))
        if param_file != '':
            param_df = pd.DataFrame(estimator.cv_results_)
            param_df.to_csv(param_file)

    # ---------------------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------------------
    
    if cv > 1 and tune_only is not True:
        from model_selection import cross_val
        from sklearn.metrics import classification_report
        from sklearn import metrics
        
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
        
        y_pred = cross_val_predict(
            estimator=estimator, X=X, y=y, cv=outer, n_jobs=n_jobs, 
            fit_params=fit_params)
        
        gs.message(os.linesep)
        gs.message('Global cross validation scores...')
        gs.message(os.linesep)
        gs.message(('Metric \t Mean \t Error'))    
        for name, func in scoring.items():
            gs.message(name + 
                       '\t' + str(func(y, y_pred).mean().round(3)) + 
                       '\t' + str(func(y, y_pred).std().round(3)))

        if mode == 'classification':    
            gs.message(os.linesep)
            gs.message('Cross validation class performance measures......:')
            gs.message(
                classification_report(y_true=y, y_pred=y_pred, sample_weight=class_weights))

#        # write cross-validation results for csv file
#        if errors_file != '':
#            errors = pd.DataFrame(scores)
#            errors.to_csv(errors_file, mode='w')
#
#        # write cross-validation predictions to csv file
#        if preds_file != '':
#            preds = pd.DataFrame(preds)
#            preds.columns = ['y_true', 'y_pred', 'fold', 'x', 'y']
#            preds.to_csv(preds_file, mode='w')
#            text_file = open(preds_file + 't', "w")
#            text_file.write('"Integer","Real","Real","integer","Real","Real"')
#            text_file.close()
#
#        # feature importances
#        if importances is True:
#            gs.message(os.linesep)
#            gs.message("Feature importances")
#            gs.message("id" + "\t" + "Raster" + "\t" + "Importance")
#
#            # mean of cross-validation feature importances
#            for i in range(len(fimp.mean(axis=0))):
#                gs.message(
#                    str(i) + "\t" + maplist[i] +
#                    "\t" + str(round(fimp.mean(axis=0)[i], 4)))
#
#            if fimp_file != '':
#                np.savetxt(fname=fimp_file, X=fimp, delimiter=',',
#                           header=','.join(maplist), comments='')

    # Save the fitted model
    from sklearn.externals import joblib
    joblib.dump((X, y, sample_coords, group_id, estimator), model_save)


if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    main()
