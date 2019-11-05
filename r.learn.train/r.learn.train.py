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
#% options: LogisticRegression,LinearRegression,LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis,KNeighborsClassifier,KNeighborsRegressor,GaussianNB,DecisionTreeClassifier,DecisionTreeRegressor,RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingClassifier,GradientBoostingRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor,SVC,SVR
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
#% key: epsilon
#% type: double
#% label: Epsilon in the SVR model
#% description: Epsilon in the SVR model
#% answer: 0.1
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
#% answer: holdout
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
#% label: Compute Feature importances
#% description: Compute feature importances using permutation (requires ELI5 package)
#% guisection: Estimator settings
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
#% answer: 1
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
from copy import deepcopy

import grass.script as gs
import numpy as np
from grass.script.utils import get_lib_path

path = get_lib_path(modname='r.learn.ml')

if path is None:
    gs.fatal('Not able to find the r.learn library directory')
sys.path.append(path)

from utils import (model_classifiers, load_training_data, save_training_data,
                   option_to_list, scoring_metrics, expand_feature_names)
from raster import RasterStack


tmp_rast = []

def cleanup():
    for rast in tmp_rast:
        gs.run_command("g.remove", name=rast, type='raster', flags='f',
                       quiet=True)


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def wrap_named_step(param_grid):
    translate = {}
    
    for k, v in param_grid.items():
        newkey = 'estimator__' + k
        translate[k] = newkey
    
    for old, new in translate.items():
        param_grid[new] = param_grid.pop(old)
    
    return param_grid


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

    # parser options
    group = options['group']
    training_map = options['training_map']
    training_points = options['training_points']
    field = options['field']
    model_save = options['save_model']

    model_name = options['model_name']
    grid_search = options['grid_search']
    hyperparams = {
        'C': options['c'],
        'epsilon': options['epsilon'],
        'min_samples_split': options['min_samples_split'],
        'min_samples_leaf': options['min_samples_leaf'],
        'n_estimators': options['n_estimators'],
        'learning_rate': options['learning_rate'],
        'subsample': options['subsample'],
        'max_depth': options['max_depth'],
        'max_features': options['max_features'],
        'n_neighbors': options['n_neighbors'],
        'weights': options['weights']
        }

    cv = int(options['cv'])
    group_raster = options['group_raster']
    tune_only = flags['t']
    importances = flags['f']
    preds_file = options['preds_file']
    classif_file = options['classif_file']
    fimp_file = options['fimp_file']
    param_file = options['param_file']

    norm_data = flags['s']
    category_maps = option_to_list(options['category_maps'])
    random_state = int(options['random_state'])
    load_training = options['load_training']
    save_training = options['save_training']
    n_jobs = int(options['n_jobs'])
    balance = flags['b']

    # make dicts for hyperparameters, datatypes and parameters for tuning
    hyperparams_type = dict.fromkeys(hyperparams, int)
    hyperparams_type['C'] = float
    hyperparams_type['epsilon'] = float
    hyperparams_type['learning_rate'] = float
    hyperparams_type['subsample'] = float
    hyperparams_type['weights'] = str
    param_grid = deepcopy(hyperparams_type)
    param_grid = dict.fromkeys(param_grid, None)

    for key, val in hyperparams.items():
        if ',' in val:
            param_grid[key] =  \
                [hyperparams_type[key](i) for i in val.split(',')]
            
            hyperparams[key] = \
                [hyperparams_type[key](i) for i in val.split(',')][0]
        else:
            hyperparams[key] = hyperparams_type[key](val)

    if hyperparams['max_depth'] == 0: hyperparams['max_depth'] = None
    if hyperparams['max_features'] == 0: hyperparams['max_features'] = 'auto'
    param_grid = {k: v for k, v in param_grid.items() if v is not None}

    estimator, mode = model_classifiers(
        model_name, random_state, n_jobs, hyperparams, balance)

    # remove dict keys that are incompatible for the selected estimator
    estimator_params = estimator.get_params()
    param_grid = {
        key: value for key, value in param_grid.items()
        if key in estimator_params
        }
    scoring, search_scorer = scoring_metrics(mode)

    # checks of input options
    if training_points != '' and field == '':
        gs.fatal('No attribute column specified for training points')

    if (any(param_grid) is True and cv == 1 and grid_search == 'cross-validation'):
        gs.fatal(
            'Hyperparameter search using cross validation requires cv > 1')
    
    if model_name == 'HistGradientBoostingClassifier' and balance is True:
        gs.warning('HistGradientBoostingClassifier does not accept class',
                   'weights. Use GradientBoostingClassifier if you want to',
                   'rebalance your classes using class weights')
        balance = False

    # define RasterStack
    maplist = (gs.read_command("i.group", group=group, flags="g").
               split(os.linesep)[:-1])
    
    stack = RasterStack(rasters=maplist)
    
    if category_maps is not None:
        stack.categorical = category_maps

    # extract training data
    if load_training != '':
        X, y, cat, group_id = load_training_data(load_training)
    else:
        gs.message('Extracting training data')

        if group_raster != '':
            stack.append(group_raster)
            
        if training_map != '':
            X, y, cat = stack.extract_pixels(training_map)
        elif training_points != '':
            X, y, cat = stack.extract_points(training_points, field)
                
        y = y.flatten()  # reshape to 1 dimension
        cat = cat.flatten()
        
        # label encoding
        if y.dtype in [np.object_, np.object]:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

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

        from sklearn.utils import shuffle
        
        if group_id is None:
            X, y, cat = shuffle(X, y, cat, random_state=random_state)
        else:
            X, y, cat, group_id = shuffle(X, y, cat, group_id,
                                          random_state=random_state)

        if save_training != '':
            save_training_data(X, y, cat, group_id, save_training)

    # define the inner search resampling method
    from sklearn.model_selection import (
        GridSearchCV, StratifiedKFold, GroupKFold, KFold, ShuffleSplit,
        GroupShuffleSplit)

    if any(param_grid) is True and grid_search == 'cross-validation':
        if group_id is None and mode == 'classification':
            inner = StratifiedKFold(n_splits=cv, random_state=random_state)
        elif group_id is None and mode == 'regression':
            inner = KFold(n_splits=cv, random_state=random_state)
        else:
            inner = GroupKFold(n_splits=cv)

    elif any(param_grid) is True and grid_search == 'holdout':
        if group_id is None:
            inner = ShuffleSplit(
                n_splits=1, test_size=0.33, random_state=random_state)
        else:
            inner = GroupShuffleSplit(
                n_splits=1, test_size=0.33, random_state=random_state)
    else:
        inner = None

    # define the outer search resampling method
    if cv > 1:
        if group_id is None and mode == 'classification':
            outer = StratifiedKFold(n_splits=cv, random_state=random_state)
        elif group_id is None and mode == 'regression':
            outer = KFold(n_splits=cv, random_state=random_state)
        else:
            outer = GroupKFold(n_splits=cv)

    # estimators that take sample_weights
    if balance is True and mode == 'classification' and model_name in (
            'GradientBoostingClassifier', 'GaussianNB'):
        from sklearn.utils import compute_class_weight

        class_weights = compute_class_weight(
            class_weight='balanced', classes=(y), y=y)
        fit_params = {'sample_weight': class_weights}
    else:
        class_weights = None
        fit_params = {}

    # wrapped permutation importance estimator
    if importances is True:
        try:
            from eli5.sklearn import PermutationImportance
            
            estimator = PermutationImportance(
                estimator=estimator,
                scoring=search_scorer,
                n_iter=5,
                random_state=random_state,
                cv=3)
            
            param_grid = wrap_named_step(param_grid)
            fit_params = wrap_named_step(fit_params)
                        
        except ImportError:
            gs.warning('Permutation feature importances require the ELI5',
                       'python package to be installed')

    # define the preprocessing pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # standardization only
    if norm_data is True and category_maps is None:
        scaler = StandardScaler()
        trans = ColumnTransformer(
            remainder='passthrough',
            transformers=[
                ('scaling', scaler, np.arange(0, stack.count))
                ]
            )
    
    # one-hot encoding only
    elif norm_data is False and category_maps is not None:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        trans = ColumnTransformer(
            remainder='passthrough',
            transformers=[('onehot', enc, stack.categorical)])
        
    # standardization and one-hot encoding
    elif norm_data is True and category_maps is not None:
        scaler = StandardScaler()
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

        numeric_idx = np.setxor1d(
            ar1=np.arange(0, stack.count),
            ar2=stack.categorical)
                
        trans = ColumnTransformer(
            remainder='passthrough',
            transformers=[
                ('onehot', enc, stack.categorical),
                ('scaling', scaler, numeric_idx)
                ]
            )

    # combine transformers
    if norm_data is True or category_maps is not None:    
        estimator = Pipeline(
            [('preprocessing', trans),
             ('estimator', estimator)])
        
        param_grid = wrap_named_step(param_grid)
        fit_params = wrap_named_step(fit_params)
    
    if any(param_grid) is True:
        estimator = GridSearchCV(
            estimator=estimator, param_grid=param_grid,
            scoring=search_scorer, n_jobs=n_jobs, cv=inner)

    # estimator training
    gs.message(os.linesep)
    gs.message(('Fitting model using ' + model_name))
        
    try:
        estimator.fit(X, y, groups=group_id, **fit_params)
    except:
        if model_name != 'HistGradientBoostingClassifier':
            estimator.fit(X, y, **fit_params)
        else:
            estimator.fit(X, y)
    
    # message best hyperparameter setup and optionally save using pandas
    if any(param_grid) is True:
        gs.message(os.linesep)
        gs.message('Best parameters:')
        gs.message(str(estimator.best_params_))
        
        if param_file != '':
            param_df = pd.DataFrame(estimator.cv_results_)
            param_df.to_csv(param_file)

    # cross-validation
    if cv > 1 and tune_only is not True:
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
        
        from sklearn.model_selection import cross_val_predict
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
                sample_weight=class_weights,
                output_dict=False)
            
            report = classification_report(
                y_true=preds['y_true'], 
                y_pred=preds['y_pred'], 
                sample_weight=class_weights,
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

    if importances is True:
        # simple model with feature importances
        try:
            fimp = estimator.feature_importances_
        except AttributeError:
            pass

        # model with gridsearch and feature importances
        try:
            fimp = estimator.best_estimator_.feature_importances_
        except AttributeError:
            pass

        # model with transformers and feature importances
        try:
            fimp = estimator.named_steps['estimator'].feature_importances_
        except AttributeError:
            pass

        # model with gridsearch-transformers-feature importances
        try:
            fimp = (estimator.
                    best_estimator_.
                    named_steps['estimator'].
                    feature_importances_)
        except AttributeError:
            pass
        
        feature_names = deepcopy(stack.names)
        feature_names = [i.split('@')[0] for i in feature_names]

        if category_maps is not None:
            try:
                enc = (estimator.
                       named_steps['preprocessing'].
                       named_transformers_['onehot']
                      )
            except AttributeError:
                pass
        
            try:
                enc = (estimator.
                       best_estimator_.
                       named_steps['preprocessing'].
                       named_transformers_['onehot']
                       )
            except AttributeError:
                pass

            try:
                enc = (estimator.
                       best_estimator_.
                       estimator_.named_steps['preprocessing'].
                       named_transformers_['onehot'])
            except AttributeError:
                pass
        
            feature_names = expand_feature_names(
                feature_names=feature_names,
                categorical_indices=stack.categorical,
                enc_categories=enc.categories_)
        
        fimp = pd.DataFrame({'Feature': feature_names, 'Importances': fimp})
        gs.message(os.linesep)
        gs.message('Feature importances')
        gs.message('Feature' + '\t' + 'Score')
        
        for index, row in fimp.iterrows():
            gs.message(row['Feature'] + '\t' + str(row['Importances']))

        if fimp_file != '':
            fimp.to_csv(fimp_file, index=False)

    # save the fitted model
    import joblib
    joblib.dump((estimator, y), model_save)


if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    main()
