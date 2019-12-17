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

#%option G_OPT_R_INPUT
#% key: group_raster
#% label: Custom group ids for training samples from GRASS raster
#% description: GRASS raster containing group ids for training samples. Samples with the same group id will not be split between training and test cross-validation folds
#% required: no
#% guisection: Cross validation
#%end

#%option G_OPT_F_OUTPUT
#% key: save_training
#% label: Save training data to csv
#% required: no
#% guisection: Optional
#%end

#%option
#% key: random_state
#% type: integer
#% description: Seed to use for random state
#% answer: 1
#% guisection: Optional
#%end

#%rules
#% required: training_map,training_points
#% exclusive: training_map,training_points
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
    gs.fatal('Not able to find the r.learn.ml library directory')
sys.path.append(path)

from utils import save_training_data, option_to_list
from raster import RasterStack


def main():
    try:
        import sklearn

        if sklearn.__version__ < '0.22':
            gs.fatal("Scikit learn 0.22 or newer is required")
        
    except ImportError:
        gs.fatal("Scikit learn 0.22 or newer is not installed")

    try:
        import pandas as pd
        
    except ImportError:
        gs.fatal("Pandas is not installed ")

    # parser options
    group = options['group']
    training_map = options['training_map']
    training_points = options['training_points']
    field = options['field']
    category_maps = option_to_list(options['category_maps'])
    group_raster = options['group_raster']
    random_state = int(options['random_state'])
    save_training = options['save_training']

    # checks of input options
    if training_points != '' and field == '':
        gs.fatal('No attribute column specified for training points')
    
    # define RasterStack
    stack = RasterStack(group=group)
    
    if category_maps is not None:
        stack.categorical = category_maps

    # extract training data
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

    save_training_data(X, y, cat, group_id, save_training)
