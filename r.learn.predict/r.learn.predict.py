#!/usr/bin/env python
# -*- coding: utf-8 -*-

############################################################################
# MODULE:        r.learn.predict
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
#% description: Apply a fitted scikit-learn estimator to rasters in a GRASS GIS imagery group
#% keyword: raster
#% keyword: classification
#% keyword: regression
#% keyword: machine learning
#% keyword: scikit-learn
#% keyword: prediction
#%end

#%option G_OPT_I_GROUP
#% key: group
#% label: Group of raster layers used for prediction
#% description: GRASS imagery group of raster maps representing feature variables to be used in the machine learning model
#% required: yes
#% multiple: no
#%end

#%option G_OPT_F_INPUT
#% key: load_model
#% label: Load model from file
#% description: File representing pickled scikit-learn estimator model
#% required: yes
#% guisection: Required
#%end

#%option G_OPT_R_OUTPUT
#% key: output
#% label: Output Map
#% description: Raster layer name to store result from classification or regression model. The name will also used as a perfix if class probabilities or intermediate of cross-validation results are ordered as maps.
#% guisection: Required
#% required: no
#%end

#%flag
#% key: p
#% label: Output class membership probabilities
#% description: A raster layer is created for each class. For the case of a binary classification, only the positive (maximum) class is output
#% guisection: Optional
#%end

#%flag
#% key: z
#% label: Only predict class probabilities
#% guisection: Optional
#%end

#%option
#% key: chunksize
#% type: double
#% description: Size of MB for each block of data to be read from the disk
#% answer: 500
#% guisection: Optional
#%end

from __future__ import absolute_import, print_function

import os
import sys
import grass.script as gs
import numpy as np
from grass.script.utils import get_lib_path

path = get_lib_path(modname='r.learn.ml')
if path is None:
    gs.fatal('Not able to find the r.learn library directory')
sys.path.append(path)

from raster import RasterStack


def main():
    try:
        import sklearn
        from sklearn.externals import joblib

        if sklearn.__version__ < '0.20':
            gs.fatal("Scikit learn 0.20 or newer is required")
        
    except ImportError:
        gs.fatal("Scikit learn 0.20 or newer is not installed")

    # -------------------------------------------------------------------------
    # Parser options
    # -------------------------------------------------------------------------
    
    group = options['group']
    output = options['output']
    model_load = options['load_model']
    probability = flags['p']
    prob_only = flags['z']
    chunksize = int(options['chunksize'])
    
    # remove @ from output in case overwriting result
    if '@' in output:
        output = output.split('@')[0]
        
    # check that probabilities=True if prob_only=True
    if prob_only is True and probability is False:
        gs.fatal('Need to set probabilities=True if prob_only=True')

    # -------------------------------------------------------------------------
    # Reload fitted model and trainign data
    # -------------------------------------------------------------------------
    X, y, sample_coords, group_id, estimator = joblib.load(model_load)
    
    # -------------------------------------------------------------------------
    # Define RasterStack
    # -------------------------------------------------------------------------

    # fetch individual raster names from group
    maplist = gs.read_command("i.group", group=group, flags="g").split(os.linesep)[:-1]
    
    # create RasterStack
    stack = RasterStack(rasters=maplist)

    # -------------------------------------------------------------------------
    # Perform raster prediction
    # -------------------------------------------------------------------------

    # calculate chunksize
    row = stack.read(1)
    rowsize_mg = row.nbytes * 1e-6
    row_incr = int(float(chunksize) / float(rowsize_mg))

    # prediction
    if prob_only is False:
    
        gs.message('Predicting classification/regression raster...')
    
        stack.predict(estimator=estimator, output=output, height=row_incr,
                      overwrite=gs.overwrite())

    if probability is True:
    
        gs.message('Predicting class probabilities...')
    
        stack.predict_proba(estimator=estimator, output=output,
                            class_labels=np.unique(y), overwrite=gs.overwrite(),
                            height=row_incr)


if __name__ == "__main__":
    options, flags = gs.parser()
    main()
