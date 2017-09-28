#!/usr/bin/env python
# -- coding: utf-8 --

"""
The module rlearn_rasters contains functions to
extract training data from GRASS rasters.
"""

from __future__ import absolute_import
import numpy as np
import tempfile
import grass.script as gs
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster import numpy2raster
from grass.pygrass.vector import VectorTopo
from grass.pygrass.utils import get_raster_for_points, pixel2coor


def extract_pixels(response, predictors, lowmem=False, na_rm=False):
    """
    Samples a list of GRASS rasters using a labelled raster
    Per raster sampling

    Args
    ----
    response: String; GRASS raster with labelled pixels
    predictors: List of GRASS rasters containing explanatory variables
    lowmem: Boolean, use numpy memmap to query predictors
    na_rm: Boolean, remove samples containing NaNs

    Returns
    -------
    training_data: Numpy array of extracted raster values
    training_labels: Numpy array of labels
    is_train: Row and Columns of label positions
    """

    current = Region()

    # open response raster as rasterrow and read as np array
    if RasterRow(response).exist() is True:
        roi_gr = RasterRow(response)
        roi_gr.open('r')

        if lowmem is False:
            response_np = np.array(roi_gr)
        else:
            response_np = np.memmap(
                tempfile.NamedTemporaryFile(),
                dtype='float32', mode='w+',
                shape=(current.rows, current.cols))
            response_np[:] = np.array(roi_gr)[:]
    else:
        gs.fatal("GRASS response raster does not exist.... exiting")

    # determine number of predictor rasters
    n_features = len(predictors)

    # check to see if all predictors exist
    for i in range(n_features):
        if RasterRow(predictors[i]).exist() is not True:
            gs.fatal("GRASS raster " + predictors[i] +
                          " does not exist.... exiting")

    # check if any of those pixels are labelled (not equal to nodata)
    # can use even if roi is FCELL because nodata will be nan
    is_train = np.nonzero(response_np > -2147483648)
    training_labels = response_np[is_train]
    n_labels = np.array(is_train).shape[1]

    # Create a zero numpy array of len training labels
    if lowmem is False:
        training_data = np.zeros((n_labels, n_features))
    else:
        training_data = np.memmap(tempfile.NamedTemporaryFile(),
                                  dtype='float32', mode='w+',
                                  shape=(n_labels, n_features))

    # Loop through each raster and sample pixel values at training indexes
    if lowmem is True:
        feature_np = np.memmap(tempfile.NamedTemporaryFile(),
                               dtype='float32', mode='w+',
                               shape=(current.rows, current.cols))

    for f in range(n_features):
        predictor_gr = RasterRow(predictors[f])
        predictor_gr.open('r')

        if lowmem is False:
            feature_np = np.array(predictor_gr)
        else:
            feature_np[:] = np.array(predictor_gr)[:]

        training_data[0:n_labels, f] = feature_np[is_train]

        # close each predictor map
        predictor_gr.close()

    # convert any CELL maps no datavals to NaN in the training data
    for i in range(n_features):
        training_data[training_data[:, i] == -2147483648] = np.nan

    # convert indexes of training pixels from tuple to n*2 np array
    is_train = np.array(is_train).T
    for i in range(is_train.shape[0]):
        is_train[i, :] = np.array(pixel2coor(tuple(is_train[i]), current))

    # close the response map
    roi_gr.close()

    # remove samples containing NaNs
    if na_rm is True:
        training_labels = training_labels[~np.isnan(training_data).any(axis=1)]
        is_train = is_train[~np.isnan(training_data).any(axis=1)]
        training_data = training_data[~np.isnan(training_data).any(axis=1)]

    return(training_data, training_labels, is_train)


def extract_points(gvector, grasters, field, na_rm=False):
    """
    Extract values from grass rasters using vector points input

    Args
    ----
    gvector: character, name of grass points vector
    grasters: list of names of grass raster to query
    field: character, name of field in table to use as response variable
    na_rm: Boolean, remove samples containing NaNs

    Returns
    -------
    X: 2D numpy array of training data
    y: 1D numpy array with the response variable
    coordinates: 2D numpy array of sample coordinates
    """
    # open grass vector
    points = VectorTopo(gvector.split('@')[0])
    points.open('r')

    # create link to attribute table
    points.dblinks.by_name(name=gvector)

    # extract table field to numpy array
    table = points.table
    cur = table.execute("SELECT {field} FROM {name}".format(field=field, name=table.name))
    #y = np.array([np.isnan if c is None else c[0] for c in cur])
    y = [c[0] for c in cur]
    y = np.array(y, dtype='float')

    # extract raster data
    X = np.zeros((points.num_primitives()['point'], len(grasters)), dtype=float)
    for i, raster in enumerate(grasters):
        rio = RasterRow(raster)
        values = np.asarray(get_raster_for_points(points, rio))
        coordinates = values[:, 1:3]
        X[:, i] = values[:, 3]
        rio.close()

    # set any grass integer nodata values to NaN
    X[X == -2147483648] = np.nan

    # remove missing response data
    X = X[~np.isnan(y)]
    coordinates = coordinates[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # int type if classes represented integers
    if all(y % 1 == 0) is True:
        y = np.asarray(y, dtype='int')

    # close
    points.close()

    # remove samples containing NaNs
    if na_rm is True:
        y = y[~np.isnan(X).any(axis=1)]
        coordinates = coordinates[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(X).any(axis=1)]

    return(X, y, coordinates)


def predict(estimator, predictors, output, predict_type='raw', index=None,
            class_labels=None, n_jobs=-2, overwrite=False):
    """
    Prediction on list of GRASS rasters using a fitted scikit learn model

    Args
    ----
    estimator: scikit-learn estimator object
    predictors: list of GRASS rasters
    output: Name of GRASS raster to output classification results
    predict_type: character, 'raw' for classification/regression;
                  'prob' for class probabilities
    index: Optional, list of class indices to export
    class_labels: Optional, class labels
    rowincr: Integer of raster rows to process at one time
    n_jobs: Number of processing cores
    """

    from sklearn.externals.joblib import Parallel, delayed
    #from sklearn.externals.joblib.pool import has_shareable_memory

    # convert potential single index to list
    if isinstance(index, int): index = [index]

    # open predictors as list of rasterrow objects
    current = Region()

    # perform predictions on lists of rows in parallel
    prediction = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(__predict_parallel)
        (estimator, predictors, predict_type, current, row)
        for row in range(current.rows))
    prediction = np.asarray(prediction)

    # determine raster dtype
    if prediction.dtype == 'float':
        ftype = 'FCELL'
    else:
        ftype = 'CELL'

    #  writing of predicted results for classification
    if predict_type == 'raw':
        numpy2raster(array=prediction, mtype=ftype, rastname=output,
                     overwrite=overwrite)

    # writing of predicted results for probabilities
    if predict_type == 'prob':

        # use class labels if supplied
        # else output predictions as 0,1,2...n
        if class_labels is None:
            class_labels = range(prediction.shape[2])

        # output all class probabilities if subset is not specified
        if index is None:
            index = class_labels

        # select indexes of predictions 3d numpy array to be exported to rasters
        selected_prediction_indexes = [i for i, x in enumerate(class_labels) if x in index]

        # write each 3d of numpy array as a probability raster
        for pred_index, label in zip(selected_prediction_indexes, index):
            rastername = output + '_' + str(label)
            numpy2raster(array=prediction[:, :, pred_index], mtype='FCELL',
                         rastname=rastername, overwrite=overwrite)


def __predict_parallel(estimator, predictors, predict_type, current, row):
    """
    Performs prediction on range of rows in grass rasters

    Args
    ----
    estimator: scikit-learn estimator object
    predictors: list of GRASS rasters
    predict_type: character, 'raw' for classification/regression;
                  'prob' for class probabilities
    current: current region settings
    row: Rasterrow to perform prediction on

    Returns
    -------
    result: 2D (classification) or 3D numpy array (class probabilities) of predictions
    ftypes: data storage type
    """

    # initialize output
    result, mask = None, None

    # open grass rasters
    n_features = len(predictors)
    rasstack = [0] * n_features

    for i in range(n_features):
        rasstack[i] = RasterRow(predictors[i])
        if rasstack[i].exist() is True:
            rasstack[i].open('r')
        else:
            gs.fatal("GRASS raster " + predictors[i] +
                          " does not exist.... exiting")

    # loop through each row, and each band and add to 2D img_np_row
    img_np_row = np.zeros((current.cols, n_features))
    for band in range(n_features):
        img_np_row[:, band] = np.array(rasstack[band][row])

    # create mask
    img_np_row[img_np_row == -2147483648] = np.nan
    mask = np.zeros((img_np_row.shape[0]))
    for feature in range(n_features):
        invalid_indexes = np.nonzero(np.isnan(img_np_row[:, feature]))
        mask[invalid_indexes] = np.nan

    # reshape each row-band matrix into a n*m array
    nsamples = current.cols
    flat_pixels = img_np_row.reshape((nsamples, n_features))

    # remove NaNs prior to passing to scikit-learn predict
    flat_pixels = np.nan_to_num(flat_pixels)

    # perform prediction for classification/regression
    if predict_type == 'raw':
        result = estimator.predict(flat_pixels)
        result = result.reshape((current.cols))

        # determine nodata value and grass raster type
        if result.dtype == 'float':
            nodata = np.nan
        else:
            nodata = -2147483648

        # replace NaN values so that the prediction does not have a border
        result[np.nonzero(np.isnan(mask))] = nodata

    # perform prediction for class probabilities
    if predict_type == 'prob':
        result = estimator.predict_proba(flat_pixels)
        result = result.reshape((current.cols, result.shape[1]))
        result[np.nonzero(np.isnan(mask))] = np.nan

    # close maps
    for i in range(n_features):
        rasstack[i].close()

    return (result)
