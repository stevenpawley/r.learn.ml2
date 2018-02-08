from __future__ import absolute_import, print_function
import numpy as np
import grass.script as gs
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster import numpy2raster
from grass.pygrass.vector import VectorTopo
from grass.pygrass.utils import get_raster_for_points, pixel2coor

def applier(rs, func, rowchunk=25, n_jobs=-1, **kwargs):
    """Applies a function to a RasterStack object in image stripes"""
    from sklearn.externals.joblib import Parallel, delayed

    # generator for row increments, tuple (startrow, endrow)
    reg = Region()
    windows = ((row, row+rowchunk) if row+rowchunk <= reg.rows else
               (row, reg.rows) for row in range(0, reg.rows, rowchunk))
    result = []

    # Loop through rasters strip-by-strip
    for start, end, in windows:
        #print_progressbar(start, rows, length=50)
        img = rs.read(window=(start, end))
        result.append(func(img, **kwargs))
    
    # perform predictions on lists of row increments in parallel
#    result = Parallel(n_jobs=n_jobs, max_nbytes=None)(
#        delayed(__predict_parallel2)
#        (estimator, predictors, predict_type, current, row_min, row_max)
#        for row_min, row_max in zip(row_mins, row_maxs))
#    prediction = np.vstack(prediction)

    result = np.concatenate(result, axis=1)
    return result


class RasterStack(object):
    """Access a group of aligned GDAL-supported raster images as a single
    dataset

    Attibutes
    ---------
    names : list (str)
        List of names of GRASS GIS rasters in the RasterStack
    fullnames : list (str)
        List of names of GRASS GIS rasters in the RasterStack including
        mapset names if supplied
    layernames : Dict
        Dict of key, value pairs containing the GRASS raster name and the
        matching layername
    mtypes : Dict
        Dict of key, value pairs containing the GRASS raster name and the
        data type of the raster
    count : int
        Number of raster layers in the RasterStack class
    """

    def __init__(self, rasters):
        """Create a RasterStack object

        Parameters
        ----------
        rasters : list (str)
            List of names of GRASS rasters to create a RasterStack object.
        """

        if isinstance(rasters, str):
            rasters = [rasters]
        
        self.names = []
        self.fullnames = []
        self.layernames = {}
        self.mtypes = {}
        self.count = len(rasters)
        self.cell_nodata = -2147483648

        for r in rasters:
            src = RasterRow(r)
            if src.exist() is True:
                self.fullnames.append('@'.join([src.name, src.mapset]))
                self.names.append(src.name)
                self.mtypes.update({src.name: src.mtype})
                
                validname = src.name.replace('.', '_')
                self.layernames.update({src.name: validname})
                setattr(self, validname, src)
            else:
                gs.fatal('GRASS raster map ' + r + ' does not exist')

    def read(self, row=None, window=None):
        """Read data from RasterStack as a 3D numpy array
        
        Parameters
        ----------
        row : int
            Integer representing the index of a single row of a raster to read
        window : tuple, optional
            Tuple of integers representing the start and end numbers of rows to
            read as a single block of rows
            
        Returns
        -------
        data : 3D array-like
            3D masked numpy array containing data from RasterStack rasters"""
        
        reg = Region()

        # create numpy array for stack
        if window:
            row_start = window[0]
            row_stop = window[1]
            width = reg.cols
            height = abs((row_stop+1)-row_start)
            shape = (self.count, height, width)
        else:
            row_start = row
            row_stop = row+1
            height = 1
            shape = (self.count, height, reg.cols)
        data = np.ma.zeros(shape)

        # read from each raster
        for band, raster in enumerate(self.layernames.iteritems()):
            # read into numpy array
            k,v = raster
            src = getattr(self, v)
            src.open()
            rowincrs = (row for row in range(row_start, row_stop))
            for i, row in enumerate(rowincrs):
                data[band, i, :] = src[row]
            
            # mask array with nodata
            if src.mtype == 'CELL':
                data = np.ma.mask_equal(data, self.cell_nodata)
            elif src.mtype in ['FCELL', 'DCELL']:
                data = np.ma.masked_invalid(data)
            src.close()

        return data

    @staticmethod
    def predfun(img, **kwargs):
        estimator = kwargs['estimator']
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))

        # create mask for NaN values and replace with number
        flat_pixels_mask = flat_pixels.mask.copy()

        # prediction
        result_cla = estimator.predict(flat_pixels)

        # replace mask and fill masked values with nodata value
        result_cla = np.ma.masked_array(
            result_cla, mask=flat_pixels_mask.any(axis=1))
        result_cla = np.ma.filled(result_cla, fill_value=-99999)

        # reshape the prediction from a 1D matrix/list
        # back into the original format [band, row, col]
        result_cla = result_cla.reshape((1, rows, cols))

        return result_cla

    @staticmethod
    def probfun(img, **kwargs):
        estimator = kwargs['estimator']
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows,cols,bands(transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape(
            (n_samples, n_features))

        # create mask for NaN values and replace with number
        flat_pixels_mask = flat_pixels.mask.copy()

        # predict probabilities
        result_proba = estimator.predict_proba(flat_pixels)

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result_proba = result_proba.reshape(
            (rows, cols, result_proba.shape[1]))
        flat_pixels_mask = flat_pixels_mask.reshape((rows, cols, n_features))

        # flatten mask into 2d
        mask2d = flat_pixels_mask.any(axis=2)
        mask2d = np.where(mask2d != mask2d.min(), True, False)
        mask2d = np.repeat(mask2d[:, :, np.newaxis],
                           result_proba.shape[2], axis=2)

        # convert proba to masked array using mask2d
        result_proba = np.ma.masked_array(
            result_proba,
            mask=mask2d,
            fill_value=np.nan)
        result_proba = np.ma.filled(
            result_proba, fill_value=-99999)

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        return result_proba

    def predict(self, estimator, output=None, rowchunk=25, overwrite=False):
        """Prediction method for RasterStack class
        
        Parameters
        ----------
        estimator : Scikit-learn compatible estimator
            Previously fitted estimator
        output : str
            Output name for prediction raster
        rowchunk : int
            Number of raster rows to pass to estimator at one time
        overwrite : bool
            Option to overwrite an existing raster
        """

        result = applier(rs=self, func=self.predfun, output=output,
                         rowchunk=rowchunk, **{'estimator': estimator})

        # determine raster dtype
        if result.dtype == 'float':
            mtype = 'FCELL'
        else:
            mtype = 'CELL'

        numpy2raster(array=result[0, :, :], mtype=mtype, rastname=output,
                     overwrite=overwrite)

        return None

    def predict_proba(self, estimator, output=None, class_labels=None,
                      index=None, rowchunk=25, overwrite=False):
        """Prediction method for RasterStack class
        
        Parameters
        ----------
        estimator : Scikit-learn compatible estimator
            Previously fitted estimator
        output : str
            Output name for prediction raster
        class_labels : 1d array-like, optional
            class labels
        index : list, optional
            list of class indices to export
        rowchunk : int
            Number of raster rows to pass to estimator at one time
        overwrite : bool
            Option to overwrite an existing raster(s)
        """
        if isinstance(class_labels, int):
            class_labels = [class_labels]

        result = applier(rs=self, func=self.probfun, output=output,
                         rowchunk=rowchunk, **{'estimator': estimator})
        
        # use class labels if supplied
        # else output predictions as 0,1,2...n
        if class_labels is None:
            class_labels = range(result.shape[0])

        # output all class probabilities if subset is not specified
        if index is None:
            index = class_labels

        # select indexes of predictions 3d numpy array to be exported to rasters
        selected_prediction_indexes = [i for i, x in enumerate(class_labels) if x in index]

        # write each 3d of numpy array as a probability raster
        for pred_index, label in zip(selected_prediction_indexes, index):
            rastername = output + '_' + str(label)
            numpy2raster(array=result[pred_index, :, :], mtype='FCELL',
                         rastname=rastername, overwrite=overwrite)

        return None

    @staticmethod
    def __value_extractor(img, **kwargs):
        # split numpy array bands(axis=0) into labelled pixels and
        # raster data
        response_arr = img[-1, :, :]
        raster_arr = img[0:-1, :, :]

        # returns indices of labelled values
        is_train = np.nonzero(~response_arr.mask)

        # get the labelled values
        labels = response_arr.data[is_train]

        # extract data at labelled pixel locations
        data = raster_arr[:, is_train[0], is_train[1]]

        # Remove nan rows from training data
        data = data.filled(np.nan)

        # combine training data, locations and labels
        data = np.vstack((data, labels))

        return data

    def extract_pixels(self, response, na_rm=True):

        if RasterRow(response).exist() is False:
            gs.fatal('GRASS raster ' + response ' does not exist')

        # create new RasterStack object with labelled pixels as
        # last band in the stack
        temp_stack = RasterStack(self.files + response)

        # extract training data
        data = applier(temp_stack, self.__value_extractor)
        data = np.concatenate(data, axis=1)
        raster_vals = data[0:-1, :]
        labelled_vals = data[-1, :]

        if na_rm is True:
            X = raster_vals[~np.isnan(raster_vals).any(axis=1)]
            y = labelled_vals[np.where(~np.isnan(raster_vals).any(axis=1))]
        else:
            X = raster_vals
            y = labelled_vals

        return (X, y)

    def extract_features(self, gvector, field, na_rm=False):
        """Samples a list of GDAL rasters using a point data set

        Parameters
        ----------
        y : str
            Name of GRASS GIS vector containing point features
        field : str
            Name of attribute containing the response variable

        Returns
        -------
        gdf : Geopandas GeoDataFrame
            GeoDataFrame containing extract raster values at the point
            locations
        """

        # open grass vector
        points = VectorTopo(gvector.split('@')[0])
        points.open('r')
    
        # create link to attribute table
        points.dblinks.by_name(name=gvector)
    
        # extract table field to numpy array
        table = points.table
        cur = table.execute("SELECT {field} FROM {name}".format(field=field, name=table.name))
        y = np.array([np.isnan if c is None else c[0] for c in cur])
        y = np.array(y, dtype='float')
    
        # extract raster data
        X = np.zeros((points.num_primitives()['point'], len(self.fullnames)), dtype=float)
        for i, raster in enumerate(self.fullnames):
            rio = RasterRow(raster)
            if rio.exist() is False:
                gs.fatal('Raster {x} does not exist....'.format(x=raster))
            values = np.asarray(get_raster_for_points(points, rio))
            coordinates = values[:, 1:3]
            X[:, i] = values[:, 3]
            rio.close()
    
        # set any grass integer nodata values to NaN
        X[X == self.cell_nodata] = np.nan
    
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
            if np.isnan(X).any() == True:
                gs.message('Removing samples with NaN values in the raster feature variables...')
    
            y = y[~np.isnan(X).any(axis=1)]
            coordinates = coordinates[~np.isnan(X).any(axis=1)]
            X = X[~np.isnan(X).any(axis=1)]
    
        return(X, y, coordinates)


def extract_pixels(response, predictors, lowmem=False, na_rm=False):
    """

    Samples a list of GRASS rasters using a labelled raster
    Per raster sampling

    Args
    ----
    response (string): Name of GRASS raster with labelled pixels
    predictors (list): List of GRASS raster names containing explanatory variables
    lowmem (boolean): Use numpy memmap to query predictors
    na_rm (boolean): Remove samples containing NaNs

    Returns
    -------
    training_data (2d numpy array): Extracted raster values
    training_labels (1d numpy array): Numpy array of labels
    is_train (2d numpy array): Row and Columns of label positions

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
        if np.isnan(training_data).any() == True:
            gs.message('Removing samples with NaN values in the raster feature variables...')
        training_labels = training_labels[~np.isnan(training_data).any(axis=1)]
        is_train = is_train[~np.isnan(training_data).any(axis=1)]
        training_data = training_data[~np.isnan(training_data).any(axis=1)]

    return(training_data, training_labels, is_train)


def extract_points(gvector, grasters, field, na_rm=False):
    """

    Extract values from grass rasters using vector points input

    Args
    ----
    gvector (string): Name of grass points vector
    grasters (list): Names of grass raster to query
    field (string): Name of field in table to use as response variable
    na_rm (boolean): Remove samples containing NaNs

    Returns
    -------
    X (2d numpy array): Training data
    y (1d numpy array): Array with the response variable
    coordinates (2d numpy array): Sample coordinates

    """

    # open grass vector
    points = VectorTopo(gvector.split('@')[0])
    points.open('r')

    # create link to attribute table
    points.dblinks.by_name(name=gvector)

    # extract table field to numpy array
    table = points.table
    cur = table.execute("SELECT {field} FROM {name}".format(field=field, name=table.name))
    y = np.array([np.isnan if c is None else c[0] for c in cur])
    y = np.array(y, dtype='float')

    # extract raster data
    X = np.zeros((points.num_primitives()['point'], len(grasters)), dtype=float)
    for i, raster in enumerate(grasters):
        rio = RasterRow(raster)
        if rio.exist() is False:
            gs.fatal('Raster {x} does not exist....'.format(x=raster))
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
        if np.isnan(X).any() == True:
            gs.message('Removing samples with NaN values in the raster feature variables...')

        y = y[~np.isnan(X).any(axis=1)]
        coordinates = coordinates[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(X).any(axis=1)]

    return(X, y, coordinates)