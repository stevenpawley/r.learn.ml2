from __future__ import absolute_import, print_function
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import grass.script as gs
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.vector import VectorTopo
from grass.pygrass.utils import get_raster_for_points, pixel2coor


def applier(rs, func, output=None, rowchunk=25, overwrite=False, **kwargs):
    """Applies a function to a RasterStack object in image strips and
    saves each strip to a GRASS raster"""

    # processing region dimensions
    reg = Region()
    reg.set_raster_region() # set region for all raster maps in session

    # generator for row increments, tuple (startrow, endrow)
    windows = ((row, row+rowchunk) if row+rowchunk <= reg.rows else
               (row, reg.rows) for row in range(0, reg.rows, rowchunk))

    # Loop through rasters strip-by-strip
    for window in windows:
        img = rs.read(window=window)
        result = func(img, **kwargs)

        # determine output types on first iteration
        if window[0] == 0:                        
            # determine raster dtype
            if result.dtype == 'float':
                mtype = 'FCELL'
                nodata = np.nan
            else:
                mtype = 'CELL'
                nodata = -2147483648
            
            # output represents a single features
            if result.shape[0] == 1:
                newrast = [RasterRow(output)]
                newrast[0].open('w', mtype=mtype, overwrite=overwrite)
            else:
                class_labels = kwargs['class_labels']
                index = kwargs['index']
                # use class labels if supplied else output preds as 0,1,2...n
                if class_labels is None:
                    class_labels = range(result.shape[0])        
                # output all class probabilities if subset is not specified
                if index is None:
                    index = class_labels
                labels = [i for i, x in enumerate(class_labels) if x in index]
                
                # create and open rasters for writing
                newrast = []
                for pred_index, label in zip(labels, index):
                    rastername = output + '_' + str(label)
                    newrast.append(RasterRow(rastername))
                    newrast[pred_index].open(
                        'w', mtype='FCELL', overwrite=overwrite)

        # writing data to GRASS raster(s))
        result = np.ma.filled(result, nodata)
        if result.shape[0] == 1:
            # write single fature to a RasterRow
            for i in range(result.shape[1]):
                newrow = Buffer((reg.cols,), mtype=mtype)
                newrow[:] = result[0, i, :]
                newrast[0].put_row(newrow)
        else:
            # write multiple features to 
            result = np.ma.filled(result, np.nan)
            for pred_index in index:
                for i in range(result.shape[1]):
                    newrow = Buffer((reg.cols,), mtype='FCELL')
                    newrow[:] = result[pred_index, i, :]
                    newrast[pred_index].put_row(newrow)

    res = [i.close() for i in newrast]

    return res

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
    categorical : list (int)
        Indices of rasters that represent categorical datatypes
    """

    def __init__(self, rasters, categorical=None):
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
        self.layernames = OrderedDict()
        self.mtypes = {}
        self.count = len(rasters)
        self.categorical = []
        self.cell_nodata = -2147483648

        # add rasters and metadata to stack
        for r in rasters:
            src = RasterRow(r)
            if src.exist() is True:
                ras_name = src.name.split('@')[0] # name of map sans mapset
                full_name = '@'.join([ras_name, src.mapset]) # name of map with mapset
                self.fullnames.append(full_name)
                self.names.append(ras_name)
                self.mtypes.update({full_name: src.mtype})
                
                validname = ras_name.replace('.', '_')
                self.layernames.update({full_name: validname})
                setattr(self, validname, src)
            else:
                gs.fatal('GRASS raster map ' + r + ' does not exist')
        
        # extract indices of category maps
        if categorical and categorical.strip() != '':  # needed for passing from grass parser
            if isinstance(categorical, str):
                categorical = [categorical]
            self.categorical = categorical


    def read(self, row=None, window=None):
        """Read data from RasterStack as a masked 3D numpy array
        
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
            height = abs(row_stop-row_start)
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
                data = np.ma.masked_equal(data, self.cell_nodata)
            elif src.mtype in ['FCELL', 'DCELL']:
                data = np.ma.masked_invalid(data)
            src.close()

        return data

    @staticmethod
    def __predfun(img, **kwargs):
        """Prediction function for RasterStack class
        Intended to be used internally
        
        Parameters
        ----------
        img : 3d array-like
            3d masked numpy array for image block to pass to estimator class
        estimator : Scikit-learn compatible estimator class
            Scikit-learn estimator that includes a fit(X,y) method
        
        Returns
        -------
        result : 2d array-like
            masked 2d numpy array containing classification result. Array order is in
            (row, col).
        """
        
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
        flat_pixels = np.ma.filled(flat_pixels, -99999)

        # prediction
        result = estimator.predict(flat_pixels)

        # replace mask and fill masked values with nodata value
        result = np.ma.masked_array(
            result, mask=flat_pixels_mask.any(axis=1))

        # reshape the prediction from a 1D matrix/list
        # back into the original format [band, row, col]
        result = result.reshape((1, rows, cols))
        
        return result

    @staticmethod
    def __probfun(img, **kwargs):
        """Prediction probabilities function for RasterStack class
        Intended to be used internally
        
        Parameters
        ----------
        img : 3d array-like
            3d masked numpy array for image block to pass to estimator class
        estimator : Scikit-learn compatible estimator class
            Scikit-learn estimator that includes a fit(X,y) method
        
        Returns
        -------
        result : 3d array-like
            masked 3d numpy array containing classification result. Array order is in
            (class, row, col)
        """

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
        flat_pixels = np.ma.filled(flat_pixels, -99999)

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

        # reshape band into raster format [band, row, col]
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
        
        result = applier(rs=self, func=self.__predfun, output=output,
                         rowchunk=rowchunk, overwrite=overwrite,
                         **{'estimator':estimator})
        
        return result

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

        if isinstance(index, int):
            index = [index]        

        result = applier(rs=self, func=self.__probfun, output=output,
                         rowchunk=rowchunk, overwrite=overwrite,
                         **{'estimator':estimator,
                            'class_labels': class_labels,
                            'index': index})

        return result

    @staticmethod
    def __value_extractor(img, **kwargs):
        """Gets multidimensional array values at labelled pixel locations
        Intended to be used internally
        
        Parameters
        ----------
        img : 3d array-like
            3d masked numpy array containing raster data as (band, row, col)
            order. The labelled pixels represent the last element in axis=0.
        
        Returns
        -------
        data : 2d array-like
            2d numpy array containing sampled data and response value.
            Array order is (n_samples, n_features) with the last n_feature
            being the labelled pixel values (response)
        """
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
        
        # convert indexes of training pixels from tuple to n*2 np array
        is_train = np.array(is_train).T

        return (data, is_train)

    def extract_pixels(self, response, rowchunk=25, na_rm=True):
        """Samples a list of GRASS rasters using a labelled raster
        Per raster sampling
    
        Args
        ----
        response : str
            Name of GRASS raster with labelled pixels
        rowchunk : int
            Number of rows to read at one time
        na_rm : bool, optional
            Remove samples containing NaNs
    
        Returns
        -------
        X : 2d array-like
            Extracted raster values. Array order is (n_samples, n_features)
        y :  1d array-like
            Numpy array of labels
        crds : 2d array-like
            2d numpy array containing x,y coordinates of labelled pixel
            locations
        """

        # create new RasterStack object with labelled pixels as
        # last band in the stack
        temp_stack = RasterStack(self.fullnames + [response])

        reg = Region()
        reg.set_raster_region() # set region for all raster maps in session
        
        # generator for row increments, tuple (startrow, endrow)
        windows = ((row, row+rowchunk) if row+rowchunk <= reg.rows else
                   (row, reg.rows) for row in range(0, reg.rows, rowchunk))

        data = []
        crds = []
        for window in windows:
            img = temp_stack.read(window=window)
            training_data, crds_data = self.__value_extractor(img)
            data.append(training_data)
            crds.append(crds_data)
        data = np.concatenate(data, axis=1).transpose()
        crds = np.concatenate(crds, axis=0)

        raster_vals = data[:, 0:-1]
        labelled_vals = data[:, -1]
        for i in range(crds.shape[0]):
            crds[i, :] = np.array(pixel2coor(tuple(crds[i]), reg))

        if na_rm is True:
            X = raster_vals[~np.isnan(crds).any(axis=1)]
            y = labelled_vals[np.where(~np.isnan(raster_vals).any(axis=1))]
            crds = crds[~np.isnan(crds).any(axis=1)]
        else:
            X = raster_vals
            y = labelled_vals

        return (X, y, crds)

    def extract_features(self, vect_name, field, na_rm=False):
        """Samples a list of GDAL rasters using a point data set

        Parameters
        ----------
        vect_name : str
            Name of GRASS GIS vector containing point features
        field : str
            Name of attribute containing the response variable
        na_rm : bool, optional
            Remove samples containing NaNs

        Returns
        -------
        X : 2d array-like
            Extracted raster values. Array order is (n_samples, n_features)
        y :  1d array-like
            Numpy array of labels
        """

        # open grass vector
        points = VectorTopo(vect_name.split('@')[0])
        points.open('r')
    
        # create link to attribute table
        points.dblinks.by_name(name=vect_name)
    
        # extract table field to numpy array
        table = points.table
        cur = table.execute("SELECT {field} FROM {name}".format(field=field, name=table.name))
        y = np.array([np.isnan if c is None else c[0] for c in cur])
        y = np.array(y, dtype='float')
    
        # extract raster data
        X = np.zeros((points.num_primitives()['point'], len(self.fullnames)), dtype=float)
        for i, raster in enumerate(self.fullnames):
            region = Region()
            old_reg = deepcopy(region)
            region.from_rast(raster)
            region.set_raster_region()
            rio = RasterRow(raster)
            rio.open()
            values = np.asarray(get_raster_for_points(points, rio, region=region))
            coordinates = values[:, 1:3]
            X[:, i] = values[:, 3]
            rio.close()
            old_reg.set_current()
    
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
