from __future__ import absolute_import, print_function
from collections import OrderedDict
import numpy as np
import grass.script as gs
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.vector import VectorTopo
from grass.pygrass.utils import get_raster_for_points, pixel2coor


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
        Indices of rasters that represent categorical datatypes"""

    def __init__(self, rasters, categorical=None):
        """Create a RasterStack object

        Parameters
        ----------
        rasters : list (str)
            List of names of GRASS rasters to create a RasterStack object."""

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
    
    def dummy_encoder(self, X):
        """Helper function to return a fitted
        sklearn.preprocessing.data.OneHotEncoder object based on the
        categorical rasters in the RasterStack
        
        Parameters
        ----------
        X : 2d-array like
            Training data derived from the current RasterStack and including
            categorical variables to return a fitted OneHotEncoder Object
        
        Returns
        -------
        enc : sklearn.preprocessing.data.OneHotEncoder"""

        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder(categorical_features=self.categorical)
        enc.fit(X)
        enc = OneHotEncoder(
            categorical_features=self.categorical, n_values=enc.n_values_,
            handle_unknown='ignore', sparse=False)
        
        return enc
        

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
            src.close()

            # mask array with nodata
            if src.mtype == 'CELL':
                data = np.ma.masked_equal(data, self.cell_nodata)
            elif src.mtype in ['FCELL', 'DCELL']:
                data = np.ma.masked_invalid(data)

            if isinstance(data.mask, np.bool_):
                mask_arr = np.empty(data.shape, dtype='bool')
                mask_arr[:] = False
                data.mask = mask_arr

        return data
       

    def predict(self, estimator, output=None, height=25, overwrite=False):
        """Prediction method for RasterStack class
        
        Parameters
        ----------
        estimator : estimator object implementing ‘fit’
            The object to use to fit the data.
        output : str
            Output name for prediction raster
        height : int
            Number of raster rows to pass to estimator at one time
        overwrite : bool
            Option to overwrite an existing raster"""
                    
        # processing region dimensions
        reg = Region()
        reg.set_raster_region() # set region for all raster maps in session
        
        # determine dtype
        img = self.read(window=self.row_windows(height=height).next())
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
        result = estimator.predict(flat_pixels)

        if result.dtype == 'float':
            mtype = 'FCELL'
            nodata = np.nan
        else:
            mtype = 'CELL'
            nodata = -2147483648
        
        # open dst
        dst = RasterRow(output)
        dst.open('w', mtype=mtype, overwrite=overwrite)
        
        for window in self.row_windows(height=height):

            img = self.read(window=window)
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
            result = np.ma.filled(result, nodata)

            # writing data to GRASS raster
            for i in range(result.shape[1]):
                newrow = Buffer((reg.cols,), mtype=mtype)
                newrow[:] = result[0, i, :]
                dst.put_row(newrow)

        dst.close()

        return None

    def predict_proba(self, estimator, output=None, class_labels=None,
                      index=None, height=25, overwrite=False):
        """Prediction method for RasterStack class
        
        Parameters
        ----------
        estimator : estimator object implementing ‘fit’
            The object to use to fit the data.
        output : str
            Output name for prediction raster
        class_labels : 1d array-like, optional
            class labels
        index : list, optional
            list of class indices to export
        height : int
            Number of raster rows to pass to estimator at one time
        overwrite : bool
            Option to overwrite an existing raster(s)"""

        if isinstance(index, int):
            index = [index]        

        # processing region dimensions
        reg = Region()
        reg.set_raster_region() # set region for all raster maps in session
                
        # use class labels if supplied else output preds as 0,1,2...n
        if class_labels is None:
            img = self.read(window=self.row_windows(height=height).next())
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
            result = estimator.predict_proba(flat_pixels)
            result = result.reshape((rows, cols, result.shape[1]))
            class_labels = range(result.shape[2])      

        # output all class probabilities if subset is not specified
        if index is None:
            index = class_labels
        labels = [i for i, x in enumerate(class_labels) if x in index]
        
        # create and open rasters for writing
        dst = []
        for pred_index, label in zip(labels, index):
            rastername = output + '_' + str(label)
            dst.append(RasterRow(rastername))
            dst[pred_index].open('w', mtype='FCELL', overwrite=overwrite)

        for window in self.row_windows(height=height):

            img = self.read(window=window)
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
            result = estimator.predict_proba(flat_pixels)
    
            # reshape class probabilities back to 3D image [iclass, rows, cols]
            result = result.reshape(
                (rows, cols, result.shape[1]))
            flat_pixels_mask = flat_pixels_mask.reshape(
                (rows, cols, n_features))
    
            # flatten mask into 2d
            mask2d = flat_pixels_mask.any(axis=2)
            mask2d = np.where(mask2d != mask2d.min(), True, False)
            mask2d = np.repeat(mask2d[:, :, np.newaxis],
                               result.shape[2], axis=2)
    
            # convert proba to masked array using mask2d
            result = np.ma.masked_array(
                result, mask=mask2d, fill_value=np.nan)
    
            # reshape band into raster format [band, row, col]
            result = result.transpose(2, 0, 1)
            
            # write multiple features to 
            result = np.ma.filled(result, np.nan)
            for pred_index in index:
                for i in range(result.shape[1]):
                    newrow = Buffer((reg.cols,), mtype='FCELL')
                    newrow[:] = result[pred_index, i, :]
                    dst[pred_index].put_row(newrow)
    
        # close maps
        [i.close() for i in dst]

        return None
    
    def row_windows(self, region=None, height=25):
        """Generator for row increments, tuple (startrow, endrow)
        
        Parameters
        ----------
        region = grass.pygrass.gis.region.Region object, optional
            Optionally restrict windows to specified region
        height = int, default = 25
            Height of window in number of image rows"""
        
        if region is None:
            region = Region()
        
        windows = ((row, row+height) if row+height <= region.rows else
                   (row, region.rows) for row in range(0, region.rows, height))
        
        return windows

    def extract_pixels(self, response, height=25, region=None, na_rm=True):
        """Samples a list of GRASS rasters using a labelled raster
        Per raster sampling
    
        Args
        ----
        response : str
            Name of GRASS raster with labelled pixels
        height : int
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
            locations"""

        # create new RasterStack object with labelled pixels as
        # last band in the stack
        temp_stack = RasterStack(self.fullnames + [response])

        if region is None: 
            region = Region()
            region.set_raster_region()
        
        data = []
        crds = []
        
        for window in self.row_windows(region=region, height=height):
            img = temp_stack.read(window=window)
            
            # split numpy array bands(axis=0) into labelled pixels and
            # raster data
            response_arr = img[-1, :, :]
            raster_arr = img[0:-1, :, :]
    
            # returns indices of labelled values and values
            val_indices = np.nonzero(~response_arr.mask)
            labels = response_arr.data[val_indices]
    
            # extract data at labelled pixel locations
            values = raster_arr[:, val_indices[0], val_indices[1]]
            values = values.filled(np.nan)
    
            # combine training data, locations and labels
            values = np.vstack((values, labels))
            val_indices = np.array(val_indices).T 
                        
            data.append(values)
            crds.append(val_indices)

        data = np.concatenate(data, axis=1).transpose()
        crds = np.concatenate(crds, axis=0)

        raster_vals = data[:, 0:-1]
        labelled_vals = data[:, -1]

        for i in range(crds.shape[0]):
            crds[i, :] = np.array(pixel2coor(tuple(crds[i]), region))

        if na_rm is True:
            X = raster_vals[~np.isnan(crds).any(axis=1)]
            y = labelled_vals[np.where(~np.isnan(raster_vals).any(axis=1))]
            crds = crds[~np.isnan(crds).any(axis=1)]
        else:
            X = raster_vals
            y = labelled_vals

        return (X, y, crds)

    def extract_points(self, vect_name, field, na_rm=True):
        """Samples a list of GDAL rasters using a point data set

        Parameters
        ----------
        vect_name : str
            Name of GRASS GIS vector containing point features
        field : str
            Name of attribute containing the response variable
        na_rm : bool, optional, default = True
            Remove samples containing NaNs

        Returns
        -------
        X : 2d array-like
            Extracted raster values. Array order is (n_samples, n_features)
        y :  1d array-like
            Numpy array of labels
        
        Notes
        -----
        Values of the RasterStack object are read for the full extent of the
        supplied vector feature, i.e. current region settings are ignored.
        If you want to extract raster data for a spatial subset of the supplied
        point features, then clip the vector features beforehand."""

        region = Region()

        # open grass vector
        points = VectorTopo(vect_name.split('@')[0])
        points.open('r')
    
        # create link to attribute table
        points.dblinks.by_name(name=vect_name)
    
        # extract table field to numpy array
        table = points.table
        cur = table.execute(
            "SELECT {field} FROM {name}".format(field=field, name=table.name))
        y = np.array([np.isnan if c is None else c[0] for c in cur])
        y = np.array(y, dtype='float')
    
        # extract raster data
        X = np.zeros(
            (points.num_primitives()['point'], len(self.fullnames)),
            dtype=float)
        points.close()
        
        for i, raster in enumerate(self.fullnames):
            region.from_rast(raster)
            region.set_raster_region()
            src = RasterRow(raster)
            src.open()
            values = np.asarray(
                get_raster_for_points(points, src, region=region))
            coordinates = values[:, 1:3]
            X[:, i] = values[:, 3]
            src.close()
    
        # set any grass integer nodata values to NaN
        X[X == self.cell_nodata] = np.nan
    
        # remove missing response data
        X = X[~np.isnan(y)]
        coordinates = coordinates[~np.isnan(y)]
        y = y[~np.isnan(y)]
    
        # int type if classes represented integers
        if all(y % 1 == 0) is True:
            y = np.asarray(y, dtype='int')
    
        # remove samples containing NaNs
        if na_rm is True:
            if np.isnan(X).any() == True:
                gs.message('Removing samples with NaN values in the raster feature variables...')
    
            y = y[~np.isnan(X).any(axis=1)]
            coordinates = coordinates[~np.isnan(X).any(axis=1)]
            X = X[~np.isnan(X).any(axis=1)]
    
        return(X, y, coordinates)
