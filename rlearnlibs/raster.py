#!/usr/bin/env python
from __future__ import absolute_import, print_function
from collections import OrderedDict
import numpy as np
import grass.script as gs
import pandas as pd
import os
import sqlite3
from copy import deepcopy
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster.buffer import Buffer
from grass.pygrass.vector import VectorTopo
from grass.pygrass.utils import get_raster_for_points
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.modules.shortcuts import raster as r
from grass.pygrass import modules
from subprocess import PIPE


class RasterStack(object):
    """
    Flexible class to represent a collection of RasterRow objects
    
    A RasterStack is initiated from a list of RasterRow objects

    Additional RasterRow objects can be added to the existing RasterStack
    object using the append() method. Any RasterRow object can also be 
    removed using the drop() method.
    """

    def __init__(self, rasters, categorical_names=None):
        """
        Args
        ----

        rasters : list, or str
            List of names of GRASS GIS raster maps to initiated the class

        categorical_names : list, optional
            List of names of GRASS GIS rasters that represent categorical
            map data
        
        Attributes
        ----------

        loc : dict
            Name-based indexing of RasterRow objects within the RasterStack
        
        iloc : int
            Index-based indexing of RasterRow objects within the RasterStack
        
        names : list
            List of syntatically-valid names of GRASS GIS raster maps
        
        full_names : list
            List of names of GRASS GIS raster maps including mapset names if
            supplied
        
        mtypes : dict
            Dict of key, value pairs of full_names and GRASS data types
        
        count : int
            Number of RasterRow objects within the RasterStack
        """

        self.loc = {}          # name-based indexing of RasterRow objects
        self.iloc = []         # index-based indexing of RasterRow objects
        self.names = []        # names of GRASS GIS rasters modified to be synatically-valid
        self.full_names = []   # names of GRASS GIS rasters including mapset names if supplied
        self.mtypes = {}       # key, value pairs of full name and GRASS data type
        self.count = 0         # number of RasterRow objects in the stack
        self.categorical = []  # list of indices of GRASS GIS rasters in stack representing categorical data
        self._cell_nodata = -2147483648
        
        self._layers = None    # set proxy for self._layers
        self.layers = (rasters, categorical_names)  # call property
        
    def __getitem__(self, layer_name):
        """
        Get a RasterLayer within the Raster object using label-based indexing
        """

        if layer_name in self.names is False:
            raise AttributeError('layername not present in RasterStack object')

        return getattr(self, layer_name)

    def iterlayers(self):
        """
        Iterate over Raster object layers
        """

        for k, v in self.loc.items():
            yield k, v

    @property
    def layers(self):
        """
        Getter method for file names within the Raster object
        """

        return self._layers

    @layers.setter
    def layers(self, layers):
        """
        Setter method for the layers attribute in the RasterStack object
        """
        
        layers, categorical_names = layers

        # some checks
        if isinstance(layers, str):
            layers = [layers]
        
        if all(isinstance(x, type(layers[0])) for x in layers) is False:
            raise ValueError("Cannot create a RasterStack object from a mixture of input types")
        
        # reset existing attributes
        for name in self.names:
            delattr(self, name)

        layer_names = [i.split('@')[0] for i in layers]
        mapset_names = [i.split('@')[1] if '@' in i else '' for i in layers ]

        self.iloc = []
        self.loc = OrderedDict()
        self.count = len(layers)
        self.full_names = []
        self.mtypes = {}
        self.categorical = []
        self._layers = layers
        
        # add rasters and metadata to stack
        for layer, mapset in zip(layer_names, mapset_names):
            
            with RasterRow(name=layer, mapset=mapset) as src:

                if src.exist() is True:
    
                    ras_name = src.name.split('@')[0]  # name of map sans mapset
                    full_name = src.name_mapset()  # name with mapset
                    valid_name = ras_name.replace('.', '_')  # syntactically correct name
    
                    self.full_names.append(full_name)
                    self.names.append(ras_name)
                    self.mtypes.update({full_name: src.mtype})
    
                    self.loc.update({valid_name: src})
                    self.iloc.append(src)
    
                    setattr(self, valid_name, src)
                
                else:
                    gs.fatal('GRASS raster map ' + r + ' does not exist')
        
        # extract indices of category maps
        if categorical_names is not None:
                
            # check that each category map is also in the imagery group
            for cat in categorical_names:
                try:
                    self.categorical.append(self.full_names.index(cat))
                except ValueError:
                    gs.fatal('Category map {0} not in the imagery group'.format(cat))

    def read(self, row=None, window=None):
        """
        Read data from RasterStack as a masked 3D numpy array
        
        Notes
        -----
        Read an entire RasterStack into a numpy array. If row or window is 
        supplied, then a single row, or a range of rows from 
        window = (start_row, end_row) is read into an array.

        Parameters
        ----------
        row : int, optional
            Integer representing the index of a single row of a raster to read

        window : tuple, optional
            Tuple of integers representing the start and end numbers of rows to
            read as a single block of rows

        Returns
        -------
        data : 3d array-like
            3d masked numpy array containing data from RasterStack rasters
        """

        reg = Region()

        # create numpy array to receive data based on row/window/dataset size
        if window:
            row_start, row_stop = window
            width = reg.cols
            height = abs(row_stop-row_start)
            shape = (self.count, height, width)

        elif row:
            row_start = row
            row_stop = row+1
            height = 1
            shape = (self.count, height, reg.cols)
        
        else:
            shape = (self.count, reg.rows, reg.cols) 

        data = np.zeros(shape)

        if row or window:
            rowincrs = [i for i in range(row_start, row_stop)]

        # read from each RasterRow object
        for band, src in enumerate(self.iloc):
            
            try:
                src.open()
        
                if row or window:
                    for i, row in enumerate(rowincrs):
                        data[band, i, :] = src[row]
                
                else:
                    data[band, :, :] = np.asarray(src)

                src.close()

            except:
                gs.fatal('Cannot read from raster {0}'.format(src.fullname))

            finally:
                src.close()

        # mask array
        data = np.ma.masked_equal(data, self._cell_nodata)
        data = np.ma.masked_invalid(data)

        if isinstance(data.mask, np.bool_):
            mask_arr = np.empty(data.shape, dtype='bool')
            mask_arr[:] = False
            data.mask = mask_arr

        return data

    def predict(self, estimator, output=None, height=25, overwrite=False):
        """
        Prediction method for RasterStack class

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        output : str
            Output name for prediction raster
        height : int
            Number of raster rows to pass to estimator at one time
        overwrite : bool
            Option to overwrite an existing raster
        """

        reg = Region()

        # determine dtype
        img = self.read(window=self.row_windows(height=height).next())
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
        flat_pixels = np.ma.filled(flat_pixels, -99999)
        result = estimator.predict(flat_pixels)

        if result.dtype == 'float':
            mtype = 'FCELL'
            nodata = np.nan
        else:
            mtype = 'CELL'
            nodata = -2147483648

        # open dst
        with RasterRow(output, mode='w', overwrite=overwrite) as dst:
            
            n_windows = len([i for i in self.row_windows(height=height)])
    
            for wi, window in enumerate(self.row_windows(height=height)):
                
                gs.percent(wi, n_windows, 1)
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

        return None

    def predict_proba(self, estimator, output=None, class_labels=None,
                      height=25, overwrite=False):
        """
        Prediction method for RasterStack class

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data
            
        output : str
            Output name for prediction raster
            
        class_labels : 1d array-like, optional
            class labels
            
        height : int
            Number of raster rows to pass to estimator at one time
            
        overwrite : bool
            Option to overwrite an existing raster(s)
        """

        reg = Region()

        # use class labels if supplied else output preds as 0,1,2...n
        if class_labels is None:
            
            # make a small prediction
            img = self.read(window=self.row_windows(height=height).next())
            n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
            n_samples = rows * cols
            flat_pixels = img.transpose(1, 2, 0).reshape(
                (n_samples, n_features))
            flat_pixels = np.ma.filled(flat_pixels, -99999)
            
            result = estimator.predict_proba(flat_pixels)
            result = result.reshape((rows, cols, result.shape[1]))
            
            # determine number of class probs
            class_labels = range(result.shape[2])
        
        # only output positive class if result is binary
        if len(class_labels) == 2:
            class_labels, indexes = [max(class_labels)], [1]
        else:
            indexes = np.arange(0, len(class_labels), 1)

        # create and open rasters for writing
        dst = []
        for i, label in enumerate(class_labels):
            rastername = output + '_' + str(label)
            dst.append(RasterRow(rastername))
            dst[i].open('w', mtype='FCELL', overwrite=overwrite)

        n_windows = len([i for i in self.row_windows(height=height)])

        try:
            for wi, window in enumerate(self.row_windows(height=height)):
                gs.percent(wi, n_windows, 1)

                img = self.read(window=window)
                n_features, rows, cols = img.shape[0], img.shape[1],\
                    img.shape[2]

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

                # reshape class probabilities back to 3D [iclass, rows, cols]
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
                result = np.ma.filled(result, np.nan)

                # write multiple features to
                for i, arr_index in enumerate(indexes):
                    for row in range(result.shape[1]):
                        newrow = Buffer((reg.cols, ), mtype='FCELL')
                        newrow[:] = result[arr_index, row, :]
                        dst[i].put_row(newrow)
        except:
            gs.fatal('Error in raster prediction')
        finally:
            [i.close() for i in dst]

        return None

    def row_windows(self, region=None, height=25):
        """
        Returns an generator for row increments, tuple (startrow, endrow)

        Args
        ----
        region = grass.pygrass.gis.region.Region object, optional
            Optionally restrict windows to specified region
            
        height = int, default = 25
            Height of window in number of image rows
        """

        if region is None:
            region = Region()

        windows = ((row, row+height) if row+height <= region.rows else
                   (row, region.rows) for row in range(0, region.rows, height))

        return windows
    
    def extract_pixels(self, response, as_df=False):
        
        data = r.stats(input=[response] + self.names,
                       separator='pipe',
                       flags=['n', 'g'], 
                       stdout_=PIPE).outputs.stdout
                
        data = data.split(os.linesep)[:-1]
        data = [i.split('|') for i in data]
        data = np.asarray(data).astype('float32')

        coordinates = data[:, 0:2]
        y = data[:, 2]
        X = data[:, 3:]
        
        if (y % 1).all() == 0:
            y = y.astype('int')
        
        if as_df is True:
            data = pd.DataFrame(data, columns = ['x', 'y', response] + self.names)
            return data
                    
        return X, y, coordinates

#    def extract_pixels(self, response, height=25, region=None, na_rm=True):
#        """
#        Samples a list of GRASS rasters using a labelled raster
#        Per raster sampling
#
#        Args
#        ----
#        response : str
#            Name of GRASS raster with labelled pixels
#        height : int
#            Number of rows to read at one time
#        na_rm : bool, optional
#            Remove samples containing NaNs
#
#        Returns
#        -------
#        X : 2d array-like
#            Extracted raster values. Array order is (n_samples, n_features)
#        y :  1d array-like
#            Numpy array of labels
#        crds : 2d array-like
#            2d numpy array containing x,y coordinates of labelled pixel
#            locations
#        """
#
#        # create new RasterStack object with labelled pixels as
#        # last band in the stack
#        temp_stack = RasterStack(self.full_names + [response])
#
#        if region is None:
#            region = Region()
#            region.set_raster_region()
#
#        data = []
#        crds = []
#
#        for window in self.row_windows(region=region, height=height):
#
#            img = temp_stack.read(window=window)
#
#            # split numpy array bands(axis=0) into labelled pixels and
#            # raster data
#            response_arr = img[-1, :, :]
#            raster_arr = img[0:-1, :, :]
#
#            # returns indices of labelled values and values
#            val_indices = np.nonzero(~response_arr.mask)
#            labels = response_arr.data[val_indices]
#
#            # extract data at labelled pixel locations
#            values = raster_arr[:, val_indices[0], val_indices[1]]
#            values = values.filled(np.nan)
#
#            # combine training data, locations and labels
#            values = np.vstack((values, labels))
#            val_indices = np.array(val_indices).T
#
#            data.append(values)
#            crds.append(val_indices)
#
#        data = np.concatenate(data, axis=1).transpose()
#        crds = np.concatenate(crds, axis=0)
#
#        raster_vals = data[:, 0:-1]
#        labelled_vals = data[:, -1]
#
#        for i in range(crds.shape[0]):
#            crds[i, :] = np.array(pixel2coor((crds[i]), region))
#
#        if na_rm is True:
#            X = raster_vals[~np.isnan(crds).any(axis=1)]
#            y = labelled_vals[np.where(~np.isnan(raster_vals).any(axis=1))]
#            crds = crds[~np.isnan(crds).any(axis=1)]
#        else:
#            X = raster_vals
#            y = labelled_vals
#
#        return (X, y, crds)

    def extract_points(self, vect_name, fields, na_rm=True, as_df=False):
        """
        Samples a list of GDAL rasters using a point data set

        Parameters
        ----------
        vect_name : str
            Name of GRASS GIS vector containing point features
            
        fields : list, str
            Name of attribute(s) containing the response variable(s)
            
        na_rm : bool, optional, default = True
            Remove samples containing NaNs
        
        as_df : bool, optional, default = False
            Extract data to pandas dataframe

        Returns
        -------
        X : 2d array-like
            Extracted raster values. Array order is (n_samples, n_features)
            
        y :  1d array-like
            Numpy array of labels
        
        coordinates : 2d array-like
            2d array of x, y coordiantes of samples
        
        df : pandas.DataFrame
            Extracted raster values as pandas dataframe if as_df = True

        Notes
        -----
        Values of the RasterStack object are read for the full extent of the
        supplied vector feature, i.e. current region settings are ignored.
        If you want to extract raster data for a spatial subset of the supplied
        point features, then clip the vector features beforehand.
        """

        region = Region()
        
        # collapse list of fields to comma separated string
        if isinstance(fields, list) and len(fields) > 1:
            fields = ','.join(fields)

        # open grass vector
        points = VectorTopo(vect_name.split('@')[0])
        points.open('r')

        # create link to attribute table
        points.dblinks.by_name(name=vect_name)

        # extract table field to numpy array
        table = points.table
#        cur = table.execute(
#            "SELECT {fields} FROM {name}".format(fields=fields, name=table.name))
#        y = np.array([np.isnan if c is None else c[0] for c in cur])
#        y = np.array(y, dtype='float')
        
        sqlpath = gs.read_command("db.databases", driver="sqlite").strip(os.linesep)
        con = sqlite3.connect(sqlpath)
        df = pd.read_sql_query(
                "SELECT {fields} FROM {name}".format(
                        fields=fields, name=table.name), con)
        y = df[fields.split(',')].values
        con.close()

        # extract raster data
        X = np.zeros(
            (points.num_primitives()['point'], len(self.full_names)),
            dtype=float)
        points.close()

        for i, raster in enumerate(self.full_names):
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
        X[X == self._cell_nodata] = np.nan

        # remove rows with missing response data
        if len(y.shape) > 1:
            na_rows = np.isnan(y).any(axis=1)
        else:
            na_rows = np.isnan(y)
        
        X = X[~na_rows]
        coordinates = coordinates[~na_rows]
        y = y[~na_rows]

        # int type if classes represented integers
        if (y % 1).all() == 0:
            y = y.astype('int')

        # remove samples containing NaNs
        if na_rm is True:
            if np.isnan(X).any() == True:
                gs.message('Removing samples with NaN values in the ' +
                           'raster feature variables...')

            y = y[~np.isnan(X).any(axis=1)]
            coordinates = coordinates[~np.isnan(X).any(axis=1)]
            X = X[~np.isnan(X).any(axis=1)]
        
        if as_df is True:
            df = pd.DataFrame(data=np.column_stack((coordinates, y, X)),
                              columns = ['x', 'y'] + fields + self.names)
            return df

        return(X, y, coordinates)

    def append(self, other):
        """
        Setter method to add new Raster objects

        Args
        ----
        other : RasterStack object or list of names of GRASS GIS raster maps
        """

        if isinstance(other, RasterStack):
            self.layers = self.layers + other.layers

        elif isinstance(other, list):
            for raster in other:
                self.layers = self.layers + raster.layers

    def drop(self, labels):
        """
        Drop individual layers from a RasterStack object

        Args
        ----
        labels : single label or list-like
            Index (int) or layer name to drop. Can be a single integer or label,
            or a list of integers or labels
        """

        # convert single label to list
        if isinstance(labels, (str, int)):
            labels = [labels]

        if len([i for i in labels if isinstance(i, int)]) == len(labels):
            # numerical index based subsetting
            self.layers = [v for (i, v) in enumerate(self.layers) if i not in labels]
            self.names = [v for (i, v) in enumerate(self.names) if i not in labels]

        elif len([i for i in labels if isinstance(i, str)]) == len(labels):
            # str label based subsetting
            self.layers = [v for (i, v) in enumerate(self.layers) if self.names[i] not in labels]
            self.names = [v for (i, v) in enumerate(self.names) if self.names[i] not in labels]

        else:
            raise ValueError('Cannot drop layers based on mixture of indexes and labels')

    def to_pandas(self, res=None, resampling='nearest'):
        """
        RasterStack to pandas DataFrame

        Args
        ----
        max_pixels: int, default=50000
            Maximum number of pixels to sample

        resampling : str, default = 'nearest'
            Other resampling methods consist of 'average', 'median', 'mode', 
            'minimum', 'maximum', 'range', 'quart1', 'quart3', 'perc90', 'sum', 
            'variance', 'stddev', 'quantile', 'count', 'diversity'
        
        Returns
        -------
        df : pandas DataFrame
        """

        reg = Region()
        temp_names = []
        
        if res:
            old_reg = deepcopy(reg)
            reg.ewres = res
            reg.nsres = res
            reg.write()
                        
            if resampling != 'nearest':
                resample_tool = modules.Module('r.resamp.stats')
                method_opts = resample_tool.params_list[2]
                        
                if resampling not in method_opts.values:
                    gs.fatal('resampling method should be one of: ' + 
                             str(method_opts.values))
                                
                for name in self.full_names:
                    temp = gs.tempname(6)
                    temp_names.append(temp)
                    
                    resample_tool(input=name, output=temp, method=resampling)
                    resample_tool.run()
            
                tmp_stack = RasterStack(temp_names)
                arr = tmp_stack.read()
            
            else:
                arr = self.read()
        
        else:
            arr = self.read()
        
        # generate x and y grid coordinate arrays
        x_range = np.linspace(start=reg.west, stop=reg.east, num=reg.cols)
        y_range = np.linspace(start=reg.south, stop=reg.north, num=reg.rows)
        xs, ys = np.meshgrid(x_range, y_range)

        # flatten 3d data into 2d array (layer, sample)
        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        arr = arr.transpose()
        
        # convert to dataframe
        df = pd.DataFrame(np.column_stack((xs.flatten(), ys.flatten(), arr)),
                          columns=['x', 'y'] + self.names)

        # set nodata values to nan
        for i, col_name in enumerate(self.names):
            df.loc[df[col_name] == self._cell_nodata, col_name] = np.nan
            
        if res:
            old_reg.write()
            
            for name in temp_names:
                g.remove(input=name, flags='f')

        return df

    def head(self):
        """
        Show the head (first rows, first columns) or tail (last rows, last columns)
        of the cells of a Raster object
        """

        window = (1, 10)
        arr = self.read(window=window)

        return arr

    def tail(self):
        """
        Show the head (first rows, first columns) or tail (last rows, last columns)
        of the cells of a Raster object
        """

        reg = Region()
        window = (reg.rows-10, reg.rows)
        arr = self.read(window=window)

        return arr
