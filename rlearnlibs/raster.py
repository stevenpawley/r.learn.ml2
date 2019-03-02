#!/usr/bin/env python
from __future__ import absolute_import, print_function
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
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.modules.shortcuts import raster as r
from grass.pygrass.modules.shortcuts import vector as v
from grass.pygrass.modules.shortcuts import imagery as im
from grass.pygrass import modules
from subprocess import PIPE
from collections import Counter

from indexing import ExtendedDict, LinkedList

class RasterStack(object):

    def __init__(self, rasters=None, group=None):
        """
        Args
        ----

        rasters : list, or str
            List of names of GRASS GIS raster maps
        
        group : str
            GRASS GIS imagery group to obtain a set of raster maps from
        
        Attributes
        ----------

        loc : dict
            Name-based indexing of RasterRow objects within the RasterStack
        
        iloc : int
            Index-based indexing of RasterRow objects within the RasterStack
                        
        mtypes : dict
            Dict of key, value pairs of full_names and GRASS data types
        
        count : int
            Number of RasterRow objects within the RasterStack
        """

        self.loc = ExtendedDict(self)           # label-based indexing
        self.iloc = LinkedList(self, self.loc)  # integer-based indexing
        self.mtypes = {}                        # key, value pairs of full name and GRASS data type
        self.count = 0                          # number of RasterRow objects in the stack
        self._categorical_idx = []              # indexes of categorical rasters in stack
        self._cell_nodata = -2147483648
        
        # some checks
        if rasters and group:
            gs.fatal('arguments "rasters" and "group" are mutually exclusive')
        
        if group:
            map_list = im.group(group=group, flags=["l", "g"], quiet=True, 
                                stdout_=PIPE)
            rasters = map_list.outputs.stdout.split(os.linesep)[:-1]
        
        self.layers = rasters  # call property
        
    def __getitem__(self, label):
        """
        Subset the RasterStack object using a label or list of labels
        
        Args
        ----
        label : str, list
            
        Returns
        -------
        A new RasterStack object only containing the subset of layers specified
        in the label argument
        """
        
        if isinstance(label, str):
            label = [label]
        
        subset_layers = []
        
        for i in label:
            
            if i in self.names is False:
                raise KeyError('layername not present in Raster object')
            else:
                subset_layers.append(self.loc[i])
            
        subset_raster = RasterStack(subset_layers)
        subset_raster.rename(
            {old : new for old, new in zip(subset_raster.names, label)})
        
        return subset_raster

    def __setitem__(self, key, value):
        """
        Replace a RasterLayer within the Raster object with a new RasterLayer
        
        Note that this modifies the Raster object in place
        
        Args
        ----
        key : str
            key-based index of layer to be replaced
        
        value : RasterRow object
            RasterRow to use for replacement
        """
        
        self.loc[key] = value
        self.iloc[self.names.index(key)] = value
        setattr(self, key, value)

    def __iter__(self):
        """
        Iterate over RasterRow objects
        """
        return(iter(self.loc.items()))
    
    @property
    def names(self):
        """
        Return the names of the RasterRow objects in the RasterStack
        """
        return list(self.loc.keys())

    @property
    def layers(self):
        return self.loc

    @layers.setter
    def layers(self, layers):
        """
        Setter method for the layers attribute in the RasterStack
        """
        
        # some checks
        if isinstance(layers, str):
            layers = [layers]
                
        # reset existing attributes
        for name in self.names:
            delattr(self, name)

        layer_names = [i.split('@')[0] for i in layers]
        mapset_names = [i.split('@')[1] if '@' in i else '' for i in layers]

        self.loc = ExtendedDict(self)
        self.iloc = LinkedList(self, self.loc)
        self.count = len(layers)
        self.mtypes = {}
        
        # add rasters and metadata to stack
        for layer, mapset in zip(layer_names, mapset_names):
            
            with RasterRow(name=layer, mapset=mapset) as src:

                if src.exist() is True:
    
                    ras_name = src.name.split('@')[0]  # name sans mapset
                    full_name = src.name_mapset()      # name with mapset
                    valid_name = ras_name.replace('.', '_')
    
                    self.mtypes.update({full_name: src.mtype})    
                    self.loc[valid_name] = src    
                    setattr(self, valid_name, src)
                
                else:
                    gs.fatal('GRASS raster map ' + r + ' does not exist')
    
    @property
    def categorical(self):
        return self._categorical_idx
    
    @categorical.setter
    def categorical(self, names):
        """
        Update the RasterStack categorical map indexes
        """
        
        if isinstance(names, str):
            names = [names]
            
        indexes = []
        
        # check that each category map is also in the imagery group
        for n in names:
            
            try:
                indexes.append(self.names.index(n))
            
            except ValueError:
                gs.fatal('Category map {0} not in the imagery group'.format(n))
        
        self._categorical_idx = indexes


    def append(self, other):
        """
        Setter method to add new RasterLayers to a Raster object
        
        Note that this modifies the Raster object in-place

        Args
        ----
        other : Raster object or list of Raster objects
        """
        
        if isinstance(other, str):
            other = [other]

        for new_raster in other:
        
            # check that other raster does not result in duplicated names
            combined_names = self.names + new_raster.names
            
            counts = Counter(combined_names)
            for s, num in counts.items():
                if num > 1:
                    for suffix in range(1, num + 1):
                        if s + "_" + str(suffix) not in combined_names:
                            combined_names[combined_names.index(s)] = s + "_" + str(suffix)
                        else:
                            i = 1
                            while s + "_" + str(i) in combined_names:
                                i += 1
                            combined_names[combined_names.index(s)] = s + "_" + str(i)

            # update layers and names
            self.layers = (list(self.loc.values()) + 
                           list(new_raster.loc.values()),
                           combined_names)

    def drop(self, names):
        """
        Drop individual rasters from the RasterStack
        
        Note that this modifies the RasterStack in-place

        Args
        ----
        names : single label or list-like
            Index (int) or name of GRASS raster to drop.
            Can be a single integer or name, or a list of integers or labels
        """

        # convert single label to list
        if isinstance(names, (str, int)):
            names = [names]

        # numerical index based subsetting
        if len([i for i in names if isinstance(i, int)]) == len(names):
            
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values())) if i not in names]
            subset_names = [v for (i, v) in enumerate(self.names) if i not in names]
            
        # str label based subsetting
        elif len([i for i in names if isinstance(i, str)]) == len(names):
            
            subset_layers = [v for (i, v) in enumerate(list(self.loc.values())) if self.names[i] not in names]
            subset_names = [v for (i, v) in enumerate(self.names) if self.names[i] not in names]

        else:
            raise ValueError('Cannot drop layers based on mixture of indexes and labels')
        
        # get grass raster names from the rasterrow objects
        subset_layers = [i.name for i in subset_layers]
        
        # update RasterStack with remaining maps and keep existing names
        self.layers = subset_layers
        self.rename({k:v for k,v in zip(self.names, subset_names)})

    def rename(self, names):
        """
        Rename a RasterLayer within the Raster object
        
        Note that this modifies the Raster object in-place

        Args
        ----
        names : dict
            dict of old_name : new_name
        """
        
        for old_name, new_name in names.items():
            self.loc[new_name] = self.loc.pop(old_name)
    
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
            Extracted raster values with shape (n_samples, n_features)
            
        y : array-like
            array of labels with shape (n_samples, n_fields)
                
        df : pandas.DataFrame
            Extracted raster values as pandas dataframe if as_df = True
            The coordinates of the sampled data are also returned as x, y
            columns and the index of the dataframe uses the GRASS cat field

        Notes
        -----
        Values of the RasterStack object are read for the full extent of the
        supplied vector feature, i.e. current region settings are ignored.
        If you want to extract raster data for a spatial subset of the supplied
        point features, then clip the vector features beforehand.
        """
        
        if isinstance(fields, str):
            fields = [fields]
        
        # collapse list of fields to comma separated string
        if len(fields) > 1:
            field_names = ','.join(fields)
        else:
            field_names = ''.join(fields)
    
        # open grass vector
        with VectorTopo(vect_name.split('@')[0], mode='r') as points:
    
            # create link to attribute table
            points.dblinks.by_name(name=vect_name)
    
            # extract table field to numpy array
            table = points.table        
            sqlpath = gs.read_command("db.databases", driver="sqlite").strip(os.linesep)
            con = sqlite3.connect(sqlpath)
            df = pd.read_sql_query(
                    "SELECT {fields} FROM {name}".format(
                            fields=field_names, name=table.name), con)
            y = df[fields].values
            df = None
            con.close()
    
            # extract raster data
            X = np.zeros((points.num_primitives()['point'], self.count),
                         dtype=float)
    
            for i, src in enumerate(self.iloc):
                rast_data = v.what_rast(
                    map=vect_name,
                    raster=src.fullname(),
                    flags='p', stdout_=PIPE).outputs.stdout
                rast_data = rast_data.split(os.linesep)[:-1]
                X[:, i] = np.asarray([k.split('|')[1] for k in rast_data])
            
            cat = np.asarray([k.split('|')[0] for k in rast_data])
            
            # get coordinate and id values
            coordinates = np.zeros((points.num_primitives()['point'], 2))
            for i, p in enumerate(points.viter(vtype='points')):
                coordinates[i, :] = np.asarray(p.coords())

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
        cat = cat[~na_rows]

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
                              columns=['x', 'y'] + fields + self.names,
                              index=cat)
            return df

        return(X, y)

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
