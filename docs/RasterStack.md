# RasterStack
 
A RasterStack enables a collection of RasterRow objects to be bundled into a multi-layer RasterStack object

## Contents
- [Parameters](#Parameters)
- [Atttributes](#Attributes)
- [Methods](#Methods)
     - [RasterStack.names](#RasterStack.names)


## Parameters
  rasters : list, str
      
      List of names of GRASS GIS raster maps. Note that although the
      maps can reside in different mapsets, they cannot have the same
      names.

  group : str (opt)
      
      Create a RasterStack from rasters contained in a GRASS GIS imagery
      group. This parameter is mutually exclusive with the `rasters`
      parameter.

## Attributes
  loc : dict
      
      Label-based indexing of RasterRow objects within the RasterStack.

  iloc : int
      
      Index-based indexing of RasterRow objects within the RasterStack.

  mtypes : dict
      
      Dict of key, value pairs of full_names and GRASS data types.

  count : int
      
      Number of RasterRow objects within the RasterStack.
     
  categorical : list
  
      A list of the names with the RasterStack that represent rasters with categorical values.
      Used to keep track of indices when using one-hot-encoding.

## Methods

### RasterStack.names

Returns the names of the grass.pygrass.raster.RasterRow objects in the RasterStack

#### Parameters
    None

#### Returns
    list

### RasterStack.append

Method to add new RasterRows to a RasterStack object

#### Parameters

other : str, or list

    Name of GRASS GIS rasters, or a list of names.
        
in_place : bool (opt). Default is True

    Whether to change the Raster object in-place or leave original and
    return a new Raster object.

### RasterStack.drop

Drop individual RasterRow objects from the RasterStack

Note that this modifies the RasterStack object in-place by default

#### Parameters
labels : single label or list-like

    Index (int) or layer name to drop. Can be a single integer or
    label, or a list of integers or labels.

in_place : bool (opt). Default is True

    Whether to change the RasterStack object in-place or leave original
    and return a new RasterStack object.

#### Returns
 RasterStack
 
     Returned only if `in_place` is True

### RasterStack.read
Read data from RasterStack as a masked 3D numpy array

If the row parameter is used then a single row is read into a 3d numpy array

If the rows parameter is used, then a range of rows from (start_row, end_row) is read
into a 3d numpy array

If no additional arguments are supplied, then all of the maps within the RasterStack are
read into a 3d numpy array (obeying the GRASS region settings)

#### Parameters
row : int (opt)

    Integer representing the index of a single row of a raster to read.

### RasterStack.predict_proba


rows : tuple (opt)

    Tuple of integers representing the start and end numbers of rows to
    read as a single block of rows.

#### Returns
data : ndarray

    3d masked numpy array containing data from RasterStack rasters.

### RasterStack.predict

Prediction method for RasterStack class

#### Parameters
estimator : estimator object implementing 'fit'

    The object to use to fit the data.

output : str

    Output name for prediction raster.

height : int (opt).

    Number of raster rows to pass to estimator at one time. If not
    specified then the entire raster is read into memory.

overwrite : bool (opt). Default is False

    Option to overwrite an existing raster.

#### Returns
RasterStack

    A new RasterStack object containing the predictions.

### RasterStack.predict_proba

Prediction method for RasterStack class

#### Parameters
estimator : estimator object implementing 'fit'

    The object to use to fit the data

output : str

    Output name for prediction raster

class_labels : ndarray (opt)

    1d array containing indices of class labels to subset the
    prediction by. Only probability outputs for these classes will be
    produced.

height : int (opt)

    Number of raster rows to pass to estimator at one time. If not
    specified then the entire raster is read into memory.

overwrite : bool (opt). Default is False

    Option to overwrite an existing raster(s)

#### Returns
RasterStack

     A new RasterStack containing the class probability rasters, one raster for each class.

### RasterStack.row_windows

Returns an generator for row increments, tuple (startrow, endrow)

#### Parameters
region : grass.pygrass.gis.region.Region (opt)

    Whether to restrict windows to specified region.

height : int (opt). Default is 25

    Height of window in number of image rows.

#### Returns

generator

    A generator that returns (row_start, row_stop) positions for the region.

### RasterStack.extract_pixels

Extract pixel values from a RasterStack using another RasterRow
object of labelled pixels

#### Parameters
rast_name : str

    Name of GRASS GIS raster map containing labelled pixels.

use_cats : bool (default is False)

    Whether to return pixel values as category labels instead of
    numeric values if the rast_name map has categories assigned to
    it. Note that if some categories are missing in the rast_name
    map then this option is ignored.

as_df : bool (opt). Default is False

    Whether to return the extracted RasterStack pixels as a Pandas
    DataFrame.

#### Returns

X : ndarray

    2D numpy array of the extracted pixels values in the raster

y : ndarray

    1D numpy array of the labelled pixels in the rast_name map

cat : ndarray

    1D numpy array of the pixel indices

df : pandas.DataFrame

    Pandas dataframe containing the extracted data. Only returned if as_df=True.

### RasterStack.extract_points

Samples a list of GRASS rasters using a point dataset

#### Parameters
vect_name : str

    Name of GRASS GIS vector containing point features.

fields : list, str

    Name of attribute(s) containing the vect_name variable(s).

na_rm : bool (opt). Default is True

    Whether to remove samples containing NaNs.

as_df : bool (opt). Default is False.

    Whether to return the extracted RasterStack values as a Pandas
    DataFrame.

#### Returns
X : ndarray

    2d array containing the extracted raster values with the dimensions
    ordered by (n_samples, n_features).

y : ndarray

    1d or 2d array of labels with the dimensions ordered by 
    (n_samples, n_fields).

cat : ndarray

    1d array of category indices of the GRASS vector map.

df : pandas.DataFrame

    Extracted raster values as Pandas DataFrame if as_df = True.

### RasterStack.to_pandas

RasterStack to a pandas.DataFrame

#### Returns

pandas.DataFrame

### RasterStack.head

Show the head (first rows, first columns) or tail (last rows, last columns) of the cells of a Raster object

### RasterStack.tail

Show the head (first rows, first columns) or tail (last rows, last columns) of the cells of a Raster object
