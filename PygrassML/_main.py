from __future__ import absolute_import, print_function
import os, sys
import subprocess
import tempfile
import shutil
import binascii
import numpy as np
import grass.script as gs
from grass.pygrass.raster import RasterRow
from grass.pygrass.gis.region import Region
from grass.pygrass.raster import numpy2raster


"""
Ctypes cannot update once python is already running, so the environmental
variable to the library that is to be loaded needs to be set first:

    export LD_LIBRARY_PATH=`grass72 --config path`/lib
    python your_script.py
"""

if sys.platform.startswith('win'):
    grass7bin = 'C:\\OSGeo4W64\\bin\\grass72.bat'
if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    grass7bin = '/usr/local/bin/grass72'
    
class gsession():

    def __init__(self, grass7bin):

        """
        Initiates a GRASS GIS 7 session
        based on:
        https://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly

        Args
        ----
            grass7bin: path to GRASS GIS executable
        """

        # Query GRASS itself for its GISBASE
        self.grass7bin = grass7bin
        startcmd = [self.grass7bin, '--config', 'path']
        startcmd = ['grass72', '--config', 'path']
        p = subprocess.Popen(
            startcmd, shell=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print (sys.stderr,
                   "ERROR: Cannot find GRASS GIS 7 start script ({s})"
                   .format(s=startcmd))
            sys.exit(-1)

        # Set GISBASE environment variable
        self.gisbase = out.strip(os.linesep)
        os.environ['GISBASE'] = self.gisbase

        # Add path to GRASS addons
        if sys.platform.startswith(('linux', 'darwin')):
            home = os.path.join(os.path.expanduser("~"), '.grass7')
        elif sys.platform.startswith('win'):
            home = os.path.join(
                os.path.expanduser("~"), 'AppData', 'Roaming', 'GRASS7')
        else:
            raise OSError('Platform not configured.')

        os.environ['PATH'] += os.pathsep + os.path.join(
                home, 'addons', 'scripts')
        os.environ['GRASS_ADDON_BASE'] = os.path.join(home, 'addons')
        os.environ['PATH'] += os.pathsep + os.path.join(
                os.environ.get('GRASS_ADDON_BASE'), 'bin')

        # Define GRASS-Python environment
        gpydir = os.path.join(self.gisbase, "etc", "python")
        sys.path.append(gpydir)

        if sys.platform == 'win32':
            os.environ['GRASS_SH'] = os.path.join(self.gisbase, 'msys', 'bin', 'sh.exe')
            os.environ['GRASS_PYTHON'] = os.path.join(
                os.environ.get('OSGEO4W_ROOT'), 'bin', 'python.exe')
            os.environ['PYTHONHOME'] = os.path.join(
                os.environ.get('OSGEO4W_ROOT'), 'apps', 'Python27')
            os.environ['GRASS_PROJSHARE'] = os.path.join(
                    os.environ.get('OSGEO4W_ROOT'), 'share', 'proj')
        else:
            os.environ['GRASS_PYTHON'] = sys.executable

        # Language
        os.environ['LANG'] = 'en_US'
        os.environ['LOCALE'] = 'C'


    def open_session(
        self, gisdb = None, location = None, mapset = 'PERMANENT', src = 4326):

        """
        Opens a GRASS GIS 7 session at a specified location or mapset

        Args
        ----
           gisdb: Path to existing grass gis database
           location: Name of grass location
           mapset: Name of grass mapset
           src: EPSG code or path to georeferenced raster/vector used to
                define a new grass session
        """

        self.gisdb = gisdb
        self.location = location
        self.src = src

        # define new location using epsg or existing vector/raster
        if type(self.src) == int:
            # Assume epsg code
            location_seed = "EPSG:{}".format(self.src)
        else:
            # Assume georeferenced vector or raster
            location_seed = self.src

        # optionally create a throwaway location
        if not self.gisdb:
            self.gisdb = tempfile.mkdtemp()
        if not self.location:
            string_length = 16
            self.location = binascii.hexlify(os.urandom(string_length))
            mapset   = 'PERMANENT'
            location_path = os.path.join(self.gisdb, self.location)
            startcmd = "{bin} -c {src} -e {loc}".format(
                    bin=self.grass7bin, src=location_seed, loc=location_path)
            print (startcmd)
            p = subprocess.Popen(
                    startcmd, shell=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            out, err = p.communicate()
            if p.returncode != 0:
                raise Exception(
                    "ERROR: GRASS GIS 7 start script ({})".format(startcmd))
            else:
                print ('Created location {s}'.format(s=location_path))

        # set GISDBASE environment variable
        os.environ['GISDBASE'] = self.gisdb

        # enter the location and mapset
        import grass.script as gs
        import grass.script.setup as gsetup
        self.rcfile = gsetup.init(
                dbase=self.gisdb, gisbase=self.gisbase, location=self.location,
                mapset=mapset)
        #from grass.lib.date import DateTime

        # say hello
        print ('--- GRASS GIS 7: Current GRASS GIS 7 environment:')
        print (gs.gisenv())


    def close_session(self):
        # Finally remove the temporary batch location from disk
        print ('Removing location %s' % os.path.join(
                self.gisdb, self.location))
        shutil.rmtree(os.path.join(self.gisdb, self.location))
        os.remove(self.rcfile)


def gvect_to_gpd(gvector, type='Point'):
    """
    Converts a GRASS vector to a geopandas object

    Args
    ----
        gvector: Name of GRASS GIS vector
        type: Geometry type (Point, Line, Polygon)

    Returns
    -------
        new_gpd: Created geopandas object
    """

    import pandas as pd
    import sqlite3
    import geopandas as gpd
    from shapely.geometry import Point

    vec = VectorTopo(gvector)
    vec.open('r')
    geom = [i.coords() for i in vec]

    if type == 'Point':
        gpd_geom = [Point(xy) for xy in geom]

    # Read in the attribute table
    sqlpath = gs.read_command("db.databases", driver="sqlite").strip(os.linesep)
    con = sqlite3.connect(sqlpath)
    df = pd.read_sql_query("select * from {table}".format(table=gvector), con)
    con.close()

    new_gpd = gpd.GeoDataFrame(df, geometry=gpd_geom)
    new_gpd.crs = {'init': 'epsg:{code}'.format(code=vec.proj)}
    vec.close()

    return(new_gpd)


def gpd_to_gvect(gdf, gvector):
    from grass.pygrass.vector import VectorTopo
    from grass.pygrass.vector.geometry import Point as gPoint

    new = VectorTopo(gvector)
    cols = [(u'cat', 'INTEGER PRIMARY KEY'),
            (u'name', 'TEXT')]

    new.open('w', tab_name='test', tab_cols=cols, overwrite=True)

    for i in gdf:
        gpoint_data = gPoint(gdf.geometry.bounds.iloc[i,0],gdf.geometry.bounds.iloc[i,1])
        new.write(gpoint_data, cat=i, attrs=('pub',))
    new.table.conn.commit()
    new.table.execute().fetchall()
    new.close()


def predict(estimator, predictors, output, predict_type='raw', index=None,
            class_labels=None, overwrite=False, rowincr=25, n_jobs=-2):
    """
    Prediction on list of GRASS rasters using a fitted scikit learn model

    Args
    ----
    estimator (object): scikit-learn estimator object
    predictors (list): Names of GRASS rasters
    output (string): Name of GRASS raster to output classification results
    predict_type (string): 'raw' for classification/regression;
        'prob' for class probabilities
    index (list): Optional, list of class indices to export
    class_labels (1d numpy array): Optional, class labels
    overwrite (boolean): enable overwriting of existing raster
    n_jobs (integer): Number of processing cores;
        -1 for all cores; -2 for all cores-1
    """

    from sklearn.externals.joblib import Parallel, delayed

    # TODO
    # better memory efficiency and use of memmap for parallel
    # processing
    #from sklearn.externals.joblib.pool import has_shareable_memory

    # first unwrap the estimator from any potential pipelines or gridsearchCV
    if type(estimator).__name__ == 'Pipeline':
       clf_type = estimator.named_steps['classifier']
    else:
        clf_type = estimator

    if type(clf_type).__name__ == 'GridSearchCV' or \
    type(clf_type).__name__ == 'RandomizedSearchCV':
        clf_type = clf_type.best_estimator_

    # check name against already multithreaded classifiers
    if type(clf_type).__name__ in [
       'RandomForestClassifier',
        'RandomForestRegressor',
        'ExtraTreesClassifier',
        'ExtraTreesRegressor',
        'KNeighborsClassifier',
        'LGBMClassifier',
        'LGBMRegressor']:
       n_jobs = 1

    # convert potential single index to list
    if isinstance(index, int): index = [index]

    # open predictors as list of rasterrow objects
    current = Region()

    # create lists of row increments
    row_mins, row_maxs = [], []
    for row in range(0, current.rows, rowincr):
        if row+rowincr > current.rows:
            rowincr = current.rows - row
        row_mins.append(row)
        row_maxs.append(row+rowincr)

    # perform predictions on lists of row increments in parallel
    prediction = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(__predict_parallel2)
        (estimator, predictors, predict_type, current, row_min, row_max)
        for row_min, row_max in zip(row_mins, row_maxs))
    prediction = np.vstack(prediction)

#    # perform predictions on lists of rows in parallel
#    prediction = Parallel(n_jobs=n_jobs, max_nbytes=None)(
#        delayed(__predict_parallel)
#        (estimator, predictors, predict_type, current, row)
#        for row in range(current.rows))
#    prediction = np.asarray(prediction)

    # determine raster dtype
    if prediction.dtype == 'float':
        ftype = 'FCELL'
    else:
        ftype = 'CELL'

    #  writing of predicted results for classification
    if predict_type == 'raw':
        numpy2raster(array=prediction, mtype=ftype, rastname=output,
                     overwrite=True)

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
    Performs prediction on a single row of a GRASS raster(s))

    Args
    ----
    estimator (object): Scikit-learn estimator object
    predictors (list): Names of GRASS rasters
    predict_type (string): 'raw' for classification/regression;
        'prob' for class probabilities
    current (dict): current region settings
    row (integer): Row number to perform prediction on

    Returns
    -------
    result (2d oe 3d numpy array): Prediction results
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

    return result


def __predict_parallel2(estimator, predictors, predict_type, current, row_min, row_max):
    """
    Performs prediction on range of rows in grass rasters

    Args
    ----
    estimator: scikit-learn estimator object
    predictors: list of GRASS rasters
    predict_type: character, 'raw' for classification/regression;
                  'prob' for class probabilities
    current: current region settings
    row_min, row_max: Range of rows of grass rasters to perform predictions

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
    img_np_row = np.zeros((row_max-row_min, current.cols, n_features))
    for row in range(row_min, row_max):
        for band in range(n_features):
            img_np_row[row-row_min, :, band] = np.array(rasstack[band][row])

    # create mask
    img_np_row[img_np_row == -2147483648] = np.nan
    mask = np.zeros((img_np_row.shape[0], img_np_row.shape[1]))
    for feature in range(n_features):
        invalid_indexes = np.nonzero(np.isnan(img_np_row[:, :, feature]))
        mask[invalid_indexes] = np.nan

    # reshape each row-band matrix into a n*m array
    nsamples = (row_max-row_min) * current.cols
    flat_pixels = img_np_row.reshape((nsamples, n_features))

    # remove NaNs prior to passing to scikit-learn predict
    flat_pixels = np.nan_to_num(flat_pixels)

    # perform prediction for classification/regression
    if predict_type == 'raw':
        result = estimator.predict(flat_pixels)
        result = result.reshape((row_max-row_min, current.cols))

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
        result = result.reshape((row_max-row_min, current.cols, result.shape[1]))
        result[np.nonzero(np.isnan(mask))] = np.nan

    # close maps
    for i in range(n_features):
        rasstack[i].close()

    return result
