# Machine Learning in GRASS GIS using Python (r.learn.ml2)

This is python module for applying scikit-learn machine learning models to GRASS GIS spatial data.

Documentation at https://stevenpawley.github.io/r.learn.ml2/

## Contents

* [Description](#description)
* [Installation](#installation)
* [Example usage as a GRASS addon](#Example-using-GRASS-GIS-command-line)
* [Quickstart using Python scripting](#Quickstart-using-Python-scripting)

## Description

r.learn.ml2 represents a front-end to the scikit learn python package. The module enables scikit-learn classification and regression models to be applied to GRASS GIS rasters that are stored as part of an imagery group.

The training component of the machine learning workflow is performed using the `r.learn.train` module. This module uses training data consisting of labelled pixels in a GRASS GIS raster map, or a GRASS GIS vector containing points, and develops a machine learning model on the rasters within a GRASS imagery group.

The `r.learn.predict` module needs to be called, which will retrieve the saved and pre-fitted model and apply it to a GRASS GIS imagery group.

## Installation

### Stable version available within GRASS GIS

To get the stable of this module, first install GRASS GIS (http://grass.osgeo.org/) and then install r.learn.ml2 module from GRASS Addons using either the GUI (Settings -> Addons extensions -> Install extension from addons) or the following command:

```
g.extension r.learn.ml2
```

The GRASS GIS module requires two additional python packages, scikit-learn (http://scikit-learn.org/stable) and pandas (https://pandas.pydata.org), which need to be installed within the GRASS GIS Python environment. For Linux users, these packages should be available through the linux package manager. For MS-Windows users the easiest way of installing the packages is by using the precompiled binaries from <a href="http://www.lfd.uci.edu/~gohlke/pythonlibs/">Christoph Gohlke</a> and by using the <a href="https://grass.osgeo.org/download/software/ms-windows/">OSGeo4W</a> installation method of GRASS, where the python3-pip can also be installed. Then, you can download the NumPy+MKL and the scikit-learn .whl' files.

### Development version (this repository)

This repository is the primary location for development of the modules behind the r.learn.ml2 addon. This repository contains the newest and experimental features, and also provides documentation for using the modules within the add-on for scripting.

The development version can also be installed using `g.extension` on Linux and Mac OS.

```
g.extension extension=r.learn.ml2 url=https://github.com/stevenpawley/r.learn.ml2
```

## Example using GRASS GIS command line

Here we are going to use the GRASS GIS sample North Carolina data set as a basis to perform a landsat classification. We are going to classify a Landsat 7 scene from 2000, using training information from an older (1996) land cover dataset.

Note that this example must be run in the "landsat" mapset of the North Carolina sample data set location.

Plot a landsat 7 (2000) bands 7,4,2 color composite:

```
i.colors.enhance red=lsat7_2000_70 green=lsat7_2000_50 blue=lsat7_2000_20
d.rgb red=lsat7_2000_70 green=lsat7_2000_50 blue=lsat7_2000_20
```

![](lsat7_2000_b742.png)

Generate some training pixels from an older (1996) land cover classification:

```
g.region raster=landclass96 -p
r.random input=landclass96 npoints=1000 raster=training_pixels
```

<p>Then we can use these training pixels to perform a classification on the more recently obtained landsat 7 image:</p>

```
r.learn.train group=lsat7_2000 training_map=training_pixels \
	model_name=RandomForestClassifier n_estimators=500 save_model=rf_model.gz

r.learn.predict group=lsat7_2000 load_model=rf_model.gz output=rf_classification
```

Now display the results:

```
# copy color scheme from landclass training map to result
r.colors rf_classification raster=training_pixels

# plot the results
d.rast rf_classification
```

![](rfclassification.png)

## Quickstart using Python scripting

### Importing the modules

Providing that r.learn.ml2 is installed as a GRASS GIS addon, the python modules can be imported directly using:

```
# add the r.learn.ml2 addon to path
import sys
from grass.script.utils import get_lib_path
path = get_lib_path("r.learn.ml2")
sys.path.append(path)

# import the addon's modules
from raster import RasterStack
```

### The RasterStack class

#### Initiation

The main module in r.learn.ml2 is the `RasterStack` class. A RasterStack can be initiated using a list of GRASS GIS raster maps:

```
stack = RasterStack(rasters=["lsat7_2002_10", "lsat7_2002_20", "lsat7_2002_30", "lsat7_2002_40"])
```

Alternatively, it can be initiated using a GRASS imagery group:

```
stack = RasterStack(group="landsat_2002")
```

#### Indexing of RasterStack objects

Individual rasters within a `RasterStack` can be accessed using several methods:

```
stack.names  # returns names of rasters

# methods that return RasterRow objects
stack.lsat7_2002_10  # use attribute name directly

stack.iloc[0]  # access by integer index

stack.iloc[0:2]  # access using slices

stack.loc["lsat7_2002_10"]  # access using a label, or list of labels

# methods that always return a new RasterStack object
stack["lsat7_2002_10"]
```

Individual rasters within the `RasterStack` can be set using:

```
from grass.pygrass.raster import RasterRow

# set layers using a single index
stack.iloc[0] = RasterRow("lsat7_2002_61") 

# set layers using a multiple indexes
stack.iloc[[0, 1]] = [RasterRow("lsat7_2002_70"), RasterRow("lsat7_2002_80")]

# set layers using a slice of indexes
stack.iloc[0:2] = [RasterRow("lsat7_2002_70"), RasterRow("lsat7_2002_80")]

# set layers using a single label
stack.loc["lsat7_2002_10"] = RasterRow("lsat7_2002_61")

# set layers using multiple labels
stack.loc[["lsat7_2002_10", "lsat7_2002_20"]] = [RasterRow("lsat7_2002_61"), RasterRow("lsat7_2002_62")]
```

#### Viewing data with a RasterStack

Quick views of the values of the rasters within a `RasterStack` object can be generated by:

```
stack = RasterStack(rasters=["lsat7_2002_10", "lsat7_2002_20", "lsat7_2002_30", "lsat7_2002_40"])

# view data from the first 10 rows
stack.head()

# view data from the last 10 rows
stack.tail()

# convert raster to pandas dataframe
stack.to_pandas()
```

#### Reading array data from a RasterStack

Data from a RasterStack can be read into a 3D numpy array using the `read` method.
The data is returned as a masked array with the GRASS GIS null values for each
raster value masked.

```
# read all data (obeying the computational window settings)
stack.read()

# read a single row
stack.read(row=1)

# read a set of rows in a contiguous interval (start, end)
stack.read(rows=(1, 10))
```

#### Extracting data from a RasterStack

Pixel values can be spatially-queried in the RasterStack using either another
raster containing labelled pixels via the `extract_pixels` method, or a GRASS 
GIS vector map containing point geomeries using the `extract_points` method.
Either method can return the extracted data as three numpy arrays, or as a pandas
dataframe.


When extracting data using another raster map, `X` will be a 3D numpy array containing
the extracted data from the RasterStack, `y` will be a 1D numpy array containing
the values of the pixels in the labelled pixels map, and `cat` is the index value
of the pixels.

```
# extract data using another raster 
X, y, cat = stack.extract_pixels(response="labelled_pixels")

# extract data using another raster, and returning the GRASS raster categories
# instead of integer values
X, y, cat = stack.extract_pixels(response="labelled_pixels", use_cats=True)

# return data as a pandas dataframe
df = stack.extract_pixels(rast_name="labelled_pixels", as_df=True)
```

When extracting data using a vector map, the `fields` parameter refers to the 
name of an attribute, or several attributes in the `vect_name` map to returned
with the extracted raster data. If several attributes are used the `y` will be 
a 2D numpy array.

```
# basic use
X, y, cat = stack.extract_points(vect_name="points_map", field="slope")
X, y, cat = stack.extract_points(vect_name="points_map"), field=["slope", "aspect"]

# as pandas
df = stack.extract_points(vect_name="points_map", field="slope", as_df=True)
```

By default, rows containing null values in any of the rasters are removed. This can
be disabled by using `na_rm=False`:

```
df = stack.extract_points(vect_name="points_map", field="slope", as_df=True, na_rm=True)
```

#### Applying a machine learning model to data within a RasterStack

Any scikit-learn compatible model that has a `predict` method can be applied to
the data within a RasterStack. The following provides a brief example within the
nc_spm_08 sample GRASS location:

```
from grass.pygrass.modules.shortcuts import raster as r

# generate some training data from another land use map
r.random(input="landclass96", npoints=1000, raster="training_pixels")

# create a stack of landsat data
stack = RasterStack(
    rasters=[
        "lsat7_2002_10", 
        "lsat7_2002_20", 
        "lsat7_2002_30", 
        "lsat7_2002_40", 
        "lsat7_2002_50", 
        "lsat_2002_70"
    ]
)

# extract training data
X, y, cat = stack.extract_pixels(rast_name="training_pixels")

# fit a ml model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)

# apply fitted model to RasterStack and returns a RasterStack
preds = stack.predict(
    estimator=rf,                # fitted model
    output="rf_classification",  # name of output GRASS raster
    height=25,                   # number of rows to predict in chunks
    overwrite=False
)

probs = stack.predict_proba(
    estimator=rf,
    output="rf_classification",
    height=25,
    overwrite=False
)
```

Multi-target prediction is also allowed for scikit-learn models which accept
multiple target features.
