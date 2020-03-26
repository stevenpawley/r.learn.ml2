# Machine Learning in GRASS GIS using Python (r.learn.ml2)

This is python module for applying scikit-learn machine learning models to GRASS GIS spatial data.

## Contents

* [Description](#description)
* [Installation](#installation)
* [Example usage as a GRASS addon](#Example-using-GRASS-GIS-command-line)
* [Quickstart using Python scripting](python_quickstart.md)

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
