#!/usr/bin/env python
# -- coding: utf-8 --

"""The module rlearn_utils contains functinons to assist
with passing pre-defined scikit learn classifiers
and other utilities for loading/saving training data."""

from __future__ import absolute_import

import grass.script as gs
import numpy as np
import os
from copy import deepcopy
from copy import deepcopy
import sqlite3
import tempfile
from grass.pygrass.modules.shortcuts import database as db
from grass.pygrass.modules.shortcuts import vector as gvect
from grass.pygrass.modules.shortcuts import raster as grast
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.vector import VectorTopo
from grass.pygrass.vector.geometry import Point


def option_to_list(x, dtype=None):
    """
    Parses a multiple choice option from into a list
    
    Args
    ----
    x : str
        String with comma-separated values
    
    dtype : func, optional
        Optionally pass a function to coerce with list elements into a
        specific type, e.g. int
    
    Returns
    -------
    x : list
        List of parsed values
    """

    if x.strip() == "":
        x = None
    elif "," in x is False:
        x = [x]
    else:
        x = x.split(",")

    if dtype:
        x = [dtype(i) for i in x]

    if x is not None:

        if len(x) == 1 and x[0] == -1:
            x = None

    return x


def predefined_estimators(estimator, random_state, n_jobs, p):
    """
    Provides the classifiers and parameters using by the module

    Args
    ----
    estimator (string): Name of scikit learn estimator
    random_state (float): Seed to use in randomized components
    n_jobs (integer): Number of processing cores to use
    p (dict): Classifier setttings (keys) and values

    Returns
    -------
    clf (object): Scikit-learn classifier object
    mode (string): Flag to indicate whether classifier performs classification
        or regression
    """
    try:
        from sklearn.experimental import enable_hist_gradient_boosting
    except ImportError:
        pass

    from sklearn.linear_model import (
        LogisticRegression,
        LinearRegression,
        SGDRegressor,
        SGDClassifier,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    )
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    estimators = {
        "SVC": SVC(C=p["C"], probability=True, random_state=random_state),
        "SVR": SVR(C=p["C"], epsilon=p["epsilon"]),
        "LogisticRegression": LogisticRegression(
            C=p["C"],
            solver="liblinear",
            random_state=random_state,
            multi_class="auto",
            n_jobs=1,
            fit_intercept=True,
        ),
        "LinearRegression": LinearRegression(n_jobs=n_jobs, fit_intercept=True),
        "SGDClassifier": SGDClassifier(
            penalty=p["penalty"],
            alpha=p["alpha"],
            l1_ratio=p["l1_ratio"],
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "SGDRegressor": SGDRegressor(
            penalty=p["penalty"],
            alpha=p["alpha"],
            l1_ratio=p["l1_ratio"],
            random_state=random_state,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=p["max_depth"],
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
        ),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=p["n_estimators"],
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True,
            oob_score=True,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=p["n_estimators"],
            max_features=p["max_features"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=random_state,
            bootstrap=True,
            n_jobs=n_jobs,
            oob_score=True,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"],
            max_features=p["max_features"],
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"],
            max_features=p["max_features"],
            random_state=random_state,
        ),
        "HistGradientBoostingClassifier": GradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"],
            max_features=p["max_features"],
            random_state=random_state,
        ),
        "HistGradientBoostingRegressor": GradientBoostingRegressor(
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"],
            max_features=p["max_features"],
            random_state=random_state,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=p["hidden_layer_sizes"],
            alpha=p["alpha"],
            random_state=random_state,
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=p["hidden_layer_sizes"],
            alpha=p["alpha"],
            random_state=random_state,
        ),
        "GaussianNB": GaussianNB(),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
        "KNeighborsClassifier": KNeighborsClassifier(
            n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=n_jobs
        ),
        "KNeighborsRegressor": KNeighborsRegressor(
            n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=n_jobs
        ),
    }

    # define classifier
    model = estimators[estimator]

    # classification or regression
    if (
        estimator == "LogisticRegression"
        or estimator == "SGDClassifier"
        or estimator == "MLPClassifier"
        or estimator == "DecisionTreeClassifier"
        or estimator == "RandomForestClassifier"
        or estimator == "ExtraTreesClassifier"
        or estimator == "GradientBoostingClassifier"
        or estimator == "HistGradientBoostingClassifier"
        or estimator == "GaussianNB"
        or estimator == "LinearDiscriminantAnalysis"
        or estimator == "QuadraticDiscriminantAnalysis"
        or estimator == "SVC"
        or estimator == "KNeighborsClassifier"
    ):
        mode = "classification"
    else:
        mode = "regression"

    return (model, mode)


def check_class_weights():

    support = [
        "LogisticRegression",
        "SGDClassifier",
        "GaussianNB",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "SVC",
    ]

    return support


def scoring_metrics(mode):
    """
    Simple helper function to return a suite of scoring methods depending on
    if classification or regression is required
    
    Args
    ----
    mode : str
        'classification' or 'regression'
    
    Returns
    -------
    scoring : list
        List of sklearn scoring metrics
    search_scorer : func, or str
        Scoring metric to use for hyperparameter tuning
    """

    from sklearn import metrics
    from sklearn.metrics import make_scorer

    if mode == "classification":
        scoring = {
            "accuracy": metrics.accuracy_score,
            "balanced_accuracy": metrics.balanced_accuracy_score,
            "matthews_correlation_coefficient": metrics.matthews_corrcoef,
            "kappa": metrics.cohen_kappa_score,
        }

        search_scorer = make_scorer(metrics.matthews_corrcoef)

    else:
        scoring = {
            "r2": metrics.r2_score,
            "explained_variance": metrics.explained_variance_score,
            "mean_absolute_error": metrics.mean_absolute_error,
            "mean_squared_error": metrics.mean_squared_error,
        }

        search_scorer = make_scorer(metrics.r2_score)

    return (scoring, search_scorer)


def save_training_data(file, X, y, cat, groups=None, names=None):
    """
    Saves any extracted training data to a csv file.
    
    Training data is saved in the following format:
        col (0..n) : feature data
        col (n) : response data
        col (n+1): grass cat value
        col (n+2): group idx

    Parameters
    ----------
    file : str
        Path to a csv file to save data to
    X : ndarray
        2d numpy array containing predictor values
    y : ndarray
        1d numpy array containing labels
    cat : ndarray
        1d numpy array of GRASS key column
    groups :ndarray (opt)
        1d numpy array containing group labels
    names : list (opt)
        Optionally pass names of features to use as a heading
    """
    if isinstance(names, str):
        names = list(names)

    if names is None:
        names = ["feature" + str(i) for i in range(X.shape[1])]

    names = ",".join(names + ["response", "cat", "groups"])

    # if there are no group labels, create a nan filled array
    if groups is None:
        groups = np.empty((y.shape[0]))
        groups[:] = np.nan

    training_data = np.column_stack([X, y, cat, groups])

    np.savetxt(fname=file, X=training_data, delimiter=",", header=names, comments="")


def load_training_data(file):
    """
    Loads training data and labels from a csv file

    Args
    ----
    file (string): Path to a csv file to save data to

    Returns
    -------
    X (2d numpy array): Numpy array containing predictor values
    y (1d numpy array): Numpy array containing labels
    cat (1d numpy array): Numpy array of GRASS key column
    groups (1d numpy array): Numpy array of group labels, or None
    """

    training_data = np.loadtxt(file, delimiter=",", skiprows=1)
    n_cols = training_data.shape[1]
    last_Xcol = n_cols - 3

    # group_ids stored in last column
    groups = training_data[:, -1]

    if bool(np.isnan(groups).all()) is True:
        groups = None

    # cat stored in 2nd last column
    cat = training_data[:, -2]

    # response stored in 3rd last column
    y = training_data[:, -3]
    X = training_data[:, 0:last_Xcol]

    return (X, y, cat, groups)


def grass_read_vect_sql(vect):
    """
    Read a GRASS GIS vector map containing point geometries into a geopandas
    GeoDataFrame

    Currently only Point geometries are supported

    Args
    ----
    vect : str
        Name of GRASS GIS vector map
    
    Returns
    -------
    geopandas.GeoDataFrame
    """

    try:
        from shapely.geometry import Point
        import pandas as pd
        import geopandas as gpd
    except ImportError:
        gs.fatal(
            "Reading GRASS GIS point data into geopandas requires the ",
            "shapely and geopandas python packages to be installed",
        )

    with VectorTopo(vect) as points:
        df = pd.read_sql_query(
            sql="select * from {vect}".format(vect=vect), con=points.table.conn
        )

        coords = [(p.cat, deepcopy(p.coords())) for p in points]
        coords = pd.DataFrame(coords, columns=["cat", "geometry"])

        df = df.join(other=coords, how="right", rsuffix=".y")

        df.geometry = [Point(p) for p in df.geometry]
        df = gpd.GeoDataFrame(df)
        df.crs = {"init": "epsg:3400"}

    df = df.drop(columns=["cat.y"])

    return df


def grass_write_vect_sql(gpdf, x="x_crd", y="y_crd", output=None, overwrite=False):
    """
    Write a geopandas.GeodataFrame of Point geometries into a GRASS GIS
    vector map

    Currently only point geometries are supported

    Args
    ----
    gpdf : geopandas.GeoDataFrame
        Containing point geometries

    x, y : str
        Name of coordinate fields to use in GRASS table
    
    output : str
        Name of output GRASS GIS vector map
    """

    if overwrite is True:
        try:
            db.droptable(table=output + "_table", flags="f")
            g.remove(name=output, type="vector", flags="f")
        except:
            pass

    sqlpath = gs.read_command("db.databases", driver="sqlite").strip(os.linesep)
    con = sqlite3.connect(sqlpath)

    gpdf[x] = gpdf.geometry.bounds.iloc[:, 0]
    gpdf[y] = gpdf.geometry.bounds.iloc[:, 1]

    (
        gpdf.drop(labels=["geometry"], axis=1).to_sql(
            output + "_table", con, index=True, index_label="cat", if_exists="replace"
        )
    )

    con.close()

    gvect.in_db(table=output + "_table", x=x, y=y, output=output, flags="t", key="cat")


def grass_read_vect(vect):
    try:
        import geopandas as gpd
    except ImportError:
        gs.fatal("Geopandas python package is required")

    temp_out = ".".join([tempfile.NamedTemporaryFile().name, "gpkg"])
    gvect.out_ogr(input=vect, output=temp_out, format="GPKG")

    return gpd.read_file(temp_out)


def grass_write_vect(gpdf, output, overwrite=False, flags=""):
    try:
        import geopandas as gpd
    except ImportError:
        import geopandas as gpd

    temp_out = ".".join([tempfile.NamedTemporaryFile().name, "gpkg"])
    gpdf.to_file(temp_out, driver="GPKG")

    gvect.in_ogr(input=temp_out, output=output, overwrite=overwrite, flags=flags)


def euclidean_distance_fields(prefix, region, overwrite=False):

    point_topleft = Point(
        region.west + region.ewres / 2, region.north - region.nsres / 2
    )
    point_topright = Point(
        region.east - region.ewres / 2, region.north - region.nsres / 2
    )
    point_lowerleft = Point(
        region.west + region.ewres / 2, region.south + region.nsres / 2
    )
    point_lowerright = Point(
        region.east - region.ewres / 2, region.south + region.nsres / 2
    )
    point_centre = Point(
        region.west + (region.east - region.west) / 2,
        region.south + (region.north - region.south) / 2,
    )

    points = {
        "topleft": point_topleft,
        "topright": point_topright,
        "lowerleft": point_lowerleft,
        "lowerright": point_lowerright,
        "centre": point_centre,
    }

    for name, p in points.items():

        point_name = "_".join([prefix, name])

        vect = VectorTopo(name=point_name)
        vect.open(
            mode="w",
            tab_name=point_name,
            tab_cols=[("cat", "INTEGER PRIMARY KEY"), ("name", "TEXT")],
        )
        vect.write(p, ("point",))
        vect.table.conn.commit()
        vect.close()

        gvect.to_rast(
            input=point_name,
            type="point",
            use="val",
            output=point_name,
            overwrite=overwrite,
        )
        grast.grow_distance(
            point_name, distance="distance_to_" + point_name, overwrite=overwrite
        )

        g.remove(name=point_name, type="raster", flags="f")
        g.remove(name=point_name, type="raster", flags="f")

