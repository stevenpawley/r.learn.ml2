#!/usr/bin/env python3

"""
MODULE:    Test of r.learn.train

AUTHOR(S): Steven Pawley <dr.stevenpawley gmail com>

PURPOSE:   Test of r.learn.train

COPYRIGHT: (C) 2020 by Steven Pawley and the GRASS Development Team

This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.
"""
import tempfile
import os

import grass.script as gs

from grass.gunittest.case import TestCase
from grass.gunittest.main import test


class TestExecution(TestCase):
    """Test learning and prediction using r.learn.ml"""

    # Raster maps be used as inputs (they exist in the NC SPM sample location)
    classif_map = "landclass96@PERMANENT"
    band1 = "lsat7_2002_10@PERMANENT"
    band2 = "lsat7_2002_20@PERMANENT"
    band3 = "lsat7_2002_30@PERMANENT"
    band4 = "lsat7_2002_40@PERMANENT"
    band5 = "lsat7_2002_50@PERMANENT"
    band7 = "lsat7_2002_70@PERMANENT"
    
    # Data that is generated and is required for the tests
    output = "classification_result"
    model_file = tempfile.NamedTemporaryFile(suffix='.gz').name
    group = "predictors"
    labelled_pixels = "landclass96_roi"
    labelled_points = "landclass96_roi_points"

    @classmethod
    def setUpClass(cls):
        """Setup that is required for all tests"""
        
        # Use temporary computational region
        cls.use_temp_region()
        cls.runModule("g.region", raster=cls.classif_map)
        
        # Create an imagery group of raster predictors
        cls.runModule(
            "i.group", 
            group=cls.group,
            input=[cls.band1, cls.band2, cls.band3, cls.band4, cls.band5, cls.band7]
        )
        
        # Generate some training data from a previous classification map
        cls.runModule("r.random", input=cls.classif_map, npoints=1000, raster=cls.labelled_pixels,
                      seed=1234)
        cls.runModule("r.to.vect", input=cls.labelled_pixels, output=cls.labelled_points,
                      type='point')
        
    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region (and anything else we created)"""
        
        cls.del_temp_region()
        cls.runModule("g.remove", flags="f", type="raster", name=cls.labelled_pixels)
        cls.runModule("g.remove", flags="f", type="vector", name=cls.labelled_points)
        cls.runModule("g.remove", flags="f", type="group", name=cls.group)

    def tearDown(self):
        """Remove the output created from the tests
        (reuse the same name for all the test functions)"""
        self.runModule("g.remove", flags="f", type="raster", name=[self.output])
        
        try:
            os.remove(self.model_file)
        except FileNotFoundError:
            pass
            
    def test_output_created_labelled_pixels(self):
        """Checks that the output is created"""
                
        # test r.learn.train using pixels
        self.assertModule(
            "r.learn.train",
            group=self.group,
            training_map=self.labelled_pixels,
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        self.assertFileExists(filename=self.model_file)
        
        # test prediction exists
        self.assertModule(
            "r.learn.predict",
            group=self.group,
            load_model=self.model_file,
            output=self.output
        )
        self.assertRasterExists(self.output, msg="Output was not created")
        
    def test_output_created_prediction_points(self):
        """Checks that output is created"""

        # training
        self.assertModule(
            "r.learn.train",
            group=self.group,
            training_points=self.labelled_points,
            field='value',
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        self.assertFileExists(filename=self.model_file)
        
        # prediction
        self.assertModule(
            "r.learn.predict",
            group=self.group,
            load_model=self.model_file,
            output=self.output
        )
        self.assertRasterExists(self.output, msg="Output was not created")
    
    def test_train_missing_parameter(self):
        """Check that module fails when parameters are missing"""
        
        # missing group
        self.assertModuleFail(
            "r.learn.train",
            training_points=self.labelled_points,
            field='value',
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        
        # missing any training data
        self.assertModuleFail(
            "r.learn.train",
            group=self.group,
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        
        # using points as input but no field specified
        self.assertModuleFail(
            "r.learn.train",
            group=self.group,
            training_points=self.labelled_points,
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        
        # using both points and labelled pixels
        self.assertModuleFail(
            "r.learn.train",
            group=self.group,
            training_points=self.labelled_points,
            training_map=self.labelled_pixels,
            model_name="RandomForestClassifier",
            n_estimators=100,
            save_model=self.model_file
        )
        
        # no output
        self.assertModuleFail(
            "r.learn.train",
            group=self.group,
            training_points=self.labelled_points,
            model_name="RandomForestClassifier",
            n_estimators=100
        )

if __name__ == "__main__":
    test()
