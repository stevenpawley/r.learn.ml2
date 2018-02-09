# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 23:07:46 2018

@author: Steven Pawley
"""

from sklearn.ensemble import RandomForestClassifier

rasters = ['L8_1314_Backfilled.1', 'L8_1314_Backfilled.2', 'L8_1314_Backfilled.3', 'MRVBF_lidar15']

stack = RasterStack(rasters)

X, y, cds = stack.extract_features(
    vect_name='TRAINING_Permafrost_mapped_4k_samples@permafrost',
    field='id', na_rm=True)

X, y, crds = stack.extract_pixels(
    response='TRAINING_Permafrost_mapped_4k_samples@permafrost', na_rm=True)
X.shape
y.shape
crds.shape

lr = RandomForestClassifier(n_estimators=100, n_jobs=-1)
lr.fit(X, y)

stack.predict_proba(estimator=lr, output='test', overwrite=True)
