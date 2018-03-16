#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:32:03 2018

@author: steven
"""

stack = RasterStack(['lsat7_2002_10', 'lsat7_2002_20'])
X, y, cds = stack.extract_features(vect_name='landclass96_roi@PERMANENT', field='id', na_rm=True)
X, y, cds = stack.extract_pixels('landclass96_roi@PERMANENT')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)

stack.predict(estimator=lr, output='test', overwrite=True)
stack.predict_proba(estimator=lr, output='test1', overwrite=True)