#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:32:03 2018

@author: steven
"""

from PygrassML import RasterStack

stack = RasterStack(['lsat7_2002_10', 'lsat7_2002_20', 'lsat7_2002_30', 'lsat7_2002_40', 'lsat7_2002_50', 'lsat7_2002_70'])
X, y, cds = stack.extract_points(vect_name='landclass96_roi@PERMANENT', field='id', na_rm=True)
X, y, cds = stack.extract_pixels('landclass96_roi@PERMANENT')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

clf = RandomForestClassifier()
clf = LogisticRegression()
clf.fit(X, y)

stack.predict(estimator=clf, output='test', overwrite=True)
stack.predict_proba(estimator=clf, output='test1', overwrite=True)

import grass.script as gs
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
scores = cross_validate(clf, X, y)
y_pred = cross_val_predict(clf, X, y)
cm = confusion_matrix(y, y_pred)
cm_precision = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis] # users (precision)
cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # producers (recall)


fimp, fimp_std = permutation_importance(clf, X, y, n_jobs=-1)