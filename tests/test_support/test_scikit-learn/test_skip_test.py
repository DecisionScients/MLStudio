# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \__init__.py                                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 9:53:59 pm                        #
# Last Modified : Saturday, July 25th 2020, 9:53:59 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Test scikit-learn check estimator."""
import numpy as np
import pytest
from pytest import mark
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks
class SampleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def _get_tags(self):
        tags = {}
        tags['poor_score'] = True
        return tags

    def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         self.X_ = X
         self.y_ = y
         # Return the REGRESSOR
         return self

    def predict(self, X):

         # Check is fit had been called
         check_is_fitted(self)

         # Input validation
         X = check_array(X)
         self.y_ = X * 5
         return self.y_

@mark.sklearn_estimator_checks
@parametrize_with_checks([SampleRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
#%%
import sklearn; sklearn.show_versions()