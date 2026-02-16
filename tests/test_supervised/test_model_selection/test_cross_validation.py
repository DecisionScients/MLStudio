# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_components.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 11th 2020, 4:54:32 am                        #
# Last Modified : Saturday, July 11th 2020, 4:54:32 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests for data analysis functions."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.metrics.regression import R2
from mlstudio.supervised.model.development import ModelBuilder
from mlstudio.IoC.loss import IoCQuadratic
from mlstudio.IoC.tasks import Tasks
from mlstudio.IoC.estimators import Estimators
# --------------------------------------------------------------------------- #
@mark.cross_validation
class CrossValidationTests:

    def test_nested_cv(self, get_regression_data_unscaled):
        X, y = get_regression_data_unscaled
        loss = IoCQuadratic().quadratic()
        task = Tasks().linear_regression(loss=loss)        
        estimator = Estimators().gradient_descent_factory(task=task, metric=R2)
        print("*************************")
        print(estimator.fit)

        param_set = [
            {"epochs": [100,200,500,1000],
             "batch_size": [32, 64]}
        ]
        ncv = ModelBuilder(estimator=estimator, parameters=param_set) 
        ncv.fit(X,y)
        print(ncv.results_.get('test_score'))
        print(ncv.results_.get('estimator'))
        


