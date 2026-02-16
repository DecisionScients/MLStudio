# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_regression copy.py                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Monday, July 20th 2020, 1:57:34 pm                          #
# Last Modified : Monday, July 20th 2020, 1:57:34 pm                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Test estimator validation."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories import tasks, algorithms

# --------------------------------------------------------------------------  #
@mark.validate_estimator
def test_estimator_validation(get_regression_data):
    X, y = get_regression_data
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory()
        est.fit(X,y)
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(task=tasks.Task.linear_regression_factory(),eta0='hat')
        est.fit(X,y)                
    with pytest.raises(ValueError):        
        est = estimators.GradientDescent.linear_regression_factory(eta0=5)
        est.fit(X,y)        
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(epochs='hat')
        est.fit(X,y)                
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(batch_size='hat')
        est.fit(X,y)        
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(val_size='hat')
        est.fit(X,y)        
    with pytest.raises(ValueError):        
        est = estimators.GradientDescent.linear_regression_factory(val_size=5)
        est.fit(X,y)                        
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(optimizer=tasks.Task.linear_regression_factory())
        est.fit(X,y)                
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(metric=tasks.Task.linear_regression_factory())
        est.fit(X,y)                
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(learning_rate=tasks.Task.linear_regression_factory())
        est.fit(X,y)                
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(observer_list=tasks.Task.linear_regression_factory())
        est.fit(X,y)                        
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(verbose=tasks.Task.linear_regression_factory())
        est.fit(X,y)                        
    with pytest.raises(TypeError):        
        est = estimators.GradientDescent.linear_regression_factory(random_state=tasks.Task.linear_regression_factory())
        est.fit(X,y)    
    with pytest.raises(ValueError):        
        X, y = None
        est = estimators.GradientDescent.linear_regression_factory()
        est.fit(X,y)                            

        
