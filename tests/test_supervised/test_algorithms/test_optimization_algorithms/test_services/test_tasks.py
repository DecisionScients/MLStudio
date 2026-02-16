#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_activations.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 10:24:08 pm                         #
# Last Modified : Monday, June 15th 2020, 10:24:25 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Test Activation Functions."""
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories.tasks import RegressionTasks, BinaryclassTasks
from mlstudio.factories.tasks import MulticlassTasks
from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.data_services.preprocessing import RegressionDataProcessor
from mlstudio.data_services.preprocessing import BinaryClassDataProcessor
from mlstudio.data_services.preprocessing import MultiClassDataProcessor
# --------------------------------------------------------------------------  #
@mark.tasks
@mark.linear_regression
class LinearRegressionTaskTests:

    def test_linear_regression_validation(self, get_regression_data):        
        with pytest.raises(TypeError):        
            task = Task.regression(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.regression(loss=loss.CrossEntropy())            
        with pytest.raises(TypeError):        
            task = Task.regression(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.regression(data_processor=BinaryClassDataProcessor())                        
        with pytest.raises(ValueError):        
            task = Task.regression(loss=\
                loss.Quadratic(regularizers.L1_L2(ratio=2)))                                    
        with pytest.raises(NotImplementedError):        
            task = Task.regression()
            X, y = get_regression_data
            theta = np.random.rand(X.shape[1],)
            task.predict_proba(theta, X)

    def test_linear_regression(self, get_regression_task_package):
        task = Task.regression()
        d = get_regression_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.Quadratic), "LinearRegression loss value not correct"        
        task = Task.regression(loss=loss.Quadratic(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "LinearRegression loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "LinearRegression loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, RegressionDataProcessor), "LinearRegression data processor error."
        # Test output / predict
        task = Task.regression()
        y_out = task.compute_output(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_out, y_pred), "LinearRegression output and prediction don't match."
        assert np.allclose(y_pred, d['y_pred']), "LinearRegression prediction error."


@mark.tasks
@mark.logistic_regression
class BinaryClassificationTaskTests:

    def test_logistic_regression_validation(self, get_logistic_regression_data):        
        with pytest.raises(TypeError):        
            task = Task.binaryclass(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.binaryclass(loss=loss.CategoricalCrossEntropy())            
        with pytest.raises(TypeError):        
            task = Task.binaryclass(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.binaryclass(data_processor=RegressionDataProcessor())                        
        with pytest.raises(ValueError):        
            task = Task.binaryclass(loss=loss.Quadratic(regularizer=regularizers.L1_L2(ratio=2)))                                    

    def test_logistic_regression(self, get_logistic_regression_task_package):
        task = Task.binaryclass()
        d = get_logistic_regression_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.CrossEntropy), "BinaryClassification loss value not correct"        
        task = Task.binaryclass(loss=loss.CrossEntropy(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "BinaryClassification loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "BinaryClassification loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, BinaryClassDataProcessor), "BinaryClassification data processor error."
        # Test output 
        task = Task.binaryclass()
        y_prob = task.predict_proba(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_pred, d['y_pred']), "BinaryClassification prediction error."
        assert np.allclose(y_prob, d['y_prob']), "BinaryClassification probability prediction error."



@mark.tasks
@mark.multiclass
class MultiClassificationTaskTests:

    def test_multiclass_validation(self):        
        with pytest.raises(TypeError):        
            task = Task.multiclass(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.multiclass(loss=loss.Quadratic())            
        with pytest.raises(TypeError):        
            task = Task.multiclass(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.multiclass(data_processor=RegressionDataProcessor())                        
        with pytest.raises(ValueError):        
            task = Task.multiclass(loss=loss.CategoricalCrossEntropy(regularizers.L1_L2(ratio=2)))                                    

    def test_multiclass(self, get_multiclass_task_package):
        task = Task.multiclass()
        d = get_multiclass_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.CategoricalCrossEntropy), "MultiClassification loss value not correct"        
        task = Task.multiclass(loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "MultiClassification loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "MultiClassification loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, MultiClassDataProcessor), "MultiClassification data processor error."
        # Test output 
        task = Task.multiclass()
        y_prob = task.predict_proba(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_prob, d['y_prob']), "MultiClassification probability prediction error."
        assert np.allclose(y_pred, d['y_pred']), "MultiClassification prediction error."
