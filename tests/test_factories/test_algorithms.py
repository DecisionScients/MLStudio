# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_algorithms.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, August 7th 2020, 4:57:39 am                         #
# Last Modified : Friday, August 7th 2020, 4:57:39 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories.algorithms import GradientDescent
from mlstudio.supervised.algorithms.optimization.observers import early_stop 
from mlstudio.supervised.algorithms.optimization.observers import learning_rate
from mlstudio.supervised.algorithms.optimization.services import loss, regularizers 
from mlstudio.supervised.algorithms.optimization.services import activations

# --------------------------------------------------------------------------- #

@mark.factories
class GDFactoryTests:

    def test_regression_factories(self):
        # Test #1: Immutable parameters:         
        estimator = GradientDescent().regressor(eta0=0.05, epochs=2000, batch_size=32, 
                                    val_size=0.4, theta_init=5, verbose=True, 
                                    random_state=5, check_gradient=True)
        assert estimator.eta0 == 0.05, "eta0 error"                                
        assert estimator.epochs == 2000, "epochs error"                                
        assert estimator.batch_size == 32, "batch_size error"                                
        assert estimator.val_size == 0.4, "val_size error"                                
        assert estimator.theta_init == 5, "theta_init error"                                
        assert estimator.verbose == True, "verbose error"                                
        assert estimator.random_state == 5, "random_state error"                                
        assert estimator.check_gradient == True, "check_gradient error"                                
        # Test #2: Loss 
        estimator = GradientDescent().regressor(loss=loss.Quadratic(regularizer=regularizers.L2()))
        assert "L2" in estimator.loss.regularizer.name, "Loss initialization error"
        # Test #3: Data Processor     
        assert "Regression" in estimator.data_processor.__class__.__name__, "Data processor error"    
        # Test #4: Optimizer     
        assert "Gradient" in estimator.optimizer.name, "Optimizer  error"    
        # Test #5: Scorer
        assert "R2" in estimator.scorer.name, "Scorer  error"    
        # Test #6: Early stop   
        estimator = GradientDescent().regressor(early_stop=early_stop.EarlyStop())     
        assert "Early" in estimator.early_stop.name, "Early stop error"    
        # Test #7: Learning rate   
        estimator = GradientDescent().regressor(learning_rate=learning_rate.TimeDecay())     
        assert "Time" in estimator.learning_rate.name, "Learning rate error"  

    def test_binaryclass_factories(self):
        # Test #1: Immutable parameters:         
        estimator = GradientDescent().binaryclass(eta0=0.05, epochs=2000, batch_size=32, 
                                    val_size=0.4, theta_init=5, verbose=True, 
                                    random_state=5, check_gradient=True)
        assert estimator.eta0 == 0.05, "eta0 error"                                
        assert estimator.epochs == 2000, "epochs error"                                
        assert estimator.batch_size == 32, "batch_size error"                                
        assert estimator.val_size == 0.4, "val_size error"                                
        assert estimator.theta_init == 5, "theta_init error"                                
        assert estimator.verbose == True, "verbose error"                                
        assert estimator.random_state == 5, "random_state error"                                
        assert estimator.check_gradient == True, "check_gradient error"                                
        # Test #2: Loss 
        estimator = GradientDescent().binaryclass(loss=loss.CrossEntropy(regularizer=regularizers.L2()))
        assert "L2" in estimator.loss.regularizer.name, "Loss initialization error"
        # Test #3: Data Processor     
        assert "Binary" in estimator.data_processor.__class__.__name__, "Data processor error"    
        # Test #4: Optimizer     
        assert "Gradient" in estimator.optimizer.name, "Optimizer  error"    
        # Test #5: Scorer
        assert "accuracy" in estimator.scorer.name, "Scorer  error"    
        # Test #6: Early stop   
        estimator = GradientDescent().binaryclass(early_stop=early_stop.EarlyStop())     
        assert "Early" in estimator.early_stop.name, "Early stop error"    
        # Test #7: Learning rate   
        estimator = GradientDescent().binaryclass(learning_rate=learning_rate.TimeDecay())     
        assert "Time" in estimator.learning_rate.name, "Learning rate error"          

    def test_multiclass_factories(self):
        # Test #1: Immutable parameters:         
        estimator = GradientDescent().multiclass(eta0=0.05, epochs=2000, batch_size=32, 
                                    val_size=0.4, theta_init=5, verbose=True, 
                                    random_state=5, check_gradient=True)
        assert estimator.eta0 == 0.05, "eta0 error"                                
        assert estimator.epochs == 2000, "epochs error"                                
        assert estimator.batch_size == 32, "batch_size error"                                
        assert estimator.val_size == 0.4, "val_size error"                                
        assert estimator.theta_init == 5, "theta_init error"                                
        assert estimator.verbose == True, "verbose error"                                
        assert estimator.random_state == 5, "random_state error"                                
        assert estimator.check_gradient == True, "check_gradient error"                                
        # Test #2: Loss 
        estimator = GradientDescent().multiclass(loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L2()))
        assert "L2" in estimator.loss.regularizer.name, "Loss initialization error"
        # Test #3: Data Processor     
        assert "Multi" in estimator.data_processor.__class__.__name__, "Data processor error"    
        # Test #4: Optimizer     
        assert "Gradient" in estimator.optimizer.name, "Optimizer  error"    
        # Test #5: Scorer
        assert "accuracy" in estimator.scorer.name, "Scorer  error"    
        # Test #6: Early stop   
        estimator = GradientDescent().multiclass(early_stop=early_stop.EarlyStop())     
        assert "Early" in estimator.early_stop.name, "Early stop error"    
        # Test #7: Learning rate   
        estimator = GradientDescent().multiclass(learning_rate=learning_rate.TimeDecay())     
        assert "Time" in estimator.learning_rate.name, "Learning rate error"                  