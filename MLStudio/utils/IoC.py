# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \IoC.py                                                           #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 8:54:28 am                        #
# Last Modified : Thursday, July 16th 2020, 8:54:29 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.algorithms.optimization.gradient_descent import GradientDescent
from mlstudio.supervised.algorithms.optimization.observers import base
from mlstudio.supervised.algorithms.optimization.observers import monitor
from mlstudio.supervised.algorithms.optimization.observers import early_stop
from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.algorithms.optimization.services import tasks
from mlstudio.supervised.algorithms.optimization import gradient_descent
from mlstudio.supervised.metrics.regression import regression, binaryclass
from mlstudio.utils.data_manager import RegressionData
from mlstudio.utils.data_manager import BinaryClassData
from mlstudio.utils.data_manager import MultiClassData
from mlstudio.utils.print import Printer
# --------------------------------------------------------------------------- #
class LossFunctions(containers.DeclarativeContainer):
    """IoC container of default loss function providers."""
    # Note: Default regularization parameters are based on tensorflow defaults.

    quadratic = providers.Factory(loss.Quadratic)
    quadratic_l1 = providers.Factory(loss.Quadratic, 
                                     regularizer=regularizers.L1(alpha=0.01))
    quadratic_l2 = providers.Factory(loss.Quadratic, 
                                     regularizer=regularizers.L2(alpha=0.01))
    quadratic_l1_l2 = providers.Factory(loss.Quadratic, 
                                     regularizer=regularizers.L1_L2(alpha=0.01,
                                     ratio=0.5))                                                                          

    cross_entropy = providers.Factory(loss.CrossEntropy)
    cross_entropy_l1 = providers.Factory(loss.CrossEntropy, 
                                     regularizer=regularizers.L1(alpha=0.01))
    cross_entropy_l2 = providers.Factory(loss.CrossEntropy, 
                                     regularizer=regularizers.L2(alpha=0.01))
    cross_entropy_l1_l2 = providers.Factory(loss.CrossEntropy, 
                                     regularizer=regularizers.L1_L2(alpha=0.01,
                                     ratio=0.5))

    categorical_cross_entropy = providers.Factory(loss.CategoricalCrossEntropy)
    categorical_cross_entropy_l1 = providers.Factory(loss.CategoricalCrossEntropy, 
                                     regularizer=regularizers.L1(alpha=0.01))
    categorical_cross_entropy_l2 = providers.Factory(loss.CategoricalCrossEntropy, 
                                     regularizer=regularizers.L2(alpha=0.01))
    categorical_cross_entropy_l1_l2 = providers.Factory(loss.CategoricalCrossEntropy, 
                                     regularizer=regularizers.L1_L2(alpha=0.01,
                                     ratio=0.5))                                               

# --------------------------------------------------------------------------- #
class Tasks(containers.DeclarativeContainer):
    """IoC container for task providers."""

    linear_regression = providers.Factory(tasks.LinearRegression,
                                          loss=LossFunctions.quadratic,
                                          data_processor=RegressionData,
                                          activation=None)
    linear_regression_l1 = providers.Factory(tasks.LinearRegression,
                                          loss=LossFunctions.quadratic_l1,
                                          data_processor=RegressionData,
                                          activation=None)                      
    linear_regression_l2 = providers.Factory(tasks.LinearRegression,
                                          loss=LossFunctions.quadratic_l2,
                                          data_processor=RegressionData,
                                          activation=None)
    linear_regression_l1_l2 = providers.Factory(tasks.LinearRegression,
                                          loss=LossFunctions.quadratic_l1_l2,
                                          data_processor=RegressionData,
                                          activation=None)

    logistic_regression = providers.Factory(tasks.BinaryClassification,
                                          loss=LossFunctions.cross_entropy,
                                          data_processor=BinaryClassData,
                                          activation=activations.Sigmoid)
    logistic_regression_l1 = providers.Factory(tasks.BinaryClassification,
                                          loss=LossFunctions.cross_entropy_l1,
                                          data_processor=BinaryClassData,
                                          activation=activations.Sigmoid)                      
    logistic_regression_l2 = providers.Factory(tasks.BinaryClassification,
                                          loss=LossFunctions.cross_entropy_l2,
                                          data_processor=BinaryClassData,
                                          activation=activations.Sigmoid)
    logistic_regression_l1_l2 = providers.Factory(tasks.BinaryClassification,
                                          loss=LossFunctions.cross_entropy_l1_l2,
                                          data_processor=BinaryClassData,
                                          activation=activations.Sigmoid)                                          

    multiclass = providers.Factory(tasks.MultiClassification,
                                          loss=LossFunctions.categorical_cross_entropy,
                                          data_processor=MultiClassData,
                                          activation=activations.Softmax)
    multiclass_l1 = providers.Factory(tasks.MultiClassification,
                                          loss=LossFunctions.categorical_cross_entropy_l1,
                                          data_processor=MultiClassData,
                                          activation=activations.Softmax)                      
    multiclass_l2 = providers.Factory(tasks.MultiClassification,
                                          loss=LossFunctions.categorical_cross_entropy_l2,
                                          data_processor=MultiClassData,
                                          activation=activations.Softmax)
    multiclass_l1_l2 = providers.Factory(tasks.MultiClassification,
                                          loss=LossFunctions.categorical_cross_entropy_l1_l2,
                                          data_processor=MultiClassData,
                                          activation=activations.Softmax)                                                

# --------------------------------------------------------------------------- #
# Performance observer
PerformanceObserver = collections.namedtuple('PerformanceObserver',['metric', 'epsilon', 'patience'])

# Create factory
performance_observer_factory = providers.Factory(PerformanceObserver)

# Create a performance observer object
po = performance_observer_factory(metric='val_score', epsilon=0.001, patience=5)

# --------------------------------------------------------------------------- #
class Summary(containers.DeclarativeContainer):       
    """IoC Container for Gradient Descent summary providers."""

    summary = providers.Factory(report.Summary,
                                printer=Printer())            

# --------------------------------------------------------------------------- #
#                          ESTIMATOR CONTAINERS                               #
# --------------------------------------------------------------------------- #
# Estimators
class Estimators(containers.DeclarativeContainer):
    """IoC Container for gradient descent estimator providers."""                                
    GradientDescent = collections.namedtuple('GradientDescent',[
                                        'task', 'eta0', 'epochs', 'batch_size',
                                        'val_size', 'theta_init',  'optimizer', 
                                        'metric', 'early_stop', 'learning_rate', 
                                        'observer_list', 'progress', 'blackbox',
                                        'summary', 'verbose', 'random_state'], 
                                        defaults=[
                                            0.01, 1000, None, 0.3, None,
                                            None, None, None, None,
                                            None, None, None, None,
                                            False, None])

    # Create the factory
    gradient_descent_factory = providers.Factory(GradientDescent,                                
                                    optimizer=optimizers.GradientDescentOptimizer,
                                    observer_list=base.ObserverList,
                                    progress=report.Progress,
                                    blackbox=history.BlackBox,
                                    summary=report.Summary
                                    )

# Obtain some dependencies
linear_regression = Tasks.linear_regression()
estimator = Estimators.gradient_descent_factory(task=linear_regression, metric=regression.R2)
print(estimator.metric().name)
    

# %%
