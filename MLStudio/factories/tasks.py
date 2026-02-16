# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \tasks.py                                                         #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, July 17th 2020, 12:34:48 am                         #
# Last Modified : Friday, July 17th 2020, 12:34:49 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers for Tasks."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.algorithms.optimization.services import tasks
from mlstudio.factories.data import DataProcessors
# --------------------------------------------------------------------------- #
L1_regularizer = collections.namedtuple('L1', ['alpha'])
L2_regularizer = collections.namedtuple('L1', ['alpha'])
L1_L2_regularizer = collections.namedtuple('L1_L2', ['alpha', 'ratio'])

# --------------------------------------------------------------------------- #
class RegressionTasks(containers.DeclarativeContainer):
    """IoC container for regression task providers."""

    base = providers.Factory(tasks.LinearRegression,
                                          loss=loss.Quadratic(),
                                          data_processor=DataProcessors.regression(),
                                          activation=None)

    lasso = providers.Factory(tasks.LinearRegression,
                                          loss=loss.Quadratic(regularizer=L1_regularizer),
                                          data_processor=DataProcessors.regression(),    
                                          activation=None)                                      

    ridge = providers.Factory(tasks.LinearRegression,
                                          loss=loss.Quadratic(regularizer=regularizers.L2(alpha=0.01)),
                                          data_processor=DataProcessors.regression(),
                                          activation=None)                           

    elasticnet = providers.Factory(tasks.LinearRegression,
                                          loss=loss.Quadratic(regularizer=regularizers.L1_L2(alpha=0.01, ratio=0.5)),
                                          data_processor=DataProcessors.regression(),
                                          activation=None)                                                         

# --------------------------------------------------------------------------- #
class BinaryclassTasks(containers.DeclarativeContainer):
    """IoC container for binary classification task providers."""

    base = providers.Factory(tasks.BinaryClassification,
                                          loss=loss.CrossEntropy(),
                                          data_processor=DataProcessors.binaryclass(),
                                          activation=activations.Sigmoid())     

    lasso = providers.Factory(tasks.BinaryClassification,                                          
                                          loss=loss.CrossEntropy(regularizer=regularizers.L1(alpha=0.01)),
                                          data_processor=DataProcessors.binaryclass(),
                                          activation=activations.Sigmoid())       

    ridge = providers.Factory(tasks.BinaryClassification,                                          
                                          loss=loss.CrossEntropy(regularizer=regularizers.L2(alpha=0.01)),
                                          data_processor=DataProcessors.binaryclass(),
                                          activation=activations.Sigmoid())              

    elasticnet = providers.Factory(tasks.BinaryClassification,                                          
                                          loss=loss.CrossEntropy(regularizer=regularizers.L1_L2(alpha=0.01, ratio=0.5)),
                                          data_processor=DataProcessors.binaryclass(),
                                          activation=activations.Sigmoid())                                                                                                                                   

# --------------------------------------------------------------------------- #
class MulticlassTasks(containers.DeclarativeContainer):
    """IoC container for regression task providers."""

    base = providers.Factory(tasks.MultiClassification,
                                          loss=loss.CategoricalCrossEntropy(),
                                          data_processor=DataProcessors.multiclass(),
                                          activation=activations.Softmax())

    lasso = providers.Factory(tasks.MultiClassification,
                                          loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L1(alpha=0.01)),
                                          data_processor=DataProcessors.multiclass(),
                                          activation=activations.Softmax())                                          

    ridge = providers.Factory(tasks.MultiClassification,
                                          loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L2(alpha=0.01)),
                                          data_processor=DataProcessors.multiclass(),
                                          activation=activations.Softmax())                                                                                    

    elasticnet = providers.Factory(tasks.MultiClassification,
                                          loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L1_L2(alpha=0.01, ratio=0.5)),
                                          data_processor=DataProcessors.multiclass(),
                                          activation=activations.Softmax())                                                                                                                              

