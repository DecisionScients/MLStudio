# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \estimators.py                                                    #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, July 17th 2020, 12:35:16 am                         #
# Last Modified : Friday, July 17th 2020, 12:35:16 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers for Estimators."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

from collections import namedtuple
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.data_services.preprocessing import RegressionDataProcessor
from mlstudio.data_services.preprocessing import BinaryClassDataProcessor
from mlstudio.data_services.preprocessing import MultiClassDataProcessor
from mlstudio.supervised.algorithms.optimization import gradient_descent
from mlstudio.supervised.algorithms.optimization.observers import debug, early_stop
from mlstudio.supervised.algorithms.optimization.observers import learning_rate
from mlstudio.supervised.algorithms.optimization.observers.base import ObserverList
from mlstudio.supervised.algorithms.optimization.observers import report, history
from mlstudio.supervised.algorithms.optimization.services import optimizers, regularizers
from mlstudio.supervised.algorithms.optimization.services import tasks, loss
from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.metrics import regression, binaryclass, multiclass
from mlstudio.utils.data_manager import AddBiasTerm, OneHotLabelEncoder, LabelEncoder
from mlstudio.utils.data_manager import LabelEncoder, DataSplitter
from mlstudio.utils.print import Printer


# --------------------------------------------------------------------------- #
Lasso = namedtuple('Lasso', ['alpha'])
Ridge = namedtuple('Ridge', ['alpha'])
Elasticnet = namedtuple('Elasticnet', ['alpha', 'ratio'])
# --------------------------------------------------------------------------- #
class GradientDescent(containers.DeclarativeContainer):
    """Application container for regression and classification factories."""

    regressor = providers.Factory(
        gradient_descent.GDRegressor,
            eta0=0.01,
            epochs=1000,
            batch_size=None,
            val_size=0.3,
            loss=loss.Quadratic(),
            data_processor = RegressionDataProcessor(add_bias_transformer=AddBiasTerm(),
                    split_transformer=DataSplitter(),
                    one_hot_label_encoder=OneHotLabelEncoder(),
                    label_encoder=LabelEncoder()),
            activation = None,
            theta_init=None,
            optimizer=optimizers.GradientDescentOptimizer(),                                    
            scorer=regression.R2(),
            early_stop=None,
            learning_rate=None,                           
            observer_list=ObserverList(),
            progress=report.Progress(),
            blackbox=history.BlackBox(),
            summary=report.Summary(printer=Printer()),
            verbose=False,
            random_state=None,
            check_gradient=False,
            gradient_checker=debug.GradientCheck())

    binaryclass = providers.Factory(
        gradient_descent.GDBinaryclass,
            eta0=0.01,
            epochs=1000,
            batch_size=None,
            val_size=0.3,
            loss=loss.CrossEntropy(),
            data_processor = BinaryClassDataProcessor(add_bias_transformer=AddBiasTerm(),
                    split_transformer=DataSplitter(),
                    one_hot_label_encoder=OneHotLabelEncoder(),
                    label_encoder=LabelEncoder()),
            activation = activations.Sigmoid(),
            theta_init=None,
            optimizer=optimizers.GradientDescentOptimizer(),                                    
            scorer=binaryclass.Accuracy(),
            early_stop=None,
            learning_rate=None,                           
            observer_list=ObserverList(),
            progress=report.Progress(),
            blackbox=history.BlackBox(),
            summary=report.Summary(printer=Printer()),
            verbose=False,
            random_state=None,
            check_gradient=False,
            gradient_checker=debug.GradientCheck())            

    multiclass = providers.Factory(
        gradient_descent.GDMulticlass,
            eta0=0.01,
            epochs=1000,
            batch_size=None,
            val_size=0.3,
            loss=loss.CategoricalCrossEntropy(),
            data_processor = MultiClassDataProcessor(add_bias_transformer=AddBiasTerm(),
                    split_transformer=DataSplitter(),
                    one_hot_label_encoder=OneHotLabelEncoder(),
                    label_encoder=LabelEncoder()),
            activation = activations.Softmax(),
            theta_init=None,
            optimizer=optimizers.GradientDescentOptimizer(),                                    
            scorer=multiclass.Accuracy(),
            early_stop=None,
            learning_rate=None,                           
            observer_list=ObserverList(),
            progress=report.Progress(),
            blackbox=history.BlackBox(),
            summary=report.Summary(printer=Printer()),
            verbose=False,
            random_state=None,
            check_gradient=False,
            gradient_checker=debug.GradientCheck())                        

