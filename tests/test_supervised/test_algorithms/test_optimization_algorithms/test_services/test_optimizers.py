# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_optimizers copy.py                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 9th 2020, 8:23:30 am                         #
# Last Modified : Thursday, July 9th 2020, 8:23:30 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
#%%
import math
import os
from pathlib import Path
import sys

import glob
import numpy as np
import pandas as pd
import pytest
from pytest import mark
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression, make_classification
from sklearn.datasets import make_multilabel_classification

homedir = str(Path(__file__).parents[3])
datadir = os.path.join(homedir, "tests\\test_data")
sys.path.append(homedir)
sys.path.append(datadir)

from mlstudio.supervised.algorithms.optimization.services.optimizers import GradientDescentOptimizer
from mlstudio.supervised.algorithms.optimization.services.optimizers import Momentum, Nesterov
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adagrad, Adadelta     
from mlstudio.supervised.algorithms.optimization.services.optimizers import RMSprop, Adam, AdaMax
from mlstudio.supervised.algorithms.optimization.services.optimizers import Nadam, AMSGrad, AdamW
from mlstudio.supervised.algorithms.optimization.services.optimizers import AggMo, QuasiHyperbolicMomentum
# --------------------------------------------------------------------------  #
# Mock gradient function
def gradient(theta):
    theta = theta * 0.95
    return theta

@mark.optimizers
@mark.momentum
def test_optimizer_momentum(get_optimization_momentum_test_package):
    p = get_optimization_momentum_test_package
    theta = p['theta_init']
    alpha = p['alpha']
    optimizer = Momentum()
    for i in range(10):
        assert np.allclose(theta, p['theta'][i]), \
            "Momentum not working, Iteration {i} expected {e}, actual {a}".format(
                i = str(i),
                e=str(p['theta'][i]), a=str(theta)
        )
        theta, grad = optimizer(gradient, alpha, theta)    
    
@mark.optimizers
@mark.nesterov
def test_optimizer_nesterov(get_optimization_nesterov_test_package):
    p = get_optimization_nesterov_test_package
    theta = p['theta_init']
    alpha = p['alpha']
    optimizer = Nesterov()
    for i in range(10):
        assert np.allclose(theta, p['theta'][i]), \
            "Nesterov not working, Iteration {i} expected {e}, actual {a}".format(
                i = str(i),
                e=str(p['theta'][i]), a=str(theta)
        )
        theta, grad = optimizer(gradient, alpha, theta)  


@mark.optimizers
@mark.adagrad
def test_optimizer_adagrad(get_optimization_adagrad_test_package):
    p = get_optimization_adagrad_test_package
    theta = p['theta_init']
    alpha = p['alpha']
    optimizer = Adagrad()
    for i in range(4):
        assert np.allclose(theta, p['theta'][i]), \
            "Adagrad not working, Iteration {i} expected {e}, actual {a}".format(
                i = str(i),
                e=str(p['theta'][i]), a=str(theta)
        )
        theta, grad = optimizer(gradient, alpha, theta)          