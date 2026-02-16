# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_performance_observer copy.py                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 2nd 2020, 2:36:18 pm                         #
# Last Modified : Thursday, July 2nd 2020, 2:36:18 pm                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests early stop."""
from collections import OrderedDict
import os
import math
import os
from pathlib import Path
import sys
testdir = str(Path(__file__).parents[2])
testdatadir = os.path.join(testdir, 'test_data')
sys.path.append(testdatadir)

import numpy as np
import pandas as pd
import pytest
from pytest import mark
from tabulate import tabulate

from mlstudio.supervised.algorithms.optimization.observers.early_stop import EarlyStop
from mlstudio.supervised.algorithms.optimization.gradient_descent import GradientDescent

# --------------------------------------------------------------------------  #
@mark.observer
@mark.early_stop
class EarlyStopTests:   

    def test_early_stop(self, make_regression_data):    
        epsilons = [0.1, 0.01, 0.001, 0.0001]
        X, y = make_regression_data
        last_epoch=0
        for epsilon in epsilons:
            est = GDRegressor(observers=[EarlyStop(epsilon=epsilon,patience=50)],
                                           random_state=5)
            est.fit(X,y)
            assert est.get_blackbox().total_epochs > last_epoch, "Early stop error " + str(epsilon)
            last_epoch = est.get_blackbox().total_epochs
