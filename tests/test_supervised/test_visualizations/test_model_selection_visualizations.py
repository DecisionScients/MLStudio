# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_model_selection.py                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Monday, June 29th 2020, 5:02:13 pm                          #
# Last Modified : Monday, June 29th 2020, 5:02:14 pm                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
import os
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.machine_learning.gradient_descent import GDRegressor
from mlstudio.supervised.visual.model_selection import LearningCurves
# --------------------------------------------------------------------------- #
@mark.visual
@mark.model_selection
class ModelSelectionVisualizationTests:

    def test_learning_curve(self, make_regression_data):
        filepath = "tests/test_visualizations/test_visualization_output/learning_curve.html"         
        est = GDRegressor()
        X, y = make_regression_data
        plot = LearningCurves(est)
        plot.fit(X, y)
        plot.show()
        plot.save(filepath)
