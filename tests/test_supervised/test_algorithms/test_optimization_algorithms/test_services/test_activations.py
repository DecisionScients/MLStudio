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

from scipy.special import expit
from sklearn.utils.extmath import softmax

from mlstudio.supervised.algorithms.optimization.services.activations import Sigmoid, Softmax
# --------------------------------------------------------------------------  #
@mark.activations
@mark.sigmoid
class SigmoidActivationTests:

    def test_sigmoid_activation(self):
        # Get an array
        x = np.random.default_rng().uniform(low=0, high=20, size=20)
        # Expected result
        exp_result = expit(x)
        # Actual result
        act = Sigmoid()
        act_result = act(x)
        assert np.allclose(exp_result, act_result), "Sigmoid error"        

@mark.activations
@mark.softmax
class SoftmaxActivationTests:

    def test_softmax_activation(self):
        # Get an array
        x = np.random.default_rng().uniform(low=0, high=20, size=(20,4))
        # Expected result        
        exp_result = softmax(x)
        # Actual result
        act = Softmax()
        act_result = act(x)
        assert np.allclose(exp_result, act_result), "Softmax error"        





