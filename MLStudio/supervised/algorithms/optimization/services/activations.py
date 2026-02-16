#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : activation.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 11:39:45 pm                        #
# Last Modified : Saturday, May 16th 2020, 11:39:45 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Activation functions."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from mlstudio.utils.validation import validate_range
# --------------------------------------------------------------------------  #
# Functions adapted from the following sources.
# Reference https://en.wikipedia.org/wiki/Activation_function 
# Reference https://www.python-course.eu/softmax.php
# --------------------------------------------------------------------------  #
class Activation(ABC, BaseEstimator):
    """Base class for activation classes."""
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

class Sigmoid(Activation):
    """Sigmoid activation function."""

    def __call__(self, x):
        if np.all(x >= 0):            
            s = 1 / (1 + np.exp(-x))
        else:
            s =  np.exp(x) / (1 + np.exp(x))    
        return s
        

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax(Activation):
    def __call__(self, x):
        # Subtract the max to avoid underflow or overflow errors 
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        s = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return s

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class TanH(Activation):
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-1))        

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU(Activation):
    def __call__(self, x):
        return np.where(x > 0, x, 0)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
