#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : regularization.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 11:17:15 pm                        #
# Last Modified : Saturday, May 16th 2020, 11:17:15 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes used to regularize cost and gradient computations."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mlstudio.utils.data_manager import ZeroBiasTerm
from mlstudio.utils import validation
# --------------------------------------------------------------------------  #
class Regularizer(ABC, BaseEstimator, TransformerMixin):
    """Base class for regularization classes."""
    @abstractmethod
    def __init__(self):
        pass

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, x):
        validation.validate_range(x, 'alpha', minimum=0, left='open', right='closed')
        self._alpha = x        

    @abstractmethod
    def cost(self, theta):
        """Computes regularization to be added to cost function.

        Parameters
        ----------
        theta : np.array shape = (n_features) or (n_features, n_classes)
            The model parameters

        m : int
            The sample size
        """
        pass

    @abstractmethod
    def gradient(self, theta):
        """Computes the regularization gradient.

        Parameters
        ----------
        theta : np.array shape = (n_features) or (n_features, n_classes)
            The model parameters

        m : int
            The sample size
        """
        pass

# --------------------------------------------------------------------------  #
class L1(Regularizer):
    """ Regularizer for Lasso Regression """
    def __init__(self, alpha=0.01):        
        self.alpha = alpha
        self.name = "Lasso (L1) Regularizer"

    def cost(self, theta):
        return self.alpha * np.sum(np.abs(theta))

    def gradient(self, theta):        
        return self.alpha * np.sign(theta)        
    
# --------------------------------------------------------------------------  #
class L2(Regularizer):
    """ Regularizer for Ridge Regression """
    def __init__(self, alpha=0.01):        
        self.alpha = alpha
        self.name = "Ridge (L2) Regularizer"
    
    def cost(self, theta):
        return self.alpha * np.sum(np.square(theta))

    def gradient(self, theta):
        return self.alpha * theta
# --------------------------------------------------------------------------  #
class L1_L2(Regularizer):
    """ Regularizer for Elastic Net Regression """

    def __init__(self, alpha=0.01, ratio=0.5):        
        self.alpha = alpha
        self.ratio = ratio
        self.name = "Elasticnet (L1_L2) Regularizer"

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, x):
        validation.validate_range(x, 'ratio', minimum=0, maximum=1, left='closed', right='closed')
        self._ratio = x                

    def cost(self, theta):
        l1_contr = self._ratio * np.sum(np.abs(theta))
        l2_contr = (1 - self._ratio)/2 * np.sum(np.square(theta))
        return self._alpha * (l1_contr + l2_contr)

    def gradient(self, theta):
        l1_contr = self._ratio * np.sign(theta)
        l2_contr = (1 - self._ratio)  * theta
        return self._alpha * (l1_contr + l2_contr) 