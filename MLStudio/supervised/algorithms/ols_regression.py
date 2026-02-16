#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : regression.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Monday, March 23rd 2020, 10:31:37 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Regression algorithms.

This class encapsulates the core behaviors for regression classes. Currently,
the following regression classes are supported.
    
    * Linear Regression
    * Lasso Regression
    * Ridge Regression
    * ElasticNet Regression

The core behaviors exposed for each class include:

    * predict : Predicts outputs as linear combination of inputs and weights.
    * compute_cost : Computes cost associated with predictions
    * compute_gradient : Computes the derivative of loss w.r.t. to weights

"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from mlstudio.supervised.metrics.regression import R2
from mlstudio.utils.data_manager import StandardScaler

# --------------------------------------------------------------------------- #
#                        LINEAR REGRESSION (OLS)                              #
# --------------------------------------------------------------------------- # 
class OLSRegression(BaseEstimator, RegressorMixin):
    """Ordinary least squares closed form linear regression."""

    def __init__(self, metric=R2()):
        self.metric_ = metric

    def fit(self, X, y):
        """Fits the linear regression ordinary least squares solution.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values 

        Returns
        -------
        self : returns instance of self._
        """
        # Validate input and target 
        X, y = check_X_y(X, y)

        # Add bias term of ones to feature matrix
        X = np.insert(X, 0, 1.0, axis=1) 

        # Obtain n samples nad comp0ute matrix rank
        n = X.shape[1]
        r = np.linalg.matrix_rank(X)        

        # Find matrix equivalent using singular value decomposition
        U, S, V = np.linalg.svd(X)

        # Derive D^+ from sigma        
        D_plus = np.zeros((X.shape[0], X.shape[1])).T
        D_plus[:S.shape[0], :S.shape[0]] = np.linalg.inv(np.diag(S))
        
        # Compute Moore-Penrose pseudoinverse of X
        X_plus = V.T.dot(D_plus).dot(U.T)

        # Weights are the dot product of X_plus and y
        theta = X_plus.dot(y)

        # Save solution in attributes        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

        return self

    def predict(self, X):
        """Computes prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : prediction
        """        
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Create prediction as linear combination of inputs and parameters
        y_pred  = self.intercept_ + X.dot(self.coef_)
        return y_pred

    def score(self, X, y):
        """Computes scores using the metric parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        y : array_like of shape (n_samples,) or (n_samples, n_classes)
            The target variable.

        Returns
        -------
        score based upon the metric object.
        
        """
        X, y = check_X_y(X,y)
        y_pred = self.predict(X)        
        score = self.metric_(y, y_pred)
        return score



