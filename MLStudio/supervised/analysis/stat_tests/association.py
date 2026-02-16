#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Studio                                                       #
# Version : 0.1.0                                                             #
# File    : association.py                                                    #
# Python  : 3.8.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/datastudio                     #
# --------------------------------------------------------------------------- #
# Created       : Saturday, February 22nd 2020, 5:49:27 am                    #
# Last Modified : Saturday, February 22nd 2020, 5:49:27 am                    #
# Modified By   : John James (jjames@decisionscients.com>)                    #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #

"""Module statistical tests of association between 2 variables.

Association Tests
-----------------
The following tests of association are supported:

    * Chi-Squared Test
    * Fisher's Exact Test
    * One-way ANOVA Test
    * Kruskal Wallis Test
    * Pearsons Correlation
    * Spearmans R Correlation
    * Analysis of Covariance
    * Canonical Correlation

Centrality
----------

    * One-Sample t-test
    * One-Sample Median Test

Compare Groups
--------------

        * Paired t-test
        * 2 Independent t-tests
        * Wilcoxon-Mann Whitney Test
        * Wilcoxon Signed Rank Test
        * One-way Repeated ANOVA Tests
        * Friedman Test
        * Factorial Anova
        * Binomial Test
        * McNemar Test

Data Reduction
--------------

        * Factor Analysis

Predictive Analytics
--------------------

    * Simple Linear Regression
    * Multiple Linear Regression
    * Multiple Logistic Regression
    * Multivariate Multiple Linear Regression
    * Discriminant Analysis
    * Factorial Logistic Regression
    * Ordered Logistic Regression
    * Repeated Metrics Logistic Regression

Note: This package makes liberal use of three statistical software packages.

    * SciPy : A Python-based ecosystem for mathematics, science and engineering.
    * Statsmodels : Statistical models, hypothesis tests and data exploration
    * scikit-learn : A machine learning platform for Python.

Each of the classes above comply with an Abstract Base Class which defines
the interface for all test classes.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from tabulate import tabulate

from datastudio.understanding.stat_tests.interface import AbstractStatisticalTest

# =========================================================================== #
#                           Association Tests                                 #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                            Fisher Exact Test                                #
# --------------------------------------------------------------------------- #
class FisherExact(AbstractStatisticalTest):
    """Performs the Fisher Exact Test.

    Fisher's exact test is a statistical test used to determine if 
    there are nonrandom associations between two categorical variables.    
    """

    def __init__(self):
        super(FisherExact, self).__init__()

    def fit(self, X, alternative='two-sided'):
        """ Performs Fisher's exact test.

        Parameters
        ----------        
        X : array_like of ints
            A 2x2 contingency table. Elements should be non-negative integers.
        alternative : {‘two-sided’, ‘less’, ‘greater’}, optional
            Defines the alternative hypothesis. The following options 
            are available (default is ‘two-sided’):
                - ‘two-sided’
                - ‘less’: one-sided
                ‘greater’: one-sided        
        """
        self._statistic, self._p = fisher_exact(X, alternative=alternative)

    def get_result(self):
        """ Returns results of statistical test.

        Returns
        -------
        oddsratio : float
            This is prior odds ratio and not a posterior estimate.
        p_value : float
            P-value, the probability of obtaining a distribution at least 
            as extreme as the one that was actually observed, assuming that 
            the null hypothesis is true.

        """
        return super(FisherExact, self).get_result()

    def print(self):
        result = {'Odds-Ratio': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))        



# --------------------------------------------------------------------------- #
#                         Pearson's Correlation Test                          #
# --------------------------------------------------------------------------- #
class PearsonR(AbstractStatisticalTest):
    """Calculates a Pearson correlation coefficient and the p-value 
    for testing non-correlation.

    The Pearson correlation coefficient measures the linear relationship 
    between two datasets. Strictly speaking, Pearson’s correlation requires 
    that each dataset be normally distributed. Like other correlation 
    coefficients, this one varies between -1 and +1 with 0 implying 
    no correlation. Correlations of -1 or +1 imply an exact linear 
    relationship. Positive correlations imply that as x increases, so does y. 
    Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated 
    system producing datasets that have a Pearson correlation at least 
    as extreme as the one computed from these datasets. The p-values 
    are not entirely reliable but are probably reasonable for datasets 
    larger than 500 or so.
    
    Attributes
    ----------
    R : Pearsons correlation coefficient
    p : 2-tailed p-value
    
    """

    def __init__(self):
        super(PearsonR, self).__init__()      

    def fit(self,X, Y):    
        """Calculates a Pearson correlation coefficient and the p-value 
        for testing non-correlation.

        Parameters
        ----------
        x : (N,) array_like
            Input
        y : (N,) array_like
            Input
        """
    
        self._statistic, self._p = pearsonr(X, Y)

    def get_result(self):
        """Returns the Pearson's correlation coefficient and 2-tailed p-value.
        
        Returns
        -------
        R : Pearsons correlation coefficient
        p : 2-tailed p-value

        """
        return super(PearsonR, self).get_result()

    def print(self):
        result = {"Pearson's Correlation Coefficient": [self._r], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))             

# --------------------------------------------------------------------------- #
#                         Spearman's Correlation Test                         #
# --------------------------------------------------------------------------- #
class SpearmanR(AbstractStatisticalTest):
    """Calculate a Spearman correlation coefficient with associated p-value.

    The Spearman rank-order correlation coefficient is a nonparametric measure 
    of the monotonicity of the relationship between two datasets. Unlike 
    the Pearson correlation, the Spearman correlation does not assume 
    that both datasets are normally distributed. Like other correlation 
    coefficients, this one varies between -1 and +1 with 0 implying no 
    correlation. Correlations of -1 or +1 imply an exact monotonic 
    relationship. Positive correlations imply that as x increases, so 
    does y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system 
    producing datasets that have a Spearman correlation at least as extreme 
    as the one computed from these datasets. The p-values are not entirely 
    reliable but are probably reasonable for datasets larger than 500 or so.
    
    Attributes
    ----------
    correlation : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient 
        (if only 2 variables are given as parameters. Correlation matrix 
        is square with length equal to total number of variables 
        (columns or rows) in a and b combined.
    pvalue: float
        The two-sided p-value for a hypothesis test whose null hypothesis 
        is that two sets of data are uncorrelated, has same dimension as rho.

    """

    def __init__(self):
        super(SpearmanR, self).__init__()      

    def fit(self,X, Y):    
        """Calculates a Spearman correlation coefficient and the p-value.

        Parameters
        ----------
        a, b : 1D or 2D array_like, b is optional
            One or two 1-D or 2-D arrays containing multiple variables and 
            observations. When these are 1-D, each represents a vector of 
            observations of a single variable. For the behavior in the 2-D 
            case, see under axis, below. Both arrays need to have the same 
            length in the axis dimension.
        axis : int or None, optional
            If axis=0 (default), then each column represents a variable, with 
            observations in the rows. If axis=1, the relationship is 
            transposed: each row represents a variable, while the columns 
            contain observations. If axis=None, then both arrays will be raveled.

        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. The following options 
            are available (default is ‘propagate’):
            - ‘propagate’: returns nan
            - ‘raise’: throws an error
            - ‘omit’: performs the calculations ignoring nan values
        """
    
        self._statistic, self._p = spearmanr(X, Y)

    def get_result(self):
        """Returns the Pearson's correlation coefficient and 2-tailed p-value.
        
        Returns
        -------
        correlation : float or ndarray (2-D square)
            Spearman correlation matrix or correlation coefficient 
            (if only 2 variables are given as parameters. Correlation matrix 
            is square with length equal to total number of variables 
            (columns or rows) in a and b combined.
        pvalue: float
            The two-sided p-value for a hypothesis test whose null hypothesis 
            is that two sets of data are uncorrelated, has same dimension as rho.

        """
        return super(SpearmanR, self).get_result()

    def print(self):
        result = {"Spearman's Correlation Coefficient": [self._r], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))             

# --------------------------------------------------------------------------- #
#                     Canonical Correlation Analysis                          #
# --------------------------------------------------------------------------- #
class CCA(AbstractStatisticalTest):
    """Performs a canonical correlation analysis.

    The cross decomposition module contains two main families of
    algorithms: the partial least squares (PLS) and the canonical
    correlation analysis (CCA).

    These families of algorithms are useful to find linear relations between
    two multivariate datasets: the X and Y arguments of the fit method
    are 2D arrays.

    Cross decomposition algorithms find the fundamental relations between 
    two matrices (X and Y). They are latent variable approaches to 
    modeling the covariance structures in these two spaces. They will 
    try to find the multidimensional direction in the X space that 
    explains the maximum multidimensional variance direction in the Y 
    space. PLS-regression is particularly suited when the matrix of 
    predictors has more variables than observations, and when there 
    is multicollinearity among X values. By contrast, standard 
    regression will fail in these cases.
    

    Attributes
    ----------
    x_weights_array, [p, n_components]
    X block weights vectors.

    y_weights_array, [q, n_components]
    Y block weights vectors.

    x_loadings_array, [p, n_components]
    X block loadings vectors.

    y_loadings_array, [q, n_components]
    Y block loadings vectors.

    x_scores_array, [n_samples, n_components]
    X scores.

    y_scores_array, [n_samples, n_components]
    Y scores.

    x_rotations_array, [p, n_components]
    X block to latents rotations.

    y_rotations_array, [q, n_components]

    n_iter_array-like
    Number of iterations of the NIPALS inner loop for each component.
    """

    def __init__(self, n_components=2, scale=True, max_iter=500, 
            tol=1e-06, copy=True):
        self._X_scores = 0
        self._Y_scores = 0
        self._cca = CCA(n_components, scale, max_iter, tol,copy)

    def fit(self, X, y):       
        """Performs a canonical correlation analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and 
            n_features is the number of predictors.

        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and 
            n_targets is the number of response variables.

        """

        self._cca.fit(X, y)        
        self._X_scores, self._Y_scores = self._cca.transform(X,y)

    def get_result(self):
        """Returns a canonical correlation analysis.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return self._X_scores, self._Y_scores

    @property
    def cca(self):
        return self._X_scores, self._Y_scores

    def print(self):        
        print(tabulate(self._X_scores, self._Y_scores))          

# --------------------------------------------------------------------------- #
#                            Covariance                                       #
# --------------------------------------------------------------------------- #
class Covariance(AbstractStatisticalTest):
    """Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together. 
    If we examine N-dimensional samples, X = [x_1, x_2, ... x_N]^T, then 
    the covariance matrix element C_{ij} is the covariance of x_i and x_j. 
    The element C_{ii} is the variance of x_i.
    """

    def __init__(self):
        self._cov = None           

    def fit(self, m, y=None, rowvar=True, bias=False, ddof=None,     
            fweights=None, aweights=None):        
        """Estimate a covariance matrix, given data and weights.

        Parameters
        ----------
        m : array_like
            A 1-D or 2-D array containing multiple variables and observations. Each row of m represents a variable, and each column a single observation of all those variables. Also see rowvar below.

        y : array_like, optional
            An additional set of variables and observations. y has the same form as that of m.

        rowvar : bool, optional
            If rowvar is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.

        bias : bool, optional
            Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate). If bias is True, then normalization is by N. These values can be overridden by using the keyword ddof in numpy versions >= 1.5.

        ddof : int, optional
            If not None the default value implied by bias is overridden. Note that ddof=1 will return the unbiased estimate, even if both fweights and aweights are specified, and ddof=0 will return the simple average. See the notes for the details. The default value is None.

        fweights : array_like, int, optional
            1-D array of integer frequency weights; the number of times each observation vector should be repeated.

        aweights : array_like, optional
            1-D array of observation vector weights. These relative weights are typically large for observations considered “important” and smaller for observations considered less “important”. If ddof=0 the array of weights can be used to assign probabilities to observation vectors.

        """
        self._cov = np.cov(m, y=y, rowvar=rowvar, bias=bias, ddof=ddof, 
            fweights=fweights, aweights=aweights)

    def get_result(self):
        """ Returns results of covariance estimate.

        Returns
        -------
        out : ndarray
            The covariance matrix of the variables.

        """        
        return self._cov

    def print(self):        
        print(tabulate(self._cov))
