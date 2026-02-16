#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Studio                                                       #
# Version : 0.1.0                                                             #
# File    : tests.py                                                          #
# Python  : 3.8.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/datastudio                     #
# --------------------------------------------------------------------------- #
# Created       : Thursday, February 20th 2020, 12:28:39 am                   #
# Last Modified : Thursday, February 20th 2020, 12:28:40 am                   #
# Modified By   : John James (jjames@decisionscients.com>)                    #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Module defines the suite of statistical tests provided in DataStudio.

This package supports 32 statistical tests, organized around various facets of 
statistical inference. The classes fall broadly into five groups:

    1. Tests of Association
    2. Tests of Central Tendency  
    3. Tests of Dispersion
    4. Tests between Groups
    5. Tests of Assumptions  
    6. Predictive Analytics.

The classes are hereby listed below by category.

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
import textwrap

# --------------------------------------------------------------------------- #
#                        AbstractStatisticalTest                              #
# --------------------------------------------------------------------------- #
class AbstractStatisticalTest(ABC):

    @abstractmethod
    def __init__(self):
        self._statistic = 0
        self._p = 0    


    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
    
    def describe(self, verbose=False, print=False):
        """Describes the test.
        
        Parameters

        """
        desc = {'Id': self._id, "Test Type" : self._type, 
                "Category": self._category, "Name": self._title,
                "Description" : self._desc_short}

        

    def get_result(self):
        """ Returns the statistic and p_value for the test."""
        return self._statistic, self._p

    @property
    def statistic(self):
        """ Returns the test statistic"""
        return self._statistic

    @property
    def p_value(self):
        """ Returns the p-value for the test statistic."""
        return self._p

    @abstractmethod
    def print(self):
        pass        

