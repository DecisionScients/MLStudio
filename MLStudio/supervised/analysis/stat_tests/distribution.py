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
"""Module defines the suite of statistical used to evaluate distributions.

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
from scipy.stats import chisquare, binom_test, anderson, zscore
from statsmodels.stats.diagnostic import lilliefors
from tabulate import tabulate

from datastudio.understanding.stat_tests.interface import AbstractStatisticalTest
# =========================================================================== #
#                           Distribution Tests                                #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                                  Z Score                                    #
# --------------------------------------------------------------------------- #
class ZScore(AbstractStatisticalTest):
    """Compute the z score.
    
    Compute the z score of each value in the sample, relative to the sample 
    mean and standard deviation.

    Parameters
    ----------
    a : array_like
        An array like object containing the sample data.

    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over 
        the whole array a.

    ddof ; int, optional
        Degrees of freedom correction in the calculation of the standard deviation. 
        Default is 0.

    nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
        Defines how to handle when input contains nan. ‘propagate’ returns nan,
        ‘raise’ throws an error, ‘omit’ performs the calculations ignoring nan 
        values. Default is ‘propagate’.

    Returns
    -------
    zscore : array_like
        The z-scores, standardized by mean and standard deviation of input array a.

    """

    def __init__(self):
        self._z = 0

    def fit(self,a, axis=0, ddof=0, nan_policy='propagate'):        
         self._z = zscore(a, axis=axis, ddof=ddof)

    def get_result(self):
        return self._z

    def print(self):
        result = {'Z Score': [self._z]}
        print(tabulate(result, headers='keys'))


# --------------------------------------------------------------------------- #
#                            Binomial Test                                    #
# --------------------------------------------------------------------------- #
class Binomial(AbstractStatisticalTest):
    """Performs the Binomial Test."""

    def __init__(self):        
        super(Binomial, self).__init__()        

    def fit(self, x, n=None, p=0.5):        
         self._p = binom_test(x,n,p)

    def get_result(self):
        return self._p

    def print(self):
        result = {'p-value': [self._p]}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                    Chi-Square Goodness-of-Fit Test                          #
# --------------------------------------------------------------------------- #
class ChiSquareGoF(AbstractStatisticalTest):
    """Performs the Chi-Squared Goodness of fit test.
    
    The chi-square test tests the null hypothesis that the categorical 
    data has the given frequencies.

    Attributes
    ----------
    statistic : float or ndarray
        The chi-squared test statistic. The value is a float if axis is 
        None or f_obs and f_exp are 1-D.
    p_value : float or ndarray
        The p-value of the test. The value is a float if ddof and   
        the return value chisq are scalars.
    """

    def __init__(self):
        super(ChiSquareGoF, self).__init__()


    def fit(self, X, exp=None, ddof=0, axis=0):   
        """Performs the statistical test.

        Parameters
        ----------
        f_obs: array_like
            Observed frequencies in each category.
        f_exp: array_like, optional
            Expected frequencies in each category. By default the 
            categories are assumed to be equally likely.
        ddof : int, optional
            “Delta degrees of freedom”: adjustment to the degrees of freedom 
            for the p-value. The p-value is computed using a chi-squared distribution 
            with k - 1 - ddof degrees of freedom, where k is the number of observed 
            frequencies. The default value of ddof is 0.
        axis: int or None, optional
            The axis of the broadcast result of f_obs and f_exp along which 
            to apply the test. If axis is None, all values in f_obs are treated as a 
            single data set. Default is 0.

        """
         
        self._statistic, self._p = chisquare(X,f_exp=exp, ddof=ddof, axis=axis)


    def print(self):
        result = {'Chi-Square': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                        Kolmogorov-Smirnov Test                              #
# --------------------------------------------------------------------------- #
class KolmogorovSmirnov(AbstractStatisticalTest):
    """Test assumed normal or exponential distribution using Lilliefors’ test.
    
    Lilliefors’ test is a Kolmogorov-Smirnov test with estimated parameters.

    Attributes
    ----------
    ksstat : float
        Kolmogorov-Smirnov test statistic with estimated mean and variance.

    pvalue : float
        If the pvalue is lower than some threshold, e.g. 0.05, then we can 
        reject the Null hypothesis that the sample comes from a 
        normal distribution.  

    """

    def __init__(self):
        super(KolmogorovSmirnov, self).__init__()


    def fit(self, x, dist='norm', pvalmethod='table'):   
        """Performs the statistical test.

        Parameters
        ----------
        x : array_like, 1d
            Data to test.

        dist : {‘norm’, ‘exp’}, optional
            The assumed distribution.

        pvalmethod : {‘approx’, ‘table’}, optional
            The method used to compute the p-value of the test statistic. 
            In general, ‘table’ is preferred and makes use of a very large 
            simulation. ‘approx’ is only valid for normality. if dist = ‘exp’ 
            table is always used. ‘approx’ uses the approximation formula of 
            Dalal and Wilkinson, valid for pvalues < 0.1. If the pvalue is 
            larger than 0.1, then the result of table is returned.

        """
         
        self._statistic, self._p = lilliefors(x, dist=dist, pvalmethod=pvalmethod)


    def print(self):
        result = {'KS Statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))        

# --------------------------------------------------------------------------- #
#                        Shapiro-Wilk Normality Test                          #
# --------------------------------------------------------------------------- #        
class Shapiro(AbstractStatisticalTest):
    """Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the data was drawn 
    from a normal distribution.

    Attributes
    ----------
    W : float
        The  test statistic.
    pvalue : float
        If the pvalue for the hypothesis test.  

    """

    def __init__(self):
        super(Shapiro, self).__init__()


    def fit(self, x, dist='norm', pvalmethod='table'):   
        """Perform the Shapiro-Wilk test for normality.

        Parameters
        ----------
        x : array_like, 1d
            Data to test.
        """
         
        self._statistic, self._p = lilliefors(x, dist=dist, pvalmethod=pvalmethod)


    def print(self):
        result = {'W Statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))  

# --------------------------------------------------------------------------- #
#                        Anderson-Darling Test                                #
# --------------------------------------------------------------------------- #        
class Anderson(AbstractStatisticalTest):
    """Perform the Shapiro-Wilk test for normality.

    Anderson-Darling test for data coming from a particular distribution.

    The Anderson-Darling tests the null hypothesis that a sample is drawn 
    from a population that follows a particular distribution. For the 
    Anderson-Darling test, the critical values depend on which distribution 
    is being tested against. This function works for normal, exponential, 
    logistic, or Gumbel (Extreme Value Type I) distributions.

    Attributes
    ----------
    statistic : float
        The Anderson-Darling test statistic.
    critical_values : list
        The critical values for this distribution.
    significance_level : list
        The significance levels for the corresponding critical values in 
        percents. The function returns critical values for a differing set 
        of significance levels depending on the distribution that is being 
        tested against.

    """
    def __init__(self):
        super(Anderson, self).__init__()
        self._critical_values = []
        self._significance_level = []


    def fit(self, x, dist='norm'):   
        """Perform the Anderson-Darling test.

        Parameters
        ----------
        x : array_like, 1d
            Data to test.

        dist : {‘norm’,’expon’,’logistic’,’gumbel’,’gumbel_l’, gumbel_r’,‘extreme1’}, optional 
            the type of distribution to test against. The default is ‘norm’ and 
            ‘extreme1’, ‘gumbel_l’ and ‘gumbel’ are synonyms.

        """
         
        self._statistic, self._critical_values, self._significance_level = \
            anderson(x, dist=dist)

    @property
    def p_value(self):
        note = "The Anderson-Darling test has no p-value."
        return note

    def get_results(self):
        """Returns results of Anderson-Darling test.

        Returns
        -------
        statistic : float
            The Anderson-Darling test statistic.
        critical_values : list
            The critical values for this distribution.
        significance_level : list
            The significance levels for the corresponding critical values in 
            percents. The function returns critical values for a differing set 
            of significance levels depending on the distribution that is being 
            tested against.    

        """

        return self._statistic, self._critical_values, self._significance_level


    def print(self):
        result = {'Critical Values': self._critical_values,
                  'Significance Level': self._significance_level,}
        print("Anderson-Darling Test Statistic : {stat}".format(stat=self._statistic))
        print("----------------------------------------------------")
        print(tabulate(result, headers='keys'))          