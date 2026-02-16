#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Studio                                                       #
# Version : 0.1.0                                                             #
# File    : dispersion.py                                                     #
# Python  : 3.8.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/datastudio                     #
# --------------------------------------------------------------------------- #
# Created       : Saturday, February 22nd 2020, 6:08:39 am                    #
# Last Modified : Saturday, February 22nd 2020, 6:08:39 am                    #
# Modified By   : John James (jjames@decisionscients.com>)                    #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
from scipy.stats import skew, kurtosis, kurtosistest, bartlett
from tabulate import tabulate

from datastudio.understanding.stat_tests.interface import AbstractStatisticalTest
# --------------------------------------------------------------------------- #
#                                   Skew                                      #
# --------------------------------------------------------------------------- #
class Skew(AbstractStatisticalTest):
    """Compute the sample skewness of a data set.
    
    For normally distributed data, the skewness should be about zero. 
    For unimodal continuous distributions, a skewness value greater than 
    zero means that there is more weight in the right tail of the distribution. 
    The function skewtest can be used to determine if the skewness value 
    is close enough to zero, statistically speaking.
    
    Attributes
    -------
    skewness : ndarray
        The skewness of values along an axis, returning 0 where all values are equal.

    """

    def __init__(self):
        super(Skew, self).__init__()

    def fit(self,a, axis=0, bias=False, nan_policy='propagate'):     
        """Compute the sample skewness of a data set.
        
        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over 
            the whole array a.

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. ‘propagate’ returns nan,
            ‘raise’ throws an error, ‘omit’ performs the calculations ignoring nan 
            values. Default is ‘propagate’.

        Returns
        -------
        skewness : ndarray
            The skewness of values along an axis, returning 0 where all values are equal.

        """           
        self._statistic = skew(a, axis=axis, bias=bias)

    @property
    def skew(self):
        return self._statistic

    def get_result(self):
        return self._statistic

    def print(self):
        result = {'Skew': [self._statistic]}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                                Kurtosis                                     #
# --------------------------------------------------------------------------- #
class Kurtosis(AbstractStatisticalTest):
    """Compute the kurtosis (Fisher or Pearson) of a dataset.
    
    Kurtosis is the fourth central moment divided by the square of the variance. 
    If Fisher’s definition is used, then 3.0 is subtracted from the result 
    to give 0.0 for a normal distribution.
    
    If bias is False then the kurtosis is calculated using k statistics 
    to eliminate bias coming from biased moment estimators
    
    This class also uses kurtosistest to see if result is close enough to normal.

    Attributes
    -------
    kurtosis : array
        The kurtosis of values along an axis. If all values are equal, 
        return -3 for Fisher’s definition and 0 for Pearson’s definition.

    statistic: float
        The computed z-score for this test.

    pvalue: float
        The two-sided p-value for the hypothesis test.        

    """

    def __init__(self):
        super(Kurtosis, self).__init__()
        self._k = 0

    def fit(self,a, axis=0, bias=False, fisher=True, nan_policy='propagate'):  
        """Compute the kurtosis (Fisher or Pearson) of a dataset.
        
        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over 
            the whole array a.

        fisher: bool, optional
            If True, Fisher’s definition is used (normal ==> 0.0). If False, 
            Pearson’s definition is used (normal ==> 3.0).        

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. ‘propagate’ returns nan,
            ‘raise’ throws an error, ‘omit’ performs the calculations ignoring nan 
            values. Default is ‘propagate’.

        Returns
        -------
        kurtosis : array
            The kurtosis of values along an axis. If all values are equal, 
            return -3 for Fisher’s definition and 0 for Pearson’s definition.

        statistic: float
            The computed z-score for this test.

        pvalue: float
            The two-sided p-value for the hypothesis test.        

        """              
        self._k = kurtosis(a, axis=axis, bias=bias, fisher=fisher)
        self._statistic, self._p = kurtosistest(a, axis=axis)

    @property
    def kurtosis(self):
        return self._k

    def get_result(self):
        return self._k, self._statistic, self._p

    def print(self):
        result = {'Kurtosis': [self._k], 'Z Score': [self._statistic], 
                  'p_value' : [self._p]}
        print(tabulate(result, headers='keys'))        

# --------------------------------------------------------------------------- #
#                                Bartlett                                     #
# --------------------------------------------------------------------------- #
class Bartlett(AbstractStatisticalTest):
    """Perform Bartlett’s test for equal variances
    
    Bartlett’s test tests the null hypothesis that all input samples are 
    from populations with equal variances.
    
    If bias is False then the kurtosis is calculated using k statistics 
    to eliminate bias coming from biased moment estimators
    
    This class also uses kurtosistest to see if result is close enough to normal.

    Attributes
    -------
    T : float
        The test statistic

    pvalue: float
        The p-value for the test.        

    """

    def __init__(self):
        super(Bartlett, self).__init__()

    def fit(self,*args):    
        """Perform Bartlett’s test for equal variances
        
        Parameters
        -------
        sample1, sample2,... : array_like
            arrays of sample data. May be different lengths.
        """            
        self._statistic, self._p = bartlett(*args)

    def print(self):
        result = {'T Statistic': [self._statistic], 'p_value' : [self._p]}
        print(tabulate(result, headers='keys'))                