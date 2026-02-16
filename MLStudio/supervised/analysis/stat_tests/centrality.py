#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : Data Studio                                                       #
# Version : 0.1.0                                                             #
# File    : centrality.py                                                     #
# Python  : 3.8.1                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/datastudio                     #
# --------------------------------------------------------------------------- #
# Created       : Thursday, February 20th 2020, 3:26:36 am                    #
# Last Modified : Saturday, February 29th 2020, 5:45:05 pm                    #
# Modified By   : John James (jjames@decisionscients.com>)                    #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Module defines the suite of statistical tests of centrality.

This package supports parametric and non-parametric tests of centrality of a 
single sample or the differences in central tendency between two groups. The 
parametric tests include:

    * TTestOne : One-Sample t-test
    * MedianTest : One-Sample Median Test
    * TTestPaired : Paired t-test
    * TTestInd : 2 Independent t-tests
    * AnovaR: One-way Repeated ANOVA Tests
    * TODO: Factorial Anova    

The non-parametric tests include:
    * Kruskal : Kruskal-Wallis H-test
    * MannWhitney : Wilcoxon-Mann Whitney Test
    * Wilcoxon : Wilcoxon Signed Rank Test

Note: This module makes liberal use of three statistical software packages.

    * SciPy : A Python-based ecosystem for mathematics, science and engineering.
    * Statsmodels : Statistical models, hypothesis tests and data exploration 
    * scikit-learn : A machine learning platform for Python.

Each of the classes above comply with an Abstract Base Class which defines
the interface for all test classes. 

"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, median_test, zscore, ttest_rel
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, f_oneway
from scipy.stats import kruskal
from statsmodels.stats.anova import AnovaRM
from tabulate import tabulate

from datastudio.understanding.stat_tests.interface import AbstractStatisticalTest
# --------------------------------------------------------------------------- #
#                            One Sample t-test                                #
# --------------------------------------------------------------------------- #
class TTestOne(AbstractStatisticalTest):
    """Calculates the T-test for the mean of ONE group of scores.
    
    Attributes
    ----------

    """

    def __init__(self):
        super(TTestOne, self).__init__()    

    def fit(self, a, popmean, axis=0):        
        """Calculates the T-test for the mean of ONE group of scores.
        
        Parameters
        ----------
        a : array_like
            sample observation
        popmean : float or array_like
            expected value in null hypothesis, if array_like than 
            it must have the same shape as a excluding the axis dimension
        axis : int or None, optional
            Axis along which to compute test. If None, compute over the whole array a.
        
        """
        self._statistic, self._p = ttest_1samp(a, popmean, axis)

    def print(self):
        result = {'t-statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                            One Sample Median Test                           #
# --------------------------------------------------------------------------- #
class MedianTest(AbstractStatisticalTest):
    """Perform a Mood’s median test.

    Test that two or more samples come from populations with the same median.

    Let n = len(args) be the number of samples. The “grand median” of all the 
    data is computed, and a contingency table is formed by classifying the 
    values in each sample as being above or below the grand median. The 
    contingency table, along with correction and lambda_, are passed to 
    scipy.stats.chi2_contingency to compute the test statistic and p-value.

    Attributes
    ----------
    statistic : float
        The test statistic. The statistic that is returned is determined by 
        lambda_. The default is Pearson’s chi-squared statistic.
    p_value : float
        The p-value of the test.
    grand_median : float
        The grand median.
    contingency_table : ndarray
        The contingency table. The shape of the table is (2, n), where 
        n is the number of samples. The first row holds the counts of 
        the values above the grand median, and the second row holds the 
        counts of the values below the grand median. The table allows 
        further analysis with, for example, scipy.stats.chi2_contingency, 
        or with scipy.stats.fisher_exact if there are two samples, without 
        having to recompute the table. If nan_policy is “propagate” and there 
        are nans in the input, the return value for table is None.
    """

    def __init__(self):
        super(MedianTest, self).__init__()   
        self._m = 0
        self._ctable = 0

    def fit(self,*args, **kwargs):   
        """Perform a Mood’s median test.

        Parameters
        ----------
        sample1, sample2, … : array_like
            The set of samples. There must be at least two samples. Each 
            sample must be a one-dimensional sequence containing at least 
            one value. The samples are not required to have the same length.
        ties : str, optional
            Determines how values equal to the grand median are classified 
            in the contingency table. The string must be one of:
            "below":
                Values equal to the grand median are counted as "below".
            "above":
                Values equal to the grand median are counted as "above".
            "ignore":
                Values equal to the grand median are not counted.
            The default is “below”.
        correction : bool, optional
            If True, and there are just two samples, apply Yates’ correction for 
            continuity when computing the test statistic associated with the 
            contingency table. Default is True.
        lambda_ : float or str, optional
            By default, the statistic computed in this test is Pearson’s 
            chi-squared statistic. lambda_ allows a statistic from the 
            Cressie-Read power divergence family to be used instead. 
            See power_divergence for details. Default is 1 
            (Pearson’s chi-squared statistic).
        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. ‘propagate’ returns 
            nan, ‘raise’ throws an error, ‘omit’ performs the calculations ignoring 
            nan values. Default is ‘propagate’.        
        """             
        self._statistic, self._p, self._m, self._ctable = median_test(*args, **kwargs)

    def get_result(self):
        """Returns results of a Mood’s median test.

        Returns
        ----------
        stat : float
            The test statistic. The statistic that is returned is determined by 
            lambda_. The default is Pearson’s chi-squared statistic.
        p : float
            The p-value of the test.
        m : float
            The grand median.
        table : ndarray
            The contingency table. The shape of the table is (2, n), where 
            n is the number of samples. The first row holds the counts of 
            the values above the grand median, and the second row holds the 
            counts of the values below the grand median. The table allows 
            further analysis with, for example, scipy.stats.chi2_contingency, 
            or with scipy.stats.fisher_exact if there are two samples, without 
            having to recompute the table. If nan_policy is “propagate” and there 
            are nans in the input, the return value for table is None.
        """            
        return self._statistic, self._p, self._m, self._ctable

    @property
    def grand_median(self):
        return self._m

    @property
    def contingency_table(self):
        return self._ctable        

    def print(self):
        result = {'Test Statistic': [self._statistic], 'p-value': [self._p],
                  'Grand Median': [self._m], 'Contingency Table': self._ctable}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                        Two Dependent Sample t-test                          #
# --------------------------------------------------------------------------- #
class TTestPaired(AbstractStatisticalTest):
    """Calculate the t-test on TWO RELATED samples of scores, a and b.
    
    This is a two-sided test for the null hypothesis that 2 related or repeated 
    samples have identical average (expected) values.

    Attributes
    ----------
    statistic : float or array
        t-statistic.
    pvalue : float or array
        Two-sided p-value.
    """

    def __init__(self):
        super(TTestPaired, self).__init__()

    def fit(self, a, b, axis=0, nan_policy='propagate'):        
        """Calculate the t-test on TWO RELATED samples of scores, a and b.
        
        Parameters
        ----------
        a, b : array_like
            The arrays must have the same shape.
        axis : int or None, optional
            Axis along which to compute test. If None, compute over the 
            whole arrays, a, and b.
        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. The following 
            options are available (default is ‘propagate’):
                ‘propagate’: returns nan
                ‘raise’: throws an error
                ‘omit’: performs the calculations ignoring nan values
        
        """
        self._statistic, self._p = ttest_rel(a=a, b=b, axis=axis, 
                                            nan_policy=nan_policy)

    def print(self):
        result = {'t-statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))

# --------------------------------------------------------------------------- #
#                        Two Independent Sample t-test                        #
# --------------------------------------------------------------------------- #
class TTestInd(AbstractStatisticalTest):
    """Calculate the T-test for the means of two independent samples of scores.
    
    This is a two-sided test for the null hypothesis that 2 independent 
    samples have identical average (expected) values. This test assumes that 
    the populations have identical variances by default.

    Attributes
    ----------
    statistic : float or array
        t-statistic.
    pvalue : float or array
        Two-sided p-value.
    """

    def __init__(self):
        super(TTestInd, self).__init__()

    def fit(self, a, b, axis=0, equal_var=True, nan_policy='propagate'):        
        """Calculate the t-test on TWO RELATED samples of scores, a and b.
        
        Parameters
        ----------
        a, b : array_like
            The arrays must have the same shape.
        axis : int or None, optional
            Axis along which to compute test. If None, compute over the 
            whole arrays, a, and b.
        equal_var : bool, optional
            If True (default), perform a standard independent 2 sample 
            test that assumes equal population variances [1]_. If False, 
            perform Welch’s t-test, which does not assume equal population 
            variance [2]_.         
        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. The following 
            options are available (default is ‘propagate’):
                ‘propagate’: returns nan
                ‘raise’: throws an error
                ‘omit’: performs the calculations ignoring nan values

        Returns
        ----------
        statistic : float or array
            t-statistic.
        pvalue : float or array
            Two-sided p-value.                

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        .. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test
        
        """
        self._statistic, self._p = ttest_ind(a=a, b=b, axis=axis, 
                                             equal_var=equal_var, 
                                             nan_policy=nan_policy)

    def print(self):
        result = {'t-statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))        

# --------------------------------------------------------------------------- #
#                             One-Way ANOVA Test                              #
# --------------------------------------------------------------------------- #        
class AnovaOne(AbstractStatisticalTest):
    """ Perform one-way ANOVA.

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean.  The test is applied to samples from two or
    more groups, possibly with differing sizes.
    
    Attributes
    ----------
    statistic : float
        The computed F-value of the test.
    pvalue : float
        The associated p-value from the F-distribution.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.
    1. The samples are independent.
    2. Each sample is from a normally distributed population.
    3. The population standard deviations of the groups are all equal.  This
       property is known as homoscedasticity.
    
    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`) although
    with some loss of power.
    The algorithm is from Heiman[2], pp.394-7.
    
    References
    ----------
    .. [1] R. Lowry, "Concepts and Applications of Inferential Statistics",
           Chapter 14, 2014, http://vassarstats.net/textbook/
    .. [2] G.W. Heiman, "Understanding research methods and statistics: An
           integrated introduction for psychology", Houghton, Mifflin and
           Company, 2001.
    .. [3] G.H. McDonald, "Handbook of Biological Statistics", One-way ANOVA.
           http://www.biostathandbook.com/onewayanova.html        
    """

    def __init__(self):
        super(AnovaOne, self).__init__()

    def fit(self, *args):        
        """Perform one-way ANOVA.
        
        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        ----------
        statistic : float
            The computed F-value of the test.
        pvalue : float
            The associated p-value from the F-distribution.        

        """
        self._statistic, self._p = f_oneway(*args)

    def print(self):
        result = {'F-statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))        

# --------------------------------------------------------------------------- #
#                       One-Way ANOVA Repeated Metrics                       #
# --------------------------------------------------------------------------- #
class AnovaR(AbstractStatisticalTest):
    """ Repeated measures Anova using least squares regression

    The full model regression residual sum of squares is
    used to compare with the reduced model for calculating the
    within-subject effect sum of squares [1]_.

    Currently, only fully balanced within-subject designs are supported.
    Calculation of between-subject effects and corrections for violation of
    sphericity are not yet implemented.       

    References
    ----------
    .. [1] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.

    """

    def __init__(self):
        super(AnovaR, self).__init__()
        self._results = None

    def fit(self, data, depvar, subject, within=None, between=None,
                 aggregate_func=None):        
        """Estimate the model and compute ANOVA table.
        
        Parameters
        ----------
        data : DataFrame
        depvar : str
            The dependent variable in `data`
        subject : str
            Specify the subject id
        within : list[str]
            The within-subject factors
        between : list[str]
            The between-subject factors, this is not yet implemented
        aggregate_func : {None, 'mean', callable}
            If the data set contains more than a single observation per subject
            and cell of the specified model, this function will be used to
            aggregate the data before running the Anova. `None` (the default) will
            not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
            An exception will be raised if aggregation is required, but no
            aggregation function was specified.

        Returns
        -------
        results : AnovaResults instance

        Raises
        ------
        ValueError
            If the data need to be aggregated, but `aggregate_func` was not
            specified.

        """
        anova = AnovaRM(data=data, depvar=depvar, subject=subject, within=within,
                        between=between, aggregate_func=aggregate_func)
        self._results = anova.fit()        
        

    def print(self):
        print(self._results)        
# =========================================================================== #
#                             NON-PARAMETRIC TESTS                            #
# =========================================================================== #        
# --------------------------------------------------------------------------- #
#                             Kruskal-Wallis Test                             #
# --------------------------------------------------------------------------- #
class Kruskal(AbstractStatisticalTest):
    """Compute the Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the population 
    median of all of the groups are equal. It is a non-parametric version of 
    ANOVA. The test works on 2 or more independent samples, which may have 
    different sizes. Note that rejecting the null hypothesis does not indicate 
    which of the groups differs. Post hoc comparisons between groups are 
    required to determine which groups are different.
    
    """

    def __init__(self):
        super(Kruskal, self).__init__()         

    def fit(self,*args, **kwargs):    
        """ Performs the Kruskal_Wallis H-test.

        Parameters
        ----------    
        sample1, sample2, … : array_like
            Two or more arrays with the sample measurements can be given 
            as arguments.
        nan_policy : {‘propagate’, ‘raise’, ‘omit’}, optional
            Defines how to handle when input contains nan. The following options 
            are available (default is ‘propagate’):
                 - ‘propagate’: returns nan
                 - ‘raise’: throws an error
                 - ‘omit’: performs the calculations ignoring nan values        
        """
        self._statistic, self._p = kruskal(*args, **kwargs)
# --------------------------------------------------------------------------- #
#                           Mann-Whitney Rank Test                            #
# --------------------------------------------------------------------------- #
class MannWhitney(AbstractStatisticalTest):
    """Compute the Mann-Whitney rank test on samples x and y.

    Attributes
    -------
    statistic : float
        The Mann-Whitney U statistic, equal to min(U for x, U for y) if
        `alternative` is equal to None (deprecated; exists for backward
        compatibility), and U for y otherwise.
    pvalue : float
        p-value assuming an asymptotic normal distribution. One-sided or
        two-sided, depending on the choice of `alternative`.
    Notes
    -----
    Use only when the number of observation in each sample is > 20 and
    you have 2 independent samples of ranks. Mann-Whitney U is
    significant if the u-obtained is LESS THAN or equal to the critical
    value of U.
    This test corrects for ties and by default uses a continuity correction.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mann-Whitney_U_test
    .. [2] H.B. Mann and D.R. Whitney, "On a Test of Whether one of Two Random
           Variables is Stochastically Larger than the Other," The Annals of
           Mathematical Statistics, vol. 18, no. 1, pp. 50-60, 1947.
    """

    def __init__(self):
        super(MannWhitney, self).__init__()

    def fit(self, x, y, use_continuity=True, alternative=None):        
        """Compute the Mann-Whitney rank test on samples x and y.
        Parameters
        ----------
        x, y : array_like
            Array of samples, should be one-dimensional.
        use_continuity : bool, optional
                Whether a continuity correction (1/2.) should be taken into
                account. Default is True.
        alternative : {None, 'two-sided', 'less', 'greater'}, optional
            Defines the alternative hypothesis.
            The following options are available (default is None):
            * None: computes p-value half the size of the 'two-sided' p-value and
                a different U statistic. The default behavior is not the same as
                using 'less' or 'greater'; it only exists for backward compatibility
                and is deprecated.
            * 'two-sided'
            * 'less': one-sided
            * 'greater': one-sided
            Use of the None option is deprecated.
        Returns
        -------
        statistic : float
            The Mann-Whitney U statistic, equal to min(U for x, U for y) if
            `alternative` is equal to None (deprecated; exists for backward
            compatibility), and U for y otherwise.
        pvalue : float
            p-value assuming an asymptotic normal distribution. One-sided or
            two-sided, depending on the choice of `alternative`.
            
        """
        self._statistic, self._p = mannwhitneyu(x,y,
                                                use_continuity=use_continuity,
                                                alternative=alternative)        

    def print(self):
        result = {'U-statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))                                                    

# --------------------------------------------------------------------------- #
#                       Wilcoxon Signed-Rank Test                             #
# --------------------------------------------------------------------------- #
class Wilcoxon(AbstractStatisticalTest):
    """Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Notes
    -----
    The test has been introduced in [4]_. Given n independent samples
    (xi, yi) from a bivariate distribution (i.e. paired samples),
    it computes the differences di = xi - yi. One assumption of the test
    is that the differences are symmetric, see [2]_.
    The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from
    zero. The one-sided test has the null hypothesis that the median is 
    positive against the alternative that it is negative 
    (``alternative == 'less'``), or vice versa (``alternative == 'greater.'``).
    The test uses a normal approximation to derive the p-value (if
    ``zero_method == 'pratt'``, the approximation is adjusted as in [5]_).
    A typical rule is to require that n > 20 ([2]_, p. 383). For smaller n,
    exact tables can be used to find critical values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`
    """

    def __init__(self):
        super(Wilcoxon, self).__init__()

    def fit(self, x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided"):        
        """Calculate the Wilcoxon signed-rank test.
    
        Parameters
        ----------
        x : array_like
            Either the first set of measurements (in which case `y` is the second
            set of measurements), or the differences between two sets of
            measurements (in which case `y` is not to be specified.)  Must be
            one-dimensional.
        y : array_like, optional
            Either the second set of measurements (if `x` is the first set of
            measurements), or not specified (if `x` is the differences between
            two sets of measurements.)  Must be one-dimensional.
        zero_method : {'pratt', 'wilcox', 'zsplit'}, optional
            The following options are available (default is 'wilcox'):
        
            * 'pratt': Includes zero-differences in the ranking process,
                but drops the ranks of the zeros, see [4]_, (more conservative).
            * 'wilcox': Discards all zero-differences, the default.
            * 'zsplit': Includes zero-differences in the ranking process and 
                split the zero rank between positive and negative ones.
        correction : bool, optional
            If True, apply continuity correction by adjusting the Wilcoxon rank
            statistic by 0.5 towards the mean value when computing the
            z-statistic.  Default is False.
        alternative : {"two-sided", "greater", "less"}, optional
            The alternative hypothesis to be tested, see Notes. Default is
            "two-sided".

        Returns
        -------
        statistic : float
            If `alternative` is "two-sided", the sum of the ranks of the
            differences above or below zero, whichever is smaller.
            Otherwise the sum of the ranks of the differences above zero.
        pvalue : float
            The p-value for the test depending on `alternative`.            
            
        """
        self._statistic, self._p = wilcoxon(x,y,zero_method=zero_method,
                                                correction=correction,
                                                alternative=alternative)        

    def print(self):
        result = {'Statistic': [self._statistic], 'p-value': [self._p]}
        print(tabulate(result, headers='keys'))                           