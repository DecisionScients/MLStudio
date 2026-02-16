#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : model_validation.py                                               #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, March 20th 2020, 10:16:08 pm                        #
# Last Modified : Friday, March 20th 2020, 10:16:08 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Functions used to analyze model validity."""

import math
from math import erf
import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- #
#                           RESIDUAL ANALYSIS                                 #
# --------------------------------------------------------------------------- #
def standardized_residuals(X, y, y_pred):
    """Computes standardized residuals.
    
    Standardized residuals (sometimes referred to as "internally studentized 
    residuals") are defined for each observation, i = 1, ..., n as an 
    ordinary residual divided by an estimate of its standard deviation:
    ..math:: r_i = \\frac{e_i}{\\sqrt{MSE(1-h_{ii})}}

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    
    y : ndarray or Series of length n
        An array or series of target or class values 

    y_pred : ndarray or Series of length n
        An array or series of predictions         
    """
    # Compute residuals
    residuals = y - y_pred
    
    # Compute Leverage
    hii = leverage(X)
    
    # Compute degrees of freedom and MSE
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank
    mse = np.matmul(residuals, residuals) / df

    # Calculate standardized 
    standardized_residuals = residuals / np.sqrt(mse  * (1-hii))
    
    return standardized_residuals

def studentized_residuals(X, y, y_pred):
    """Computes studentized residuals.

    Studentized residuals are just a deleted residual divided by its estimated
    standard deviation. This turns out to be equivalent to the ordinary residual
    divided by a factor that includes the mean square error based on the 
    estimated model with the ith observation deleted, MSE(i), and the leverage, hii
    .. math:: r_i = \\frac{e_i}{\\sqrt{MSE_{(i)}(1-h_{ii})}}
    
    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    
    y : ndarray or Series of length n
        An array or series of target or class values 

    y_pred : ndarray or Series of length n
        An array or series of predictions         
    """    
    # Using the calculation from 
    # https://newonlinecourses.science.psu.edu/stat462/node/247/
    n = X.shape[0]
    k = X.shape[1]
    r = standardized_residuals(X=X, y=y, y_pred=y_pred)

    # Calculate studentized residuals 
    studentized_residuals = r * np.sqrt((n-k-2)/(n-k-1-np.square(r)))

    return studentized_residuals   

# --------------------------------------------------------------------------- #
#                               QUANTILES                                     #
# --------------------------------------------------------------------------- #
def quantile(p):
    """Computes the theoretical quantiles for random variable.

    Parameters
    ----------
    p : array-like of real-valued numbers
        The vector of real numbers for which the theoretical quantiles 
        are computed.


    """
    j = np.arange(len(p)) + 1
    j_star = (j-0.5) / len(p)
    return norm.ppf(j_star)    
     
def rational_approximation(t):

    # Abramowitz and Stegun formula 26.2.23.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    numerator = (c[2]*t + c[1])*t + c[0]
    denominator = ((d[2]*t + d[1])*t + d[0])*t + 1.0
    return t - numerator / denominator
    

def normal_CDF_inverse(p):

    assert p > 0.0 and p < 1

    # See article above for explanation of this section.
    if p < 0.5:
        # F^-1(p) = - G^-1(p)
        return -rational_approximation( math.sqrt(-2.0*math.log(p)) )
    else:
        # F^-1(p) = G^-1(1-p)
        return rational_approximation( math.sqrt(-2.0*math.log(1.0-p)) )   
# --------------------------------------------------------------------------- #
#                           INFLUENCE MEASURES                                #
# --------------------------------------------------------------------------- #
def leverage(X):
    """Computes leverage scores for a data set.

    Leverage is a measure of how much the independent variables of an 
    observation differs from the other observations. The leverage score for
    the ith osbservation is defined as:
    .. math:: h_{ii} = 
    the ith diagonal element of the project matrix \\mathbb{H}=X(X^TX)^{-1}X^T,
    where X is the design matrix whose rows are observations and columns
    are the independent variables.
    
    The diagonal of the projection matrix, commonly known as the hat matrix, 
    is a standardized measure of the distance from the ith observation from
    the center (or centroid) of the x-space.
    
    Points with leverage greater than \\frac{2p}{n}, where p is the number
    independent variables, including the bias, and n is the number of 
    observations, are considered remote enough from the rest of the 
    data to be designated a leverage point. [2]_
    
    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    
    Returns
    -------
    leverage : ndarray or DataFrame of shape n x 1
        Contains the leverage scores for each observation.
    """

    hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    hii = np.diagonal(hat)
    return hii

def cooks_distance(X, y, y_pred):
    """Computes Cook's Distance, a commonly used measure of data point influence.
    
    Cook's Distance is used in least-squares regression analysis to identify 
    influencial data points. Cook's distance for a given data point is given by:
    .. math:: \\mathbb{D}_i = \\frac{\\epsilon^2_i}{ps^2}\\big[\\frac{h_{ii}}{(1-h_{ii})^2}][1]_
    
    An alternative formulation relates Cooks Distance to studentized
    residuals. 
    .. math:: \\mathbb{D}_i =\\bigg[\\frac{1}{p}\\frac{n}{n-p}]t_i^2\\frac{h_{ii}}{1-h{ii}}
    
    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    
    y : ndarray or Series of length n
        An array or series of target or class values 

    y_pred : ndarray or Series of length n
        An array or series of predictions         
    
    Returns
    -------
    cooks_distance : ndarray or DataFrame of shape n x 1
        A matrix of cooks distances computed for each observation
    
    Reference
    ---------
    .. [1]  Wikipedia contributors. (2019, October 24). Cook's distance. 
            In Wikipedia, The Free Encyclopedia. Retrieved 01:32, 
            November 29, 2019, from 
            https://en.wikipedia.org/w/index.php?title=Cook%27s_distance&oldid=922838890
    """
    # Compute residual
    e = y - y_pred
    # Set number of observations and predictors
    n = X.shape[0]
    p = X.shape[1]
    # S squared 
    s2 = np.matmul(e.T,e) / (n-p)
    # Compute leverage
    hii = leverage(X)
    # Obtain studentized residuals 
    cooks_d = (e**2/ (p * s2)) * (hii/(1-hii)**2)    
    return cooks_d

def dffits(X, y, y_pred):
    """Computes the difference in fits (DFFITS).

    The difference in fits for observation i, denoted (\\DFFITS_i), is defined
    as:
    .. math:: \\DFFITS_i = \\frac{\\hat{y}_i-\\hat{y}_{(i)}}{\\sqrt{MSE_{(i)}h_{ii}}}.
    
    The numerator measures the difference in the predicted response obtained
    with and without the ith data point. The denominator is the estimated 
    standard deviation of the difference between predicted responses. Hence,
    the diffeence in fits quantifies the number of standard deviations that 
    the fitted value changes when the ith observation is omitted.
    
    An observation is deemed influential if the absolute value of its DFFITS
    value is greater than:
    .. math:: 2\\sqrt{\\frac{p+1}{n-p-1}}
    
    where n is the number of observations, p is the number of predictors 
    including the bias term.
    
    DFFITS is also related to the students residual, and is in fact the latter
    times 
    
    .. math:: \\sqrt(\\frac{h_{ii}}{1-h_{ii}})[1]_
    
    Observations with DFFITS greater that 2\\sqrt{\\frac{p}{n}}, where p is the 
    number of predictors including the bias and n is the number of observations,
    should be investigated.[2]_
    
    Parameters
    ----------
    model : Estimator or BaseEstimator
        ML Studio or Scikit Learn estimator
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values 
    
    References
    ----------    
    .. [1] Montogomery, Douglas C.; Peck, Elizabeth A.; Vining, G. 
           Geoffrey (2012). Introduction to Linear Regression Analysis 
           (5th ed.). Wiley. p. 218. ISBN 978-0-470-54281-1. Retrieved 
           22 February 2013. Thus, DFFITSi is the value of R-student 
           multiplied by the leverage of the ith observation [hii/(1-hii)]1/2.
    .. [2] Belsley, David A.; Kuh, Edwin; Welsh, Roy E. (1980). 
           Regression Diagnostics: Identifying Influential Data and 
           Sources of Collinearity. Wiley Series in Probability 
           and Mathematical Statistics. New York: John Wiley & Sons. 
           pp. 11–16. ISBN 0-471-05856-4.
    """
    r_student = studentized_residuals(X, y, y_pred)
    hii = leverage(X)
    df = r_student * np.sqrt(hii/(1-hii))
    return df