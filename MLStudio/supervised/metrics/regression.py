# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \regression.py                                                    #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:25:57 am                        #
# Last Modified : Thursday, July 16th 2020, 2:25:58 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
import math
import numpy as np

from mlstudio.supervised.metrics.base import BaseRegressionMetric
# --------------------------------------------------------------------------- #
class ResidualSumSquaredError(BaseRegressionMetric):    
    """Computes sum squared residuals given"""

    _mode  = 'min'
    _code = 'SSR'
    _name  = 'residual_sum_squared_error'
    _label  = "Residual Sum Squared Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1

    def __call__(self, y, y_pred, *args, **kwargs):
        e = y - y_pred
        return np.sum(e**2)  

class TotalSumSquaredError(BaseRegressionMetric):
    """Computes total sum of squares"""

    
    _mode  = 'min'
    _code = "SST"
    _name  = 'total_sum_squared_error'
    _label  = "Total Sum Squared Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1

    
    def __call__(self, y, y_pred, *args, **kwargs):
        y_avg = np.mean(y)
        e = y-y_avg                
        return np.sum(e**2)

class R2(BaseRegressionMetric):
    """Computes coefficient of determination."""

    
    _mode  = 'max'   
    _code = "R2"
    _name  = 'R2'
    _label  = 'Coefficient of Determination (R2)'
    
    _best  = max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
        self._ssr = ResidualSumSquaredError()
        self._sst = TotalSumSquaredError()
        r2 = 1 - (self._ssr(y, y_pred)/self._sst(y, y_pred))     
        return r2


class AdjustedR2(BaseRegressionMetric):
    """Computes adjusted coefficient of determination."""

    
    _mode  = 'max'   
    _code = "AR2"
    _name  = 'adjusted_r2'
    _label  = "Adjusted R2"
    
    _best  = max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1

    
    def __call__(self, y, y_pred, n_features, *args, **kwargs):
        r2_scorer = R2()
        r2 = r2_scorer(y, y_pred)
        n = y.shape[0]
        p = n_features
        ar2 = 1 - (1 - r2) * (n-1) / (n-p-1)
        return ar2

class PercentVarianceExplained(BaseRegressionMetric):
    """Computes proportion of variance explained."""

    
    _mode  = 'max'
    _code = "PVE"
    _name  = 'percent_variance_explained'
    _label  = "Percent Variance Explained"
    
    _best  = max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
        var_explained = 1 - (np.var(y-y_pred) / np.var(y))
        return var_explained * 100                   

class MeanAbsoluteError(BaseRegressionMetric):
    """Computes mean absolute error given data and parameters."""

    
    _mode  = 'min'
    _code = "MAE"
    _name  = 'mean_absolute_error'
    _label  = "Mean Absolute Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
        e = abs(y-y_pred)
        return np.mean(e)


class MeanSquaredError(BaseRegressionMetric):
    """Computes mean squared error given data and parameters."""

    
    _mode  = 'min'
    _code = "MSE"
    _name  = 'mean_squared_error'
    _label  = "Mean Squared Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
        e = y - y_pred
        return np.mean(e**2)

class NegativeMeanSquaredError(BaseRegressionMetric):
    """Computes negative mean squared error given data and parameters."""

    
    _mode  = 'max'
    _code = "NMSE"
    _name  = 'negative_mean_squared_error'
    _label  = "Negative Mean Squared Error"
    
    _best  = max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):        
        e = y - y_pred
        return -np.mean(e**2)

class RootMeanSquaredError(BaseRegressionMetric):
    """Computes root mean squared error given data and parameters."""

    
    _mode  = 'min'
    _code = "RMSE"
    _name  = 'root_mean_squared_error'
    _label  = "Root Mean Squared Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
        e = y-y_pred
        return np.sqrt(np.mean(e**2)) 

class NegativeRootMeanSquaredError(BaseRegressionMetric):
    """Computes negative root mean squared error given data and parameters."""

    
    _mode  = 'max'
    _code = "NRMSE"
    _name  = 'negative_root_mean_squared_error'
    _label  = "Negative Root Mean Squared Error"
    
    _best  = max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
        e = y-y_pred
        return -np.sqrt(np.mean(e**2))

class MeanSquaredLogError(BaseRegressionMetric):
    """Computes mean squared log error given data and parameters."""

    
    _mode  = 'min'
    _code = "MSLE"
    _name  = 'mean_squared_log_error'
    _label  = "Mean Squared Log Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
        e = np.log(y+1)-np.log(y_pred+1)
        y = np.clip(y, 1e-15, 1-1e-15)    
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)    
        e = np.log(y)-np.log(y_pred)
        return np.mean(e**2)

class MedianAbsoluteError(BaseRegressionMetric):
    """Computes median absolute error given data and parameters."""

    
    _mode  = 'min'
    _code = "MdAE"
    _name  = 'median_absolute_error'
    _label  = "Median Absolute Error"
    
    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
        return np.median(np.abs(y_pred-y))

class MeanAbsolutePercentageError(BaseRegressionMetric):
    """Computes mean absolute percentage given data and parameters."""
    _mode  = 'min'
    _code = "MAPE"
    _name  = 'mean_absolute_percentage_error'
    _label  = "Mean Absolute Percentage Error"

    _best  = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
        return 100*np.mean(np.abs((y-y_pred)/y))

