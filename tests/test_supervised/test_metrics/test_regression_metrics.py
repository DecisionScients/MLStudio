# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_regression_metrics.py                                       #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 22nd 2020, 3:46:43 pm                       #
# Last Modified : Wednesday, July 22nd 2020, 3:46:43 pm                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests Regression Metric classes."""
#%%
import math
import numpy as np
import pytest
from pytest import mark
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score

from mlstudio.supervised.metrics import regression 
# --------------------------------------------------------------------------- #
@mark.metrics
@mark.regression_metrics
class RegressionMetricsTests:

    def test_r2(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.R2()(y, y_pred)         
        skl = r2_score(y, y_pred)   
        assert x<=1, "R2 is not less than 1"
        assert np.isclose(x,skl,rtol=1e-2), "R2 not close to sklearn value"

    def test_var_explained(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.VarExplained()(y, y_pred)        
        skl = explained_variance_score(y, y_pred)
        assert x<=1, "Variance explained not between 0 and 1"        
        assert np.isclose(x,skl,rtol=1e-2), "Variance explained not close to sklearn value"

    def test_mae(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.MAE()(y, y_pred)        
        skl = mean_absolute_error(y, y_pred)
        assert x>0, "MAE is not positive"       
        assert np.isclose(x,skl,rtol=1e-2), "Mean absolute error not close to sklearn value" 

    def test_mse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.MeanSquaredError()(y, y_pred)        
        skl = mean_squared_error(y, y_pred)
        assert isinstance(x, float), "MSE is not a float"        
        assert x > 0, "MSE is not positive"
        assert np.isclose(x,skl,rtol=1e-2), "Mean squared error not close to sklearn value"

    def test_nmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.NegativeMeanSquaredError()(y, y_pred)      
        skl = -1*mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "NMSE is not a float"                
        assert x < 0, "NMSE is not negative"
        assert np.isclose(x,skl,rtol=1e-2), "Negative mean squared error not close to sklearn value"

    def test_rmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.RootMeanSquaredError()(y, y_pred)      
        skl = mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "RMSE is not a float"                
        assert x > 0, "RMSE is not positive"        
        assert np.isclose(x,np.sqrt(skl),rtol=1e-2), "root mean squared error not close to sklearn value"

    def test_nrmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.NRMSE()(y, y_pred)       
        skl = mean_squared_error(y, y_pred)   
        assert isinstance(x, float), "NRMSE is not a float"                
        assert x < 0, "NRMSE is not negative"         
        assert np.isclose(x,-np.sqrt(skl),rtol=1e-2), "negative root mean squared error not close to sklearn value"

    def test_msle(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.MSLE()(y, y_pred) 
        if all(y_pred > 0) and (y > 0):
            skl = mean_squared_log_error(y, y_pred)  
            assert x > 0, "MSLE is not  positive"
            assert np.isclose(x,skl,rtol=1e-2), "Mean squared log error not close to sklearn value" 
        else:
            print("\nUnable to compute MSLE with negative targets.")                               

    def test_rmsle(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.RMSLE()(y, y_pred)        
        if all(y_pred > 0) and (y > 0):
            skl = np.sqrt(mean_squared_log_error(y, y_pred))
            assert x > 0, "RMSLE is not  positive"
            assert np.isclose(x,skl,rtol=1e-2), "Root mean squared log error not close to sklearn value" 
        else:
            print("\nUnable to compute RMSLE with negative targets.")                               

    def test_medae(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = regression.MEDAE()(y, y_pred)        
        skl = median_absolute_error(y, y_pred)
        assert isinstance(x, float), "MEDAE is not a float"                
        assert x > 0, "MEDAE is not  positive"          
        assert np.isclose(x,skl,rtol=1e-2), "Median absolute error not close to sklearn value"

    def test_ar2(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        n = X.shape[0]
        p = X.shape[1]
        x = regression.AdjustedR2()(y, y_pred, p)        
        skl = r2_score(y, y_pred)
        skl_ar2 = 1 - (1-skl) * (n-1)/(n-p-1)
        assert isinstance(x, float), "Adjusted R2 is not a float"                
        assert x > 0, "Adjusted R2 is not  positive"          
        assert np.isclose(x,skl_ar2,rtol=1e-2), "Median absolute error not close to sklearn value"
