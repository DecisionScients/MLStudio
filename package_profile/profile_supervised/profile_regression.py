#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : profile_regression.py                                             #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 2nd 2020, 5:35:27 pm                          #
# Last Modified : Saturday, May 2nd 2020, 5:35:27 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Profiles linear regression."""
import cProfile
import pstats
from pstats import SortKey
import re
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from mlstudio.supervised.machine_learning.linear_regression import LinearRegression
from mlstudio.supervised.machine_learning.gradient_descent import GDRegressor

# --------------------------------------------------------------------------- #
#                                  DATA                                       #
# --------------------------------------------------------------------------- #  
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

# --------------------------------------------------------------------------- #
#                       PROFILE LINEAR REGRESSION                             #
# --------------------------------------------------------------------------- #  
def profile_linear_regression():    
    X, y = get_regression_data()
    lr = GDRegressor(algorithm=LinearRegression())        
    pr = cProfile.Profile()
    pr.enable()
    lr.fit(X,y)
    pr.disable()
    pr.print_stats(sort='time')

# --------------------------------------------------------------------------- #
#                       PROFILE LINEAR REGRESSION SGD                         #
# --------------------------------------------------------------------------- #  
def profile_linear_regression_sgd():    
    X, y = get_regression_data()
    lr = GDRegressor(algorithm=LinearRegression(), batch_size=1)        
    pr = cProfile.Profile()
    pr.enable()
    lr.fit(X,y)
    pr.disable()
    pr.print_stats(sort='time')
profile_linear_regression_sgd()
#%%
#%%