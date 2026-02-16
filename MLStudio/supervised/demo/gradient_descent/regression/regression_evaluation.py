#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : regression.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 3:27:23 pm                         #
# Last Modified : Wednesday, June 10th 2020, 9:11:49 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
from collections import OrderedDict
import os
from pathlib import Path
import sys
homedir = str(Path(__file__).parents[4])
demodir = str(Path(__file__).parents[2])
datadir = os.path.join(demodir, "data")
sys.path.append(homedir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

from mlstudio.supervised.machine_learning.gradient_descent import GDRegressor
from mlstudio.supervised.algorithms.optimization.services.loss import Quadratic
from mlstudio.supervised.algorithms.optimization.services.regularizers import L1, L2, L1_L2
from mlstudio.supervised.metrics.regression import R2
from mlstudio.supervised.algorithms.optimization.observers.early_stop import EarlyStop
from mlstudio.utils.data_manager import StandardScaler, data_split
from mlstudio.utils.file_manager import save_fig
def get_data():
    X, y = make_regression(n_samples=100, n_features=10, 
                           n_informative=5, bias=0.5, effective_rank=8, 
                           noise=1.0, random_state=5)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)    
    return X, y

def get_sgdregressor_results(X,y, algorithm):
    params = {'penalty': ['l1', 'l2', 'elasticnet'], 
              'alpha': [0.001, 0.01, 0.1],
              'l1_ratio': [0.5, 0.25, 0.15],
              'random_state': [50],
              'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
              'eta0': [0.1, 0.01, 0.001,]}
    estimator = SGDRegressor()
    clf = GridSearchCV(estimator, params, scoring='r2')
    clf.fit(X, y)
    # Create scores dataframe
    scores = OrderedDict()
    scores['Algorithm'] = algorithm
    scores[r'$R^2$']= [max(0,x) for x in clf.cv_results_['mean_test_score']]
    scores = pd.DataFrame.from_dict(scores) 
    # Create times dataframe
    times = OrderedDict()
    times['Algorithm'] = algorithm
    times['Fit Times (secs)']= clf.cv_results_['mean_fit_time']
    times = pd.DataFrame.from_dict(times)     
    return scores, times

def get_mlstudio_results(X, y, algorithm, batch_size):
    params = {'eta0': [0.1, 0.01, 0.001],              
              'observers': [[EarlyStop(mode='active', metric='val_score', epsilon=0.01)]],
              'objective': [Quadratic(regularizer=L1(alpha=0.001)),
                            Quadratic(regularizer=L1(alpha=0.01)),
                            Quadratic(regularizer=L1(alpha=0.1)),
                            Quadratic(regularizer=L2(alpha=0.001)),
                            Quadratic(regularizer=L2(alpha=0.01)),
                            Quadratic(regularizer=L2(alpha=0.1)),                            
                            Quadratic(regularizer=L1_L2(alpha=0.001, ratio=0.5)),
                            Quadratic(regularizer=L1_L2(alpha=0.001, ratio=0.25)),
                            Quadratic(regularizer=L1_L2(alpha=0.001, ratio=0.15)),
                            Quadratic(regularizer=L1_L2(alpha=0.01, ratio=0.5)),
                            Quadratic(regularizer=L1_L2(alpha=0.01, ratio=0.25)),
                            Quadratic(regularizer=L1_L2(alpha=0.01, ratio=0.15)),
                            Quadratic(regularizer=L1_L2(alpha=0.1, ratio=0.5)),
                            Quadratic(regularizer=L1_L2(alpha=0.1, ratio=0.25)),
                            Quadratic(regularizer=L1_L2(alpha=0.1, ratio=0.15))],
              'epochs': [1000], 'batch_size': batch_size,
              'random_state' : [50]}
    estimator = GDRegressor()
    clf = GridSearchCV(estimator, params, scoring='r2')
    clf.fit(X,y)    
    # Create scores dataframe
    scores = OrderedDict()
    scores['Algorithm'] = algorithm
    scores[r'$R^2$']= [max(0,x) for x in clf.cv_results_['mean_test_score']]
    scores = pd.DataFrame.from_dict(scores) 
    # Create times dataframe
    times = OrderedDict()
    times['Algorithm'] = algorithm
    times['Fit Times (secs)']= clf.cv_results_['mean_fit_time']
    times = pd.DataFrame.from_dict(times)     
    return scores, times        

def get_results(X, y):
    """Trains batch, stochastic and minibatch gradient descent."""    

    scores_filename = "GridsearchCV Scores.csv"
    scores_filepath = os.path.join(datadir, scores_filename)
    times_filename = "GridsearchCV Times.csv"
    times_filepath = os.path.join(datadir, times_filename)

    if os.path.exists(scores_filepath):
        scores = pd.read_csv(scores_filepath)
        times = pd.read_csv(times_filepath)

    else:
        scores, times = get_sgdregressor_results(X, y,
                                                algorithm="SKLearn SGDRegressor")        
        scores_batch, times_batch = get_mlstudio_results(X, y, 
                                                algorithm='Batch Gradient Descent', 
                                                batch_size=[None])
        scores_stochastic, times_stochastic = get_mlstudio_results(X, y, 
                                                algorithm='Stochastic Gradient Descent',
                                                batch_size=[1])
        scores_minibatch, times_minibatch = get_mlstudio_results(X, y, 
                                                algorithm='Minibatch Gradient Descent',
                                                batch_size=[32, 64, 128])    
        # Concatenate dataframes
        scores = pd.concat([scores, scores_batch, scores_stochastic, scores_minibatch], axis=0)
        times = pd.concat([times, times_batch, times_stochastic, times_minibatch], axis=0)
        # Save results as csv files
        scores.to_csv(scores_filepath)
        times.to_csv(times_filepath)
    return scores, times

X, y = get_data()
scores, times = get_results(X, y)
fig, axes = plt.subplots(1,2, figsize=(10,5))
fig.suptitle("Comparison Scikit-Learn vs ML Studio")
sns.barplot(x='Algorithm', y=r'$R^2$', data=scores, ci='sd', 
            palette="Blues_d", ax=axes[0])
axes[0].set_title("Coefficient of Determination Scores")
sns.barplot(x='Algorithm', y='Fit Times (secs)', data=times, ci='sd', 
            palette="Blues_d", ax=axes[1]) 
axes[1].set_title("Fit Times (secs)")
for ax in fig.axes:     
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')                   
plt.draw()
#%%
