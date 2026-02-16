#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : variants.py                                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 1:52:41 pm                         #
# Last Modified : Friday, April 10th 2020, 1:52:41 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[3]
site.addsitedir(PROJECT_DIR)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from mlstudio.supervised.estimator.gradient import GDRegressor
from mlstudio.utils.data_manager import StandardScaler
from mlstudio.supervised.visual.animations.gradient import MultiModelSearch3D
from mlstudio.supervised.visual.animations.gradient import MultiModelFit2D

# Obtain and standardize data
X, y, coef = make_regression(n_samples=1000, n_features=1, effective_rank=5, 
                             noise=50, random_state=5, coef=True)
scaler = StandardScaler()
X = scaler.transform(X)

# Run Models
name = np.array(['Batch Gradient Descent', 'Minibatch Gradient Descent', 'Stochastic Gradient Descent'])
theta_init = np.array([-15., 15.])
batch_size = np.array([None, 32, 1])
models = {}
for i in range(len(batch_size)):
    models[name[i]] = GDRegressor(theta_init=theta_init, batch_size=batch_size[i]).fit(X,y)

# Run visualizations and save
directory = "../figures/variants/"
filename = "gradient_descent_search.gif"
viz = MultiModelSearch3D()
viz.search(models=models, directory=directory, filename=filename)
plt.show()
filename = "gradient_descent_fit.gif"
viz = MultiModelFit2D()
viz.fit(models=models, directory=directory, filename=filename)
plt.show()




# %%
