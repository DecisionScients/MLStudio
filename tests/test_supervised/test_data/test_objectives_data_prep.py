#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_objectives.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 3:45:31 pm                          #
# Last Modified : Monday, June 15th 2020, 3:45:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
import math
import os
from pathlib import Path
import sys

import glob
import numpy as np
import pandas as pd
import pytest
from pytest import mark
from scipy.special import softmax
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression, make_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer

homedir = str(Path(__file__).parents[2])
datadir = os.path.join(homedir, "tests\\test_data")
sys.path.append(homedir)
sys.path.append(datadir)

from mlstudio.utils.data_manager import StandardScaler
from mlstudio.supervised.algorithms.optimization.services.loss import Quadratic, CrossEntropy
from mlstudio.supervised.algorithms.optimization.services.loss import CategoricalCrossEntropy
from mlstudio.supervised.algorithms.optimization.services.regularizers import L1, L2, L1_L2

# --------------------------------------------------------------------------  #
def create_regression_prediction(y=None, X=None, theta=None):
    """Creates vector of predictions based upon target plus random noise."""
    noise = np.random.normal(0,1, size=y.shape)    
    y_pred = np.add(y,noise)
    return y_pred

def create_classification_prediction(y=None, X=None, theta=None):
    """Creates classification prediction as probability [0,1]"""
    return np.random.uniform(0,1, size=y.shape)

def create_multiclass_prediction(y=None, X=None, theta=None):
    """Creates multiclassification prediction."""
    z = X.dot(theta)
    return softmax(z, axis=1)

def make_regression_data():
    X, y = make_regression(n_samples=100, n_features=5, random_state=5)    
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    return X, y

def make_classification_data():
    X, y, = make_classification(n_samples=100, n_features=5, random_state=5)        
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    return X, y

def make_multiclass_data():
    X, y, = make_classification(n_samples=100, n_features=5, n_classes=4, 
                                n_informative=3, random_state=5)     
    enc = LabelBinarizer()
    y = enc.fit_transform(y) 
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    print(X.shape)
    print(y.shape)
    return X, y            

def create_data():
    # Designate filenames and create filepaths
    mse_filename = "test_objective_cost_functions_mse.xlsx"
    xe_filename = "test_objective_cost_functions_xe.xlsx"
    cxe_filename = "test_objective_cost_functions_cxe.xlsx"    
    mse_filepath = os.path.join(datadir, mse_filename)
    xe_filepath = os.path.join(datadir, xe_filename)
    cxe_filepath = os.path.join(datadir, cxe_filename)
    # Obtain data
    X_reg, y_reg = make_regression_data()
    X_bin, y_bin = make_classification_data()
    X_multi, y_multi = make_multiclass_data()
    # Create parameters
    regression_theta = np.random.default_rng().uniform(low=0, high=1, size=X_reg.shape[1])            
    classification_theta = np.random.default_rng().uniform(low=0, high=1, size=X_bin.shape[1])            
    multiclass_theta = np.random.default_rng().uniform(low=0, high=1, size=(X_multi.shape[1],y_multi.shape[1]))            
    # Create packages 
    regression_pack = {'locked': True, 'filepath': mse_filepath, 'X':X_reg, 
                       'y': y_reg, 'theta': regression_theta, 
                       'predict': create_regression_prediction}
    classification_pack = {'locked': True, 'filepath': xe_filepath, 
                           'X':X_bin, 'y': y_bin, 'theta': classification_theta,
                           'predict': create_classification_prediction}
    multiclass_pack = {'locked': True, 'filepath': cxe_filepath, 'X':X_multi, 
                       'y': y_multi, 'theta': multiclass_theta,
                       'predict': create_multiclass_prediction}
    data_packs = [regression_pack, classification_pack, multiclass_pack]
    # Write to files
    for data in data_packs:        
        if data['locked']:
            pass
        else:
            X = pd.DataFrame(data=data['X'])
            y = pd.DataFrame(data=data['y'])        
            theta = pd.DataFrame(data=data['theta'])
            y_pred = pd.DataFrame(data=data['predict'](y, X, theta))
            with pd.ExcelWriter(data['filepath']) as writer:
                X.to_excel(writer, sheet_name='X')
                y.to_excel(writer, sheet_name='y')
                y_pred.to_excel(writer, sheet_name='y_pred')
                theta.to_excel(writer, sheet_name='theta')

#create_data()
#%%