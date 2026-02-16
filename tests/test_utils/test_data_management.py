#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_data_management.py                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 11th 2020, 8:33:38 pm                           #
# Last Modified : Monday, May 11th 2020, 8:33:38 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests data management utilities."""
#%%
import numpy as np
import pytest
from pytest import mark
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from mlstudio.utils.data_manager import MinMaxScaler, DataSplitter, GradientScaler
from mlstudio.utils.data_manager import DataShuffler
from mlstudio.utils.data_manager import AddBiasTerm, ZeroBiasTerm, unpack_parameters
from mlstudio.utils.data_manager import LabelEncoder, OneHotLabelEncoder
# --------------------------------------------------------------------------  #
#                       TEST ADD BIAS TERM TRANSFORMER                        #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_np():
    X = np.random.rand(5,5)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X[:,0] == 1.0), "Column zero not ones."
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."
    

@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_csr():
    X = np.random.rand(5,5)
    X = csr_matrix(X)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X.toarray()[:,0] == 1.0), "Column zero not ones."   
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."    

# --------------------------------------------------------------------------  #
#                       TEST ZERO BIAS TERM TRANSFORMER                       #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.zero_bias_term
def test_zero_bias_term():
    X = np.random.rand(5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert X[0] == 0.0, "Bias not zeroed out ."
    X = np.random.rand(5,5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert np.array_equal(X[0,:], np.zeros(shape=X.shape[1])), "Bias not zeroed out ."


# --------------------------------------------------------------------------  #
#                        TEST GRADIENT SCALER                                 #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_1d
def test_gradient_scaler_1d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=20)                
        X_orig_norm = np.linalg.norm(X)        
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
        X_old_norm = np.linalg.norm(X_old)

        assert np.isclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))
        
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_2d
def test_gradient_scaler_2d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=(20,4))                
        X_orig_norm = (np.linalg.norm(X))                
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
        X_old_norm = np.linalg.norm(X_old)

        assert np.allclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))

# --------------------------------------------------------------------------  #
#                       TEST MINMAX SCALER                                    #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.minmax
def test_minmax_scaler():
    x = np.array([[0,0,22],
                [0,1,17],
                [0,1,2]], dtype=float)
    x_new = np.array([[0,0,1],
                    [0,1,15/20],
                    [0,1,0]], dtype=float)
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(x)
    assert np.array_equal(x_new, x_t), "Minmax scaler not working"    
# --------------------------------------------------------------------------  #
#                        TEST DATA SPLIT                                      #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.data_split  
def test_data_split():
    X, y = make_classification(n_classes=4, n_informative=3)
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.fit_transform(X,y, test_size=0.3,
                                                                stratify=True)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    train_values, train_counts = np.unique(y_train, return_counts=True)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    train_proportions = train_counts / n_train
    test_proportions = test_counts / n_test
    assert np.allclose(train_proportions, test_proportions, rtol=0.15), "Data split stratification problem "



# --------------------------------------------------------------------------  #
#                           TEST LABEL ENCODER                                #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.label_encoder
def test_label_encoder():
    X, y = make_classification(n_classes=4, n_informative=3)
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    assert len(np.unique(y)) == 4, "Label encoder error: Incorrect number of classes"
    assert np.unique(y) in np.arange(4), "Label encoder didn't work"
    

# --------------------------------------------------------------------------  #
#                           TEST ONE HOT LABEL ENCODER                        #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.one_hot_label_encoder
def test_one_hot_label_encoder():
    X, y = make_classification(n_classes=4, n_informative=3)
    enc = OneHotLabelEncoder()
    y = enc.fit_transform(y)
    assert len(np.unique(y)) == 2, "One Hot Label encoder error: incorrect unique values"
    assert np.unique(y) in np.arange(2), "Label encoder didn't work"
    assert y.shape[1] == 4, "OneHot Label encoder error. Shape of y is {s}. Expected 4.".format(s=str(y.shape[1]))
    


# --------------------------------------------------------------------------  #
#                        TEST UNPACK PARAMETERS                               #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.unpack_parameters
def test_unpack_parameters():
    X = np.random.rand(5)
    bias, weights = unpack_parameters(X)
    assert bias == X[0], "Bias not unpacked correctly"
    assert np.array_equal(weights, X[1:]), "weights not unpacked correctly"
    X = np.random.rand(5,3)
    bias, weights = unpack_parameters(X)
    assert bias.shape == (3,), "Bias not unpacked correctly"
    assert np.array_equal(bias, X[0,:]), "Bias not unpacked correctly "
    assert weights.shape == (4,3), "Weights not unpacked correctly"
    assert np.array_equal(weights, X[1:,:]), "Weights not unpacked correctly"


