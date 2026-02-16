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
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.algorithms.optimization.services.regularizers import L1, L2, L1_L2
# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.lasso
class LassoTests:

    def test_lasso_validation(self):
        with pytest.raises(TypeError):        
            reg = L1(alpha='hat') 
        with pytest.raises(ValueError):        
            reg = L1(alpha=0)             

    def test_lasso(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1                
        lasso = L1(alpha=alpha)
        # Lasso loss regularization        
        exp_result = regularization_package['l1_cost'] 
        act_result = lasso(theta)
        assert np.isclose(exp_result,act_result), "Lasso regularization cost error"

    def test_lasso_gradient(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1                
        lasso = L1(alpha=alpha)
        # Lasso loss regularization        
        exp_result = regularization_package['l1_grad'] 
        act_result = lasso.gradient(theta)
        assert np.allclose(exp_result,act_result), "Lasso regularization gradient error"

# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.ridge
class RidgeTests:

    def test_ridge_validation(self):
        with pytest.raises(TypeError):        
            reg = L2(alpha='hat') 
        with pytest.raises(ValueError):        
            reg = L2(alpha=0)             

    def test_ridge(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1                
        ridge = L2(alpha=alpha)
        # Lasso loss regularization        
        exp_result = regularization_package['l2_cost'] 
        act_result = ridge(theta)
        assert np.isclose(exp_result,act_result), "Ridge regularization cost error"

    def test_ridge_gradient(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1                
        ridge = L2(alpha=alpha)
        # Lasso loss regularization        
        exp_result = regularization_package['l2_grad'] 
        act_result = ridge.gradient(theta)
        assert np.allclose(exp_result,act_result), "Ridge regularization gradient error"

# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.elasticnet
class ElasticNetTests:

    def test_elasticnet_validation(self):
        with pytest.raises(TypeError):        
            reg = L1_L2(alpha='hat') 
        with pytest.raises(ValueError):        
            reg = L1_L2(alpha=0)             
        with pytest.raises(TypeError):        
            reg = L1_L2(ratio='hat') 
        with pytest.raises(ValueError):        
            reg = L1_L2(ratio=-1)
        with pytest.raises(ValueError):        
            reg = L1_L2(ratio=5)                                     

    def test_elasticnet(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1            
        ratio = 0.5    
        elasticnet = L1_L2(alpha=alpha, ratio=ratio)
        # Lasso loss regularization        
        exp_result = regularization_package['l1_l2_cost'] 
        act_result = elasticnet(theta)
        assert np.isclose(exp_result,act_result), "ElasticNet regularization cost error"

    def test_elasticnet_gradient(self, get_regularization_package):
        regularization_package = get_regularization_package
        theta = np.array([40, 6, -22, 31,26])   
        m = len(theta)
        alpha = 0.1
        ratio = 0.5                
        elasticnet = L1_L2(alpha=alpha, ratio=ratio)
        # Lasso loss regularization        
        exp_result = regularization_package['l1_l2_grad'] 
        act_result = elasticnet.gradient(theta)
        assert np.allclose(exp_result,act_result), "ElasticNet regularization gradient error"
