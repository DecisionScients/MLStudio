#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_py                                                #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, May 24th 2020, 5:15:40 am                           #
# Last Modified : Sunday, May 24th 2020, 5:15:40 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests validation utilities."""
import sys
import numpy as np
import pytest
from pytest import mark
import scipy.sparse as sp

from mlstudio.utils.validation import is_one_hot
from mlstudio.utils.validation import is_multilabel, is_valid_array_size
# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
@mark.data_checks
@mark.is_one_hot
def test_is_one_hot(get_data_management_data):
    d = get_data_management_data
    for k, y in d.items():        
        msg = "Is one-hot of " + k + " didn't work."
        if k == 'one_hot':
            assert is_one_hot(y), msg
        else:
            assert not is_one_hot(y), msg

# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
@mark.data_checks
@mark.is_multilabel
def test_is_multilabel(get_data_management_data):
    d = get_data_management_data
    for k, y in d.items():        
        msg = "Is multilabel of " + k + " didn't work."
        if 'multilabel' in k:
            assert is_multilabel(y), msg
        else:
            assert not is_multilabel(y), msg

# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
@mark.data_checks
def test_is_valid_array_size():
    from mlstudio.utils.validation import is_valid_array_size
    X = np.random.default_rng().uniform(low=100, high=200, size=20)       
    Y = np.random.default_rng().uniform(low=1, high=5, size=20)       
    assert not is_valid_array_size(X, lower=1, upper=2), "is_valid_array_size failed"
    assert is_valid_array_size(Y, lower=1, upper=100), "is_valid_array_size failed"
  
# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
def test_validate_zero_to_one():
    from mlstudio.utils.validation import validate_zero_to_one
    with pytest.raises(ValueError) as v:
        validate_zero_to_one(5, 'test_param')
        assert "assertion error"  in str(v.value)
    with pytest.raises(ValueError) as v:
        validate_zero_to_one(0, left='open')
        assert "assertion error"  in str(v.value)    
    with pytest.raises(ValueError) as v:
        validate_zero_to_one(1, right='open')
        assert "assertion error"  in str(v.value)    
    validate_zero_to_one(0, left="closed")


# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
def test_validate_range():
    from mlstudio.utils.validation import validate_range
    with pytest.raises(ValueError) as v:
        validate_range(param=1, minimum=0, maximum=1, param_name='test_param')
        assert "value error"  in str(v.value)
    with pytest.raises(ValueError) as v:
        validate_range(param=0, minimum=0, maximum=1, param_name='test_param')
        assert "value error"  in str(v.value)        
    validate_range(param=1, minimum=0, maximum=1, right='closed')
    validate_range(param=0, minimum=0, maximum=1, left='closed')

# --------------------------------------------------------------------------  #    

@mark.utils
@mark.validation
def test_validate_string():
    from mlstudio.utils.validation import validate_string
    valid_values = ['epoch', 'batch']
    with pytest.raises(ValueError) as v:
        validate_string('hand', valid_values=valid_values)
        assert "assertion error"  in str(v.value)
    validate_string('batch', valid_values=valid_values)

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_activation():    
    from mlstudio.utils.validation import validate_activation
    from mlstudio.supervised.algorithms.optimization.services.activations import Sigmoid
    with pytest.raises(TypeError) as v:
        validate_activation('hand')
        assert "value error"  in str(v.value)
    validate_activation(Sigmoid())

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_objective():    
    from mlstudio.utils.validation import validate_objective
    from mlstudio.supervised.algorithms.optimization.services.loss import Quadratic
    with pytest.raises(TypeError) as v:
        validate_objective('hand')
        assert "value error"  in str(v.value)
    validate_objective(Quadratic())    

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_optimizer():    
    from mlstudio.utils.validation import validate_optimizer
    from mlstudio.supervised.algorithms.optimization.services.optimizers import Adam 
    with pytest.raises(TypeError) as v:
        validate_optimizer('hand')
        assert "value error"  in str(v.value)
    validate_optimizer(Adam())     

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_regularizer():    
    from mlstudio.utils.validation import validate_regularizer
    from mlstudio.supervised.algorithms.optimization.services.regularizers import L1
    with pytest.raises(TypeError) as v:
        validate_regularizer('hand')
        assert "value error"  in str(v.value)
    validate_regularizer(L1())       

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_monitor():    
    from mlstudio.utils.validation import validate_monitor
    from mlstudio.supervised.metrics.regression import MSE    
    with pytest.raises(TypeError) as v:
        validate_monitor('hand')
        assert "value error"  in str(v.value)
    validate_monitor(MSE())           

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_application():    
    from mlstudio.utils.validation import validate_application
    from mlstudio.supervised.algorithms.optimization.services.tasks import BinaryClassification    
    with pytest.raises(TypeError) as v:
        validate_application('hand')
        assert "value error"  in str(v.value)
    validate_application(BinaryClassification())           

