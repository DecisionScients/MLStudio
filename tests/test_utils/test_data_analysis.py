#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_data_analysis.py                                             #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 25th 2020, 4:37:35 am                           #
# Last Modified : Monday, May 25th 2020, 4:37:36 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests for data analysis functions."""
import numpy as np
import pytest
from pytest import mark
import string

from scipy import stats

from mlstudio.utils.data_analyzer import describe_categorical_array
from mlstudio.utils.data_analyzer import describe_numeric_array
# --------------------------------------------------------------------------  #
#                             TEST DESCRIBE                                   #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_analysis
@mark.describe
def test_describe_numeric_array():
    a = np.random.randint(low=1, high=100, size=100)
    d = describe_numeric_array(a)
    columns = ['min', 'max', 'mean', 'std', 'skew', 'kurtosis', '25th percentile',
               '50th percentile', '75th percentile', 'count']
    assert len(d) == 10, "Describe numeric array dictionary wrong length."
    for c in columns:        
        assert d.get(c), "Describe numeric didn't return " + c

@mark.utils
@mark.data_analysis
@mark.describe
def test_describe_categorical_array():
    s = string.ascii_lowercase
    s = list(s)
    a = np.random.choice(s, size=100)
    d = describe_categorical_array(a)
    columns = ['count', 'top', 'unique', 'freq']
    assert len(d) == 4, "Describe categorical array dictionary wrong length."
    for c in columns:        
        assert d.get(c) is not None, "Describe categorical didn't return " + c




