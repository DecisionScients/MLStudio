#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_performance_observer.py                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 8:59:49 am                          #
# Last Modified : Monday, June 15th 2020, 8:59:49 am                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests observer."""
from collections import OrderedDict
import os
import math
import os
from pathlib import Path
import sys
testdir = str(Path(__file__).parents[2])
testdatadir = os.path.join(testdir, 'test_data')
sys.path.append(testdatadir)

import numpy as np
import pandas as pd
import pytest
from pytest import mark
from tabulate import tabulate

from mlstudio.supervised.algorithms.optimization.observers.early_stop import EarlyStop

# --------------------------------------------------------------------------  #
#                          TEST OBSERVER                                      #
# --------------------------------------------------------------------------  #
@mark.observer
@mark.performance
class PerformanceTests:
    def test_performance_validation(self): 
        # Validate metric  
        with pytest.raises(ValueError) as v:
            observer = EarlyStop(metric='hair')
            observer.on_train_begin()
        with pytest.raises(TypeError) as v:
            observer = EarlyStop(metric=1)
            observer.on_train_begin()
        # Validate epsilon
        with pytest.raises(TypeError) as v:
            observer = EarlyStop(epsilon='hair')                
            observer.on_train_begin()        
        with pytest.raises(ValueError) as v:
            observer = EarlyStop(epsilon=1.1)                
            observer.on_train_begin()
        with pytest.raises(ValueError) as v:
            observer = EarlyStop(epsilon=-0.1)                
            observer.on_train_begin()        
        # Validate patience
        with pytest.raises(TypeError) as v:
            observer = EarlyStop(patience='hair')        
            observer.on_train_begin()
        with pytest.raises(ValueError) as v:
            observer = EarlyStop(patience=0)        
            observer.on_train_begin()        

    def _get_expected_results(self, filepath):    
        return pd.read_excel(filepath, sheet_name='results', header=0, 
                                names=['epoch', 'train_cost', 'baseline', 'improvement',
                                    'sig', 'iter_no_change','factor', 'stable', 'best'], 
                                usecols="A:I",
                                dtype={"sig":bool,"stable":bool})
            

    @mark.observer
    @mark.performance
    def test_performance_reporting(self, get_mock_estimator):    
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_performance_observer_cost.xlsx")
        exp_results = self._get_expected_results(filepath)    

        # Create dict of logs dicts
        logs = exp_results.to_dict('index')
        
        # Create an estimator
        Estimator = get_mock_estimator
        est = Estimator()
        # Obtain and on_train_begin observer
        observer = EarlyStop(metric='train_cost', epsilon=0.02, patience=5)
        observer.on_train_begin()

        # Register with the observer
        observer.set_model(est)
        
        # Iterate through logs
        for epoch, log in logs.items():
            observer.on_epoch_end(epoch, log)
            
        # Gather expected results in numpy format
        exp_improvement = exp_results['sig'].to_numpy()
        exp_stability = exp_results['stable'].to_numpy()
        

        # Gather actual results and convert to numpy format
        log = observer.performance_log_
        act_improvement = log['improved']
        act_stability = log['stabilized']

        # Compare expected and actual results
        assert np.array_equal(exp_improvement, act_improvement), "Adaptive errors: \nExp {e} \nAct {a}".format(\
            e=str(exp_improvement), a=str(act_improvement))
        assert np.array_equal(exp_stability, act_stability), "Stability errors"
        
        print(tabulate(log, headers="keys"))