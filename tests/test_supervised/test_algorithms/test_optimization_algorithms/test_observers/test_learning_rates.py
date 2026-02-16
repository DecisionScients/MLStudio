#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_learning_rates.py                                            #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, May 17th 2020, 2:27:44 am                           #
# Last Modified : Sunday, May 17th 2020, 2:34:01 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests learning rate classes."""
#%%
import math
import os
from pathlib import Path
import sys
testdir = str(Path(__file__).parents[2])
testdatadir = os.path.join(testdir, 'test_data')
sys.path.append(testdatadir)

import pandas as pd
import pytest
from pytest import mark
import numpy as np

from mlstudio.supervised.algorithms.optimization.observers.learning_rate import StepDecay, TimeDecay
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import SqrtTimeDecay, ExponentialSchedule
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import PowerSchedule
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import BottouSchedule, Adaptive

@mark.observer
@mark.lrs
@mark.step_decay
class StepDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_step_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = StepDecay(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = StepDecay(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(eta_min=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_factor=2)
            lrs.on_train_begin()       
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_steps=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_steps=-1)
            lrs.on_train_begin()
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_steps=np.inf)
            lrs.on_train_begin()      

    def test_step_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_step_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = StepDecay(eta0=0.1, eta_min=0.01,
                        decay_factor=0.5, decay_steps=5)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg

@mark.observer
@mark.lrs
@mark.time_decay
class TimeDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_time_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = TimeDecay(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(eta0=0, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(eta0=1, decay_factor=0.5)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = TimeDecay(eta_min=None, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(eta_min=0, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(eta_min=1, decay_factor=0.5)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = TimeDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(decay_factor=2)
            lrs.on_train_begin()       
    

    def test_time_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_time_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = TimeDecay(eta0=0.1, eta_min=0.02,
                        decay_factor=0.5)
        observers = [lrs]              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg

@mark.observer
@mark.lrs
@mark.sqrt_time_decay
class SqrtTimeDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_sqrt_time_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(eta_min=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(decay_factor=2)
            lrs.on_train_begin()       
    

    def test_sqrt_time_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_sqrt_time_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = SqrtTimeDecay(eta0=0.1, eta_min=0.045,
                        decay_factor=0.5)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg    

@mark.observer
@mark.lrs
@mark.exponential_decay
class ExponentialDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_exponential_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(eta_min=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(decay_factor=2)
            lrs.on_train_begin()       
    
    def test_exponential_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_exponential_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = ExponentialDecay(eta0=0.1, eta_min=0.05,
                        decay_factor=0.1)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg                

@mark.observer
@mark.lrs
@mark.polynomial_decay
class PolynomialDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_polynomial_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(eta_min=1)
            lrs.on_train_begin()      
        # Validate power
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(power=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(power=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(power=2)
            lrs.on_train_begin()       
    
    def test_polynomial_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_polynomial_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = PolynomialDecay(eta0=0.1, eta_min=0.025,
                        power=0.95)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg           

@mark.observer
@mark.lrs
@mark.power_decay
class PowerScheduleTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_power_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = PowerSchedule(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = PowerSchedule(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(eta_min=1)
            lrs.on_train_begin()      
        # Validate power
        with pytest.raises(TypeError):
            lrs = PowerSchedule(power=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(power=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(power=2)
            lrs.on_train_begin()       
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = PowerSchedule(decay_steps=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(decay_steps=0)
            lrs.on_train_begin()                
    
    def test_power_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_power.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = PowerSchedule(eta0=0.1, eta_min=0.04,
                        power=0.95, decay_steps=5)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg       

@mark.observer
@mark.lrs
@mark.bottou_decay
class BottouScheduleTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_bottou_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = BottouSchedule(eta0='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(eta0=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(eta0=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = BottouSchedule(eta_min=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(eta_min=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(eta_min=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = BottouSchedule(decay_factor=None)            
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(decay_factor=2)
            lrs.on_train_begin()       
            
    
    def test_bottou_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_bottou.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = BottouSchedule(eta0=0.1, eta_min=0.07,
                        decay_factor=0.5)
        observers=[lrs]
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, eta0=0.1, observers=observers)
        est.fit()
        # Extract learning rate monitoring
        epochs = est.get_blackbox().epoch_log.get('epoch')
        act_results = est.get_blackbox().epoch_log.get('eta')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg                

@mark.observer
@mark.lrs
@mark.improvement
class AdaptiveTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['cost','lr'],
                                    usecols="C,H")

    def test_improvement_decay_validation(self, get_mock_estimator):
        est = get_mock_estimator        
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = Adaptive(eta0='h')
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(eta0=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(eta0=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = Adaptive(eta_min=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(eta_min=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(eta_min=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = Adaptive(decay_factor=None)            
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(decay_factor=-1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Adaptive(decay_factor=2)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()   
    
    def test_improvement_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_improvement.xlsx")
        exp_results = self._get_expected_results(filepath)
        cost = exp_results['cost'].values
        exp_results = exp_results['lr'].values
        # Instantiate learning rate schedule and create it as an observer
        lrs = Adaptive(eta0=0.1, eta_min=0.007,
                        decay_factor=0.5, epsilon=0.01, patience=2)
        observers=[lrs]    
        # Instantiate mock estimator
        est = Estimator(epochs=20, eta0=0.1, observers=observers)       
        lrs.set_model(est)
        # Simulate training 
        lrs.on_train_begin()
        act_results = []
        lr = 0.1
        for i in range(20):            
            log = {'train_cost':cost[i], 'eta': lrs.model.eta}    
            lrs.on_epoch_end(epoch=i, log=log)                
            act_results.append(lrs.model.eta)
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg         