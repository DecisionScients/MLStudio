#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_regressor.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, June 19th 2020, 4:29:58 am                          #
# Last Modified : Friday, June 19th 2020, 4:29:58 am                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Integration test for GDRegressor class."""
#%%
import numpy as np
import pytest
from pytest import mark
from sklearn.linear_model import SGDRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import SGDRegressor, SGDClassifier

from mlstudio.factories.algorithms import GradientDescent
from mlstudio.factories.observers import ObserverFactory
from mlstudio.supervised.algorithms.optimization.observers import base
from mlstudio.supervised.algorithms.optimization.observers import history
from mlstudio.supervised.algorithms.optimization.observers import report
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.algorithms.optimization.observers import learning_rate
from mlstudio.supervised.algorithms.optimization.observers import early_stop
from mlstudio.utils.data_manager import GradientScaler
from mlstudio.supervised import metrics 
# --------------------------------------------------------------------------  #

@mark.regressor
class GradientRegressorTests:

    @mark.gd_instantiation
    def test_regressor_instantiation(self):        
        est = GradientDescent().regressor(eta0=0.05, epochs=1500, batch_size=64, val_size=0.35,
                                          verbose=True, random_state=50)       
        assert est.eta0 == 0.05, "Error GradientRegressorTest: eta0 is invalid."
        assert est.epochs == 1500, "Error GradientRegressorTest: epochs is invalid."
        assert est.batch_size == 64, "Error GradientRegressorTest: batch_size is invalid."
        assert est.val_size == 0.35, "Error GradientRegressorTest: val_size is invalid."        
        assert est.verbose is True, "Error GradientRegressorTest: verbose is invalid."
        assert est.random_state == 50, "Error GradientRegressorTest: random_state is invalid."        
        assert isinstance(est.optimizer, optimizers.GradientDescentOptimizer), "Error GradientRegressorTest: optimizer is invalid."        
        assert isinstance(est.scorer, metrics.base.BaseRegressionMetric), "Error GradientRegressorTest: scorer is invalid."        
        assert isinstance(est.observer_list, base.ObserverList), "Error GradientRegressorTest: observer_list is invalid."        
        assert isinstance(est.progress, report.Progress), "Error GradientRegressorTest: progress is invalid."        
        assert isinstance(est.blackbox, history.BlackBox), "Error GradientRegressorTest: blackbox is invalid."
        assert isinstance(est.summary, report.Summary), "Error GradientRegressorTest: summary is invalid."

    @mark.gd_validation
    def test_regression_validation(self, get_regression_data):
        # Validation
        X, y = get_regression_data
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(eta0='a')
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(epochs='hat')                               
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(batch_size='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(loss='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(data_processor='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(activation='hat')                                           
            est.fit(X, y)
        with pytest.raises(ValueError):
            est = GradientDescent().regressor(theta_init=np.array([0,1]))                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(optimizer='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(scorer='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(early_stop='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(learning_rate='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(observer_list='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(progress='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(blackbox='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(summary='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(verbose='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(random_state='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(check_gradient='hat')                                           
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = GradientDescent().regressor(gradient_checker='hat')                                           
            est.fit(X, y)

    # Test learning rates
    learning_rates = \
        [ObserverFactory.time_decay(eta0=0.5, decay_factor=0.8), 
         ObserverFactory.step_decay(eta0=0.5, step_size=20), 
         ObserverFactory.sqrt_time_decay(eta0=0.1),
         ObserverFactory.exponential_decay(eta0=0.1), 
         ObserverFactory.polynomial_decay(eta0=0.1), 
         ObserverFactory.power_decay(eta0=0.1), ObserverFactory.bottou_decay(eta0=0.1), 
         ObserverFactory.adaptive_decay(eta0=0.1, monitor='val_score', epsilon=0.01,
                                        patience=50)]

    scenarios = [[learning_rate] for learning_rate in learning_rates]         
    
    lr_scenarios = [GradientDescent().regressor(learning_rate=scenario[0],
                                                loss=loss.Quadratic(gradient_scaling=True,
                                                               gradient_scaler=GradientScaler()),
                                                        epochs=1000) 
                                                        for scenario in scenarios]
    @mark.regressor_learning_rates                                                            
    def test_regressor_learning_rates(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split        
        for estimator in GradientRegressorTests.lr_scenarios:                            
            est = estimator            
            est.fit(X_train, y_train)
            skl = SGDRegressor()
            skl.fit(X_train, y_train)
            # Test data split
            m = X_train.shape[0]
            assert np.isclose(est.X_train_.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
            assert np.isclose(est.X_val_.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
            assert est.X_train_.shape[0] == len(est.y_train_), "Regressor: X_train,y_train shape mismatch "
            assert est.X_val_.shape[0] == len(est.y_val_), "Regressor: X_val,y_val shape mismatch "
            # Test blackbox
            bb = est.get_blackbox()
            assert bb.total_epochs == 1000, "Regressor: blackbox error, num epochs incorrect"
            assert bb.total_batches == 1000, "Regressor: blackbox error, num batches incorrect"
            
            assert len(bb.epoch_log.get('train_cost')) == 1000, "Regressor: blackbox error, train_cost length != num epochs"
            assert len(bb.epoch_log.get('val_cost')) == 1000, "Regressor: blackbox error, val_cost length != num epochs"
            assert len(bb.epoch_log.get('train_score')) == 1000, "Regressor: blackbox error, train_score length != num epochs"
            assert len(bb.epoch_log.get('val_score')) == 1000, "Regressor: blackbox error, val_score length != num epochs"        
            assert len(bb.epoch_log.get('theta')) == 1000, "Regressor: blackbox error, theta shape incorrect"
            assert len(bb.epoch_log.get('gradient')) == 1000, "Regressor: blackbox error, gradient shape incorrect"
            assert len(bb.epoch_log.get('gradient_norm')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('cpu_time')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('current_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('peak_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            # Evaluate results vis-a-vis sklearn
            skl_scores = skl.score(X_train, y_train)
            est_scores = est.score(X_train, y_train)
            msg = est.learning_rate.name
            d = (skl_scores-est_scores)/skl_scores
            print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
            est.summarize()
            assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-1), "Regressor: inaccurate train scores " + msg
            assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-1), "Regressor: inaccurate test scores " + msg
            # Test scorer
            est.set_scorer(metrics.regression.AdjustedR2())
            est_scores_ar2 = est.score(X_train, y_train)
            assert est_scores != est_scores_ar2, "Regressor: ar2 score not different than r2"
            assert isinstance(est_scores_ar2, float), "Regressor: ar2 score not float"
            est.set_scorer(metrics.regression.MeanAbsolutePercentageError())
            est_scores_mape =  est.score(X_train, y_train)
            assert est_scores_ar2 != est_scores_mape, "Regressor: ar2 score not different than mape"
            assert isinstance(est_scores_mape, float), "Regressor: mape score not float"

    # Test regularizers
    regs = \
        [regularizers.L1(alpha=0.1), regularizers.L2(alpha=0.02), regularizers.L1_L2(alpha=0.03, ratio=0.6)]

    scenarios = [[regularizer] for regularizer in regs]         
    
    reg_scenarios = [GradientDescent().regressor(loss=loss.Quadratic(regularizer=scenario[0]),
                                                batch_size=32) 
                                                        for scenario in scenarios]
    @mark.regressor_regularizers                                                           
    def test_regressor_regularizers(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split        
        for estimator in GradientRegressorTests.reg_scenarios:            
            est = estimator 
            est.fit(X_train, y_train)
            skl = SGDRegressor()
            skl.fit(X_train, y_train)
            # Test data split
            m = X_train.shape[0]
            assert np.isclose(est.X_train_.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
            assert np.isclose(est.X_val_.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
            assert est.X_train_.shape[0] == len(est.y_train_), "Regressor: X_train,y_train shape mismatch "
            assert est.X_val_.shape[0] == len(est.y_val_), "Regressor: X_val,y_val shape mismatch "
            # Test blackbox
            bb = est.get_blackbox()
            assert bb.total_epochs == 1000, "Regressor: blackbox error, num epochs incorrect"
            assert bb.total_batches == 8000, "Regressor: blackbox error, num batches incorrect"
            
            assert len(bb.epoch_log.get('train_cost')) == 1000, "Regressor: blackbox error, train_cost length != num epochs"
            assert len(bb.epoch_log.get('val_cost')) == 1000, "Regressor: blackbox error, val_cost length != num epochs"
            assert len(bb.epoch_log.get('train_score')) == 1000, "Regressor: blackbox error, train_score length != num epochs"
            assert len(bb.epoch_log.get('val_score')) == 1000, "Regressor: blackbox error, val_score length != num epochs"        
            assert len(bb.epoch_log.get('theta')) == 1000, "Regressor: blackbox error, theta shape incorrect"
            assert len(bb.epoch_log.get('gradient')) == 1000, "Regressor: blackbox error, gradient shape incorrect"
            assert len(bb.epoch_log.get('gradient_norm')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('cpu_time')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('current_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('peak_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            # Evaluate results vis-a-vis sklearn
            skl_scores = skl.score(X_train, y_train)
            est_scores = est.score(X_train, y_train)
            msg = est.loss.regularizer.name
            d = (skl_scores-est_scores)/skl_scores
            print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
            est.summarize()
            assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-1), "Regressor: inaccurate train scores " + msg
            assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-1), "Regressor: inaccurate test scores " + msg
            # Test scorer
            est.set_scorer(metrics.regression.AdjustedR2())
            est_scores_ar2 = est.score(X_train, y_train)
            assert est_scores != est_scores_ar2, "Regressor: ar2 score not different than r2"
            assert isinstance(est_scores_ar2, float), "Regressor: ar2 score not float"
            est.set_scorer(metrics.regression.MeanAbsolutePercentageError())
            est_scores_mape =  est.score(X_train, y_train)
            assert est_scores_ar2 != est_scores_mape, "Regressor: ar2 score not different than mape"
            assert isinstance(est_scores_mape, float), "Regressor: mape score not float"


    @mark.regressor_early_stop                                                           
    def test_regressor_early_stop(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split                   
        est = GradientDescent().regressor(early_stop=ObserverFactory().early_stop(monitor='val_score', 
                                                                                  epsilon=0.001, 
                                                                                  patience=30))
        est.fit(X_train, y_train)
        skl = SGDRegressor()
        skl.fit(X_train, y_train)
        # Test data split
        m = X_train.shape[0]
        assert np.isclose(est.X_train_.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
        assert np.isclose(est.X_val_.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
        assert est.X_train_.shape[0] == len(est.y_train_), "Regressor: X_train,y_train shape mismatch "
        assert est.X_val_.shape[0] == len(est.y_val_), "Regressor: X_val,y_val shape mismatch "
        # Evaluate results vis-a-vis sklearn
        skl_scores = skl.score(X_train, y_train)
        est_scores = est.score(X_train, y_train)
        msg = est.early_stop.name
        d = (skl_scores-est_scores)/skl_scores
        print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
        est.summarize()
        assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-1), "Regressor: inaccurate train scores " + msg
        assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-1), "Regressor: inaccurate test scores " + msg
        # Test scorer
        est.set_scorer(metrics.regression.AdjustedR2())
        est_scores_ar2 = est.score(X_train, y_train)
        assert est_scores != est_scores_ar2, "Regressor: ar2 score not different than r2"
        assert isinstance(est_scores_ar2, float), "Regressor: ar2 score not float"
        est.set_scorer(metrics.regression.MeanAbsolutePercentageError())
        est_scores_mape =  est.score(X_train, y_train)
        assert est_scores_ar2 != est_scores_mape, "Regressor: ar2 score not different than mape"
        assert isinstance(est_scores_mape, float), "Regressor: mape score not float"    