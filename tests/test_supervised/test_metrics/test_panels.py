# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_panels.py                                                   #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 29th 2020, 4:16:12 am                       #
# Last Modified : Wednesday, July 29th 2020, 4:16:12 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests panels, panel factory and repo."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories.algorithms import GDRegressor, GDBinaryClassifier
from mlstudio.factories.algorithms import GDMultiClassifier
from mlstudio.supervised.metrics.base import *
from mlstudio.supervised.metrics import regression, binaryclass, multiclass
from mlstudio.supervised.algorithms.optimization.observers.early_stop import EarlyStop
from mlstudio.supervised.metrics.panel import PanelRepo
from mlstudio.supervised.metrics.panel import RegressionPanel
from mlstudio.supervised.metrics.panel import BinaryClassPanel
from mlstudio.supervised.metrics.panel import MultiClassPanel
from mlstudio.factories.panels import PanelFactories
# --------------------------------------------------------------------------- #
@mark.panel
@mark.regression_panel
class RegressionPanelTests:

    def test_regression_panel_properties(self):
        panel = RegressionPanel(name="regression_panel_01",
                                description='First Regression Panel')        
        assert panel.name == 'regression_panel_01', "Regression Panel: panel name error"
        assert panel.label == "Regression Panel 01", "Regression Panel: panel label error"
        assert panel.description == "First Regression Panel", "Regression Panel: panel description error"
        assert panel.panel_type == "Regression", "Regression Panel: panel type error"
        panel.description = "First Regression Panel Today"
        assert panel.description == "First Regression Panel Today", "Regression Panel: panel description error"

    def test_regression_panel_add_get_scorer(self):
        panel = RegressionPanel(name="regression_panel_02",
                                description='Second Regression Panel')
        panel.add_scorer(scorer=regression.MeanSquaredError())
        assert isinstance(panel.get_scorer(key='MSE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_squared_error'), BasePerformance), "Regression Panel: add scorer error."
        with pytest.raises(KeyError):
            panel.add_scorer(scorer=regression.MeanSquaredError())
        panel.add_scorer(scorer=regression.MeanAbsolutePercentageError())
        assert isinstance(panel.get_scorer(key='MAPE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_absolute_percentage_error'), BasePerformance), "Regression Panel: add scorer error."


    def test_regression_panel_del_scorer(self):
        panel = RegressionPanel(name="regression_panel_03",
                                description='Third Regression Panel')
        panel.add_scorer(scorer=regression.MeanSquaredError())
        panel.add_scorer(scorer=regression.MeanAbsolutePercentageError())
        panel.del_scorer(key="MAPE")
        with pytest.raises(KeyError):
            panel.get_scorer(key="MAPE")
        with pytest.raises(KeyError):
            panel.get_scorer(key="mean_absolute_percentage_error")            
        assert isinstance(panel.get_scorer(key='MSE'), BasePerformance), "Regression Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='mean_squared_error'), BasePerformance), "Regression Panel: add scorer error."        

    def test_regression_panel_print_scorers(self):
        panel = RegressionPanel(name="regression_panel_04",
                                description='Forth Regression Panel')
        panel.add_scorer(scorer=regression.MeanSquaredError())
        panel.add_scorer(scorer=regression.MeanAbsolutePercentageError())        
        panel.print_scorers()

    def test_regression_panel_call(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split
        est = GradientDescent.regression(epochs=5000)
        est.fit(X_train, y_train)
        panel = RegressionPanel(name="regression_panel_05",
                                description='Fifth Regression Panel') 
        panel.add_scorer(scorer=regression.MeanSquaredError())
        panel.add_scorer(scorer=regression.MeanAbsolutePercentageError())        
        panel.add_scorer(scorer=regression.RootMeanSquaredError())                                               
        panel.add_scorer(scorer=regression.NegativeMeanSquaredError())                                               
        panel.add_scorer(scorer=regression.NegativeRootMeanSquaredError())                                               
        df = panel(estimator=est, X=X_test, y=y_test, reference='regression_test_01')
        assert isinstance(df, pd.DataFrame), "Regression Panel: call error"
        print(df)
        panel.del_scorer(key='MSE')
        panel.del_scorer(key='MAPE')        
        panel.del_scorer(key='RMSE')                                               
        panel.del_scorer(key='NMSE')                                               
        panel.del_scorer(key='NRMSE')   
        panel.add_scorer(scorer=regression.R2())                                               
        panel.add_scorer(scorer=regression.AdjustedR2())                                               
        panel.add_scorer(scorer=regression.PercentVarianceExplained())     
        panel.add_scorer(scorer=regression.MeanAbsoluteError())     
        panel.add_scorer(scorer=regression.MeanSquaredLogError())     
        panel.add_scorer(scorer=regression.MedianAbsoluteError())           
        df2 = panel(estimator=est, X=X_test, y=y_test, reference='regression_test_02')        
        assert df.shape != df2.shape, "Regression panel: all scores equal current scores."
        print(df2)        

@mark.panel
@mark.binaryclass_panel
class BinaryClassPanelTests:

    def test_binaryclass_panel_properties(self):
        panel = BinaryClassPanel(name="binaryclass_panel_01",
                                description='First BinaryClass Panel')
        assert panel.name == 'binaryclass_panel_01', "BinaryClass Panel: panel name error"
        assert panel.label == "Binaryclass Panel 01", "BinaryClass Panel: panel label error"
        assert panel.description == "First BinaryClass Panel", "BinaryClass Panel: panel description error"
        assert panel.panel_type == "Binary Classification", "BinaryClass Panel: panel type error"
        panel.description = "First BinaryClass Panel Today"
        assert panel.description == "First BinaryClass Panel Today", "BinaryClass Panel: panel description error"

    def test_binaryclass_panel_add_get_scorer(self):
        panel = BinaryClassPanel(name="binaryclass_panel_02",
                                description='Second BinaryClass Panel')
        panel.add_scorer(scorer=binaryclass.Accuracy())
        panel.add_scorer(scorer=binaryclass.F1())
        panel.add_scorer(scorer=binaryclass.F2())
        assert isinstance(panel.get_scorer(key='Accuracy'), BasePerformance), "BinaryClass Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='f1_score'), BasePerformance), "BinaryClass Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='f2_score'), BasePerformance), "BinaryClass Panel: add scorer error."
        with pytest.raises(KeyError):
            panel.add_scorer(scorer=binaryclass.Accuracy())
        panel.add_scorer(scorer=binaryclass.FalseNegativeRate())
        assert isinstance(panel.get_scorer(key='FNR'), BasePerformance), "BinaryClass Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='false_negative_rate'), BasePerformance), "BinaryClass Panel: add scorer error."


    def test_binaryclass_panel_del_scorer(self):
        panel = BinaryClassPanel(name="binaryclass_panel_03",
                                description='Third BinaryClass Panel')
        panel.add_scorer(scorer=binaryclass.F2())
        panel.add_scorer(scorer=binaryclass.F1())
        panel.del_scorer(key="F1")
        with pytest.raises(KeyError):
            panel.get_scorer(key="F1")
        with pytest.raises(KeyError):
            panel.get_scorer(key="f1_score")            
        assert isinstance(panel.get_scorer(key='F2'), BasePerformance), "BinaryClass Panel: add scorer error."
        assert isinstance(panel.get_scorer(key='f2_score'), BasePerformance), "BinaryClass Panel: add scorer error."        

    def test_binaryclass_panel_print_scorers(self):
        panel = BinaryClassPanel(name="binaryclass_panel_04",
                                description='Forth BinaryClass Panel')
        panel.add_scorer(scorer=binaryclass.F1())
        panel.add_scorer(scorer=binaryclass.F2())        
        panel.print_scorers()

    def test_binaryclass_panel_call(self, get_logistic_regression_data_split):
        X_train, X_test, y_train, y_test = get_logistic_regression_data_split
        est = GradientDescent.binaryclass(epochs=5000)
        est.fit(X_train, y_train)
        panel = BinaryClassPanel(name="binaryclass_panel_05",
                                description='Fifth BinaryClass Panel') 
        panel.add_scorer(scorer=binaryclass.F1())
        panel.add_scorer(scorer=binaryclass.F2())        
        panel.add_scorer(scorer=binaryclass.FalseNegativeRate())                                               
        panel.add_scorer(scorer=binaryclass.FalsePositiveRate())                                               
        panel.add_scorer(scorer=binaryclass.FalseNegative())                                               
        df = panel(estimator=est, X=X_test, y=y_test, reference='binaryclass_test_01')
        assert isinstance(df, pd.DataFrame), "BinaryClass Panel: call error"
        print(df)
        panel.del_scorer(key='F1')
        panel.del_scorer(key='F2')        
        panel.del_scorer(key='FNR')                                               
        panel.del_scorer(key='FPR')                                               
        panel.del_scorer(key='FN')   
        panel.add_scorer(scorer=binaryclass.TrueNegative())                                               
        panel.add_scorer(scorer=binaryclass.TruePositiveRate())                                               
        panel.add_scorer(scorer=binaryclass.PositiveLikelihoodRatio())     
        panel.add_scorer(scorer=binaryclass.PositivePredictiveValue())     
        panel.add_scorer(scorer=binaryclass.Precision())     
        panel.add_scorer(scorer=binaryclass.Recall())           
        df2 = panel(estimator=est, X=X_test, y=y_test, reference='binaryclass_test_02')        
        assert df.shape != df2.shape, "BinaryClass panel: all scores equal current scores."
        print(df2)        

#TODO: Test multiclass once metrics are done. 

