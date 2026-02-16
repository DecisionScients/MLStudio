# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_classification_metrics.py                                   #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 22nd 2020, 3:46:52 pm                       #
# Last Modified : Wednesday, July 22nd 2020, 3:46:52 pm                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests Classification Metrics and Metrics."""
#%%
import math
import numpy as np
import pytest
from pytest import mark
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

from mlstudio.supervised.metrics import binaryclass  

@mark.metrics
@mark.classification_metrics
class ClassificationMetricTests:

    _skl_scorers = {'ACC': accuracy_score, 'BACC': balanced_accuracy_score,
                    'F1': f1_score, 'PPV': precision_score, 'TPR': recall_score}

    def _get_expected_results(self, d, metric):
        return d['metrics'].loc[d['metrics']['metric'] == metric,'value'].to_numpy()

    def _evaluate(self, d, scorer, test, metric):
        exp_result = self._get_expected_results(d, metric)
        act_result = scorer(y=d['y'], y_pred=d['y_pred'])
        assert np.isclose(exp_result, act_result), test + 'error. Expected Result: ' + \
            str(exp_result) + ' Actual Result: ' + str(act_result)
        skl_scorer = self._skl_scorers.get(metric)
        if skl_scorer:
            exp_result = skl_scorer(y_true=d['y'], y_pred=d['y_pred'])
            assert np.isclose(exp_result, act_result), test + " not close to sklearn. " +\
                "Expected Result: " + str(exp_result) + ' Actual Result: ' + str(act_result)        

    def test_accuracy(self, get_classification_metric_test_package):
        test = 'Accuracy score '
        metric = 'ACC'
        scorer = classification.Accuracy()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)

    def test_detection_rate(self, get_classification_metric_test_package):
        test = 'Detection Rate score '
        metric = 'DR'
        scorer = classification.DetectionRate()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)

    def test_rejection_rate(self, get_classification_metric_test_package):
        test = 'Rejection Rate score '
        metric = 'CRR'
        scorer = classification.RejectionRate()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)        

    def test_positive_predictive_value(self, get_classification_metric_test_package):
        test = 'Positive Predictive Value score '
        metric = 'PPV'
        scorer = classification.PositivePredictiveValue()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)        

    def test_false_discover_rate(self, get_classification_metric_test_package):
        test = 'False Discovery Rate score '
        metric = 'FDR'
        scorer = classification.FalseDiscoveryRate()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)   

    def test_false_omission_rate(self, get_classification_metric_test_package):
        test = 'False Omission Rate score '
        metric = 'FOR'
        scorer = classification.FalseOmissionRate()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)  

    def test_negative_predictive_value(self, get_classification_metric_test_package):
        test = 'Negative Predictive Value score '
        metric = 'NPV'
        scorer = classification.NegativePredictiveValue()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                        

    def test_missclassification_rate(self, get_classification_metric_test_package):
        test = 'Missclassification Rate score '
        metric = 'MCR'
        scorer = classification.MissclassificationRate()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)       

    def test_likelihood_ratio_positive(self, get_classification_metric_test_package):
        test = 'Likelihood Ratio Positive score '
        metric = 'LRP'
        scorer = classification.PositiveLikelihoodRatio()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)        

    def test_likelihood_ratio_negative(self, get_classification_metric_test_package):
        test = 'Likelihood Ratio Negative score '
        metric = 'LRN'
        scorer = classification.NegativeLikelihoodRatio()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)        

    def test_inform(self, get_classification_metric_test_package):
        test = 'Inform score '
        metric = 'INFORM'
        scorer = classification.Informedness()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                 

    def test_balanced_accuracy(self, get_classification_metric_test_package):
        test = 'Balanced Accuracy score '
        metric = 'BACC'
        scorer = classification.BalancedAccuracy()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric) 

    def test_g_measure(self, get_classification_metric_test_package):
        test = 'G Measure score '
        metric = 'GM'
        scorer = classification.GMetric()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                                 

    def test_markedness(self, get_classification_metric_test_package):
        test = 'Markedness score '
        metric = 'MARK'
        scorer = classification.Markedness()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                                         

    def test_f1(self, get_classification_metric_test_package):
        test = 'F1 score '
        metric = 'F1'
        scorer = classification.F1()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)

    def test_f2(self, get_classification_metric_test_package):
        test = 'F2 score '
        metric = 'F2'
        scorer = classification.F2()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                                                         

    def test_f05(self, get_classification_metric_test_package):
        test = 'F0.5 score '
        metric = 'F05'
        scorer = classification.F05()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)        

    def test_cohens_kappa_chance(self, get_classification_metric_test_package):
        test = 'Cohens Kappa Chance score '
        metric = 'CKC'
        scorer = classification.CohensKappaChance()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)  

    def test_cohens_kappa(self, get_classification_metric_test_package):
        test = 'Cohens Kappa score '
        metric = 'CK'
        scorer = classification.CohensKappa()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)

    def test_matthews_correlation_coefficient(self, get_classification_metric_test_package):
        test = 'Matthews Correlation Coefficient score '
        metric = 'MCC'
        scorer = classification.MatthewsCorrelationCoefficient()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)               

    def test_odds_ratio(self, get_classification_metric_test_package):
        test = 'Odds Ratio score '
        metric = 'OR'
        scorer = classification.OddsRatio()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                        

    def test_discrimitive_power(self, get_classification_metric_test_package):
        test = 'Discrimitive Power score '
        metric = 'DP'
        scorer = classification.DiscrimitivePower()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)  

    def test_prevalence(self, get_classification_metric_test_package):
        test = 'Prevalence score '
        metric = 'PREV'
        scorer = classification.Prevalence()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                                        

    def test_skew(self, get_classification_metric_test_package):
        test = 'Skew score '
        metric = 'SKEW'
        scorer = classification.Skew()
        d = get_classification_metric_test_package        
        self._evaluate(d, scorer, test, metric)                                                                