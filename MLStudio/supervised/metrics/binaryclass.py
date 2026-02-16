# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \classification.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:26:07 am                        #
# Last Modified : Thursday, July 16th 2020, 2:26:07 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Scorers for binary classification metrics."""
import math
import numpy as np
from mlstudio.supervised.metrics.base import BaseBinaryClassificationMetric
from mlstudio.supervised.metrics.base import BaseBinaryClassificationMetric
# --------------------------------------------------------------------------- #
#                        CLASSIFICATION MEASURES                              #
# --------------------------------------------------------------------------- #
# -------------------------- BASE MEASURES ---------------------------------- #

class TruePositive(BaseBinaryClassificationMetric):
    """Computes the number true positives."""    
    _code = "TP"        
    _name = 'true_positive'
    _label = "True Positive (Power)"

    def __call__(self, y, y_pred,  *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == 1) & (df['y_pred'] == 1)]
        return len(result.index)    


class TrueNegative(BaseBinaryClassificationMetric):
    """Computes the number true negatives."""
    _code = "TN"        
    _name = 'true_negative'
    _label = "True Negative"

    def __call__(self, y, y_pred,  *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == 0) & (df['y_pred'] == 0)]
        return len(result.index)    

class FalsePositive(BaseBinaryClassificationMetric):
    """Computes the number false positives."""
    _code = "FP"        
    _name = 'false_positive'
    _label = "False Positive (Type I Error)"

    def __call__(self, y, y_pred,  *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == 0) & (df['y_pred'] == 1)]
        return len(result.index)    

class FalseNegative(BaseBinaryClassificationMetric):
    """Computes the number false negatives."""
    _code = "FN"        
    _name = 'false_negative'
    _label = "False Negative (Type II Error)"

    def __call__(self, y, y_pred,  *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == 1) & (df['y_pred'] == 0)]
        return len(result.index)            
# -------------------------- 1st LEVEL MEASURES ----------------------------- #

class PositiveCondition(BaseBinaryClassificationMetric):
    """Computes positive condition as true positives + false negatives."""
    _code = "P"        
    _name = 'positive_condition'
    _label = "Positive Condition"

    def __call__(self, y, y_pred,  *args, **kwargs):
        tp = TruePositive()(y=y, y_pred=y_pred)
        fn = FalseNegative()(y=y, y_pred=y_pred)                              
        return tp + fn

class NegativeCondition(BaseBinaryClassificationMetric):
    """Computes negative condition as false positives + true negatives."""    
    _code = "N"        
    _name = 'negative_condition'
    _label = "Negative Condition"

    def __call__(self, y, y_pred,  *args, **kwargs):
        fp = FalsePositive()(y=y, y_pred=y_pred)
        tn = TrueNegative()(y=y, y_pred=y_pred)                              
        return fp + tn        
# --------------------------------------------------------------------------- #

class OutcomePositive(BaseBinaryClassificationMetric):
    """Computes outcome positive as true positives plus false positives."""
    _code = "OP"    
    _name = 'outcome_positive'    
    _label = "Outcome Positive"

    def __call__(self, y, y_pred,  *args, **kwargs):
        tp = TruePositive()(y=y, y_pred=y_pred)
        fp = FalsePositive()(y=y, y_pred=y_pred)                              
        return tp + fp

class OutcomeNegative(BaseBinaryClassificationMetric):
    """Computes outcome positive as false negatives plus true negatives."""    
    _code = "ON"        
    _name = 'outcome_negative'
    _label = "Outcome Negative"

    def __call__(self, y, y_pred,  *args, **kwargs):
        fn = FalseNegative()(y=y, y_pred=y_pred)
        tn = TrueNegative()(y=y, y_pred=y_pred)                              
        return fn + tn        
# --------------------------------------------------------------------------- #

class TrueClassification(BaseBinaryClassificationMetric):
    """True classification is the sum of true positives and true negatives.""" 
    _code = "TC"        
    _name = 'true_classification'
    _label = "True Classification"

    def __call__(self, y, y_pred,  *args, **kwargs):
        tp = TruePositive()(y=y, y_pred=y_pred)
        tn = TrueNegative()(y=y, y_pred=y_pred)                              
        return tp + tn        

class FalseClassification(BaseBinaryClassificationMetric):
    """False classification is the sum of false positives and false negatives."""
    _code = "FC"        
    _name = 'false_classification'
    _label = "False Classification"

    def __call__(self, y, y_pred,  *args, **kwargs):
        fp = FalsePositive()(y=y, y_pred=y_pred)
        fn = FalseNegative()(y=y, y_pred=y_pred)                              
        return fp + fn        
# ------------------------ 2ND LEVEL MEASURES ------------------------------- #

class PositiveLikelihoodRatio(BaseBinaryClassificationMetric):
    """Positive likelihood ratio is true positive rate / false positive rate."""
    _code = 'LRP'
    _name = 'positive_likelihood_ratio'
    _label = "Positive Likelihood Ratio"

    def __call__(self, y, y_pred,  *args, **kwargs):
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)
        fpr = FalsePositiveRate()(y, y_pred, *args, **kwargs)
        return tpr / fpr    

class NegativeLikelihoodRatio(BaseBinaryClassificationMetric):
    """Negative likelihood ratio is false negative rate / true negative rate."""
    _code = 'LRP'
    _name = 'negative_likelihood_ratio'
    _label = "Negative Likelihood Ratio"

    def __call__(self, y, y_pred,  *args, **kwargs):
        fnr = FalseNegativeRate()(y, y_pred, *args, **kwargs)
        tnr = TrueNegativeRate()(y, y_pred, *args, **kwargs)
        return fnr / tnr
# --------------------------------------------------------------------------- #

class Bias(BaseBinaryClassificationMetric):
    """Computes bias as the ratio of outcome positive and the sample size."""
    _code = 'BIAS'
    _name = 'bias'
    _label = "Bias"

    def __call__(self, y, y_pred,  *args, **kwargs):
        op = OutcomePositive()(y, y_pred, *args, **kwargs)        
        return op / y.shape[0]
# --------------------------------------------------------------------------- #

class Prevalence(BaseBinaryClassificationMetric):
    """Computes prevalence as positive condition / sample size."""
    _code = 'PREV'
    _name = 'prevalence'
    _label = "Prevalence"

    def __call__(self, y, y_pred,  *args, **kwargs):
        p = PositiveCondition()(y, y_pred, *args, **kwargs)        
        return p / y.shape[0]

class Skew(BaseBinaryClassificationMetric):
    """Computes skew as ratio of negative and positive condition."""
    _code = 'SKEW'
    _name = 'skew'
    _label = "Skew"

    def __call__(self, y, y_pred,  *args, **kwargs):
        n = NegativeCondition()(y, y_pred, *args, **kwargs)        
        p = PositiveCondition()(y, y_pred, *args, **kwargs)        
        return n / p
# --------------------------------------------------------------------------- #

class CohensKappaChance(BaseBinaryClassificationMetric):
    """Computes Cohens Kappa Chance as (P*OP + N*ON) / sample size squared."""
    _code = 'CKc'
    _name = 'cohens_kappa_chance'
    _label = "Cohen's Kappa Chance"

    def __call__(self, y, y_pred,  *args, **kwargs):
        n = NegativeCondition()(y, y_pred, *args, **kwargs)        
        p = PositiveCondition()(y, y_pred, *args, **kwargs)        
        op = OutcomePositive()(y, y_pred, *args, **kwargs)        
        on = OutcomeNegative()(y, y_pred, *args, **kwargs)                
        return ((p * op) + (n * on)) / y.shape[0]**2
# ------------------------ 3RD LEVEL MEASURES ------------------------------- #

class OddsRatio(BaseBinaryClassificationMetric):
    """Computes odds ratio as (tp-tn) / (fp-fn)."""
    _code = 'OR'
    _name = 'odds_ratio'
    _label = "Odds Ratio"

    def __call__(self, y, y_pred,  *args, **kwargs):
        tp = TruePositive()(y, y_pred, *args, **kwargs)
        tn = TrueNegative()(y, y_pred, *args, **kwargs)        
        fp = FalsePositive()(y, y_pred, *args, **kwargs)        
        fn = FalseNegative()(y, y_pred, *args, **kwargs)        
        return (tp-tn) / (fp-fn)        

class DiscrimitivePower(BaseBinaryClassificationMetric):
    """Computes descrimitive power as:
    .. math:: \frac{\sqrt{3}}{\pi}\text{log(OR)}
    """
    _code = 'DP'
    _name = 'discrimitive_power'
    _label = "Discrimitive Power"

    def __call__(self, y, y_pred,  *args, **kwargs):
        return (np.sqrt(3) / np.pi) * \
            np.log(OddsRatio()(y, y_pred, *args, **kwargs))        
# --------------------------------------------------------------------------- #
#                        CLASSIFICATION METRICS                               #
# --------------------------------------------------------------------------- #
# --------------------------- BASE METRICS ---------------------------------- #

class Accuracy(BaseBinaryClassificationMetric):
    """Computes accuracy ratio of true classification and sample size."""
    _code = "ACC"
    _name = 'accuracy'
    _label = "Accuracy"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):                
        tc = TrueClassification()(y, y_pred, *args, **kwargs)   
        return tc / y.shape[0]

class DetectionRate(BaseBinaryClassificationMetric):
    """Computes detection rate as tp/sn."""
    _code = "DR"
    _name = 'detection_rate'
    _label = "Detection Rate"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):   
        tp = TruePositive()(y, y_pred, *args, **kwargs)
        return tp / y.shape[0]        

class RejectionRate(BaseBinaryClassificationMetric):
    """Computes rejection rate as tn/sn."""
    _code = "CRR"
    _name = 'rejection_rate'
    _label = "Rejection Rate (Corrected)"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tn = TrueNegative()(y, y_pred, *args, **kwargs)
        return tn / y.shape[0]
# --------------------------------------------------------------------------- #

class PositivePredictiveValue(BaseBinaryClassificationMetric):
    """Positive predictive value as ratio of true positives and outcome positive."""       
    _code = "PPV"
    _name = 'precision'
    _label = "Positive Predictive Value (Precision)"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)
        op = OutcomePositive()(y, y_pred, *args, **kwargs)
        return tp / op

class Precision(PositivePredictiveValue):
    """Alias class for PositivePredictiveValue."""
    _code = "PREC"
    _name = 'precision'
    _label = "Precision"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False    

class NegativePredictiveValue(BaseBinaryClassificationMetric):
    """Negative predictive value as ratio of true negatives and outcome negative."""       
    _code = "NPV"
    _name = 'negative_predictive_value'
    _label = "Negative Predictive Value"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tn = TrueNegative()(y, y_pred, *args, **kwargs)
        on = OutcomeNegative()(y, y_pred, *args, **kwargs)
        return tn / on

class FalseDiscoveryRate(BaseBinaryClassificationMetric):
    """False discovery rate as ratio of false positives and outcome positive."""       
    _code = "FDR"
    _name = 'false_discovery_rate'
    _label = "False Discovery Rate"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        fp = FalsePositive()(y, y_pred, *args, **kwargs)
        op = OutcomePositive()(y, y_pred, *args, **kwargs)
        return fp / op

class FalseOmissionRate(BaseBinaryClassificationMetric):
    """False omission rate as ratio of false negatives and outcome negative."""       
    _code = "FOR"
    _name = 'false_omission_rate'
    _label = "False Omission Rate"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        fn = FalseNegative()(y, y_pred, *args, **kwargs)
        on = OutcomeNegative()(y, y_pred, *args, **kwargs)
        return fn / on

class PredictedPositiveConditionRate(BaseBinaryClassificationMetric):
    """False omission rate as ratio of false negatives and outcome negative."""       
    _code = "PPCR"
    _name = 'predicted_positive_condition_rte'
    _label = "Predicted Positive Condition Rate"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)
        fp = FalsePositive()(y, y_pred, *args, **kwargs)
        return tp / y.shape[0]        
# --------------------------------------------------------------------------- #

class AUC:
    """Area under Receiver Operating Characteristic (ROC) Curve """
    @property
    def auc(self):
        try:
            return self._roc_data['auc'].sum()
        except:
            raise Exception("AUC has not been calculated.")            
    @property
    def roc_data(self):
        try:
            return self._roc_data
        except:
            raise Exception("ROC data has not been calculated.")
    def _compute_roc_data(self, df):
        # Compute predictions and sort by probabilities (desc)
        df['y_prob'] = df['y_pred']
        df['y_pred'] = int(df['y_prob'].round({'y_prob':0}))
        df = df.sort_values(by='y_prob', axis=0, ascending=False)
        # Add true positives and false positives
        df['tp'] = np.where(df['y_pred']==1 & df['y']==1, 1, 0)
        df['fp'] = np.where(df['y_pred']==1 & df['y']==0, 1, 0)
        df['tn'] = np.where(df['y_pred']==0 & df['y']==0, 1, 0)
        df['fn'] = np.where(df['y_pred']==0 & df['y']==1, 1, 0)
        # Count them up
        df['n_tp'] = df.groupby('tp').cumcount()
        df['n_fp'] = df.groupby('fp').cumcount()
        df['n_tn'] = df.groupby('tn').cumcount()
        df['n_fn'] = df.groupby('fn').cumcount()
        # Compute statistics
        df['tpr'] = df['n_tp'] / (df['n_tp']+df['n_fn'])
        df['fpr'] = df['n_fp'] / (df['n_fp']+df['n_tn'])
        width = df['fpr'].diff()
        height = (df['tpr'].shift(1) + df['tpr'])/2
        df['auc'] = width * height
        self._roc_data = df        

    def __call__(self, y, y_pred,  *args, **kwargs):                       
        df = self._format_results(y, y_pred, *args, **kwargs)
        self._compute_roc_data(df)
        return self._roc_data['auc'].sum()
# --------------------------------------------------------------------------- #

class TruePositiveRate(BaseBinaryClassificationMetric):
    """True positive rate computed as true positives / positive condition."""       
    _code = "TPR"
    _name = 'true_positive_rate'
    _label = "True Positive Rate"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)
        p = PositiveCondition()(y, y_pred, *args, **kwargs)
        return tp / p

class Sensitivity(TruePositiveRate):
    """Alias for True Positive Rate."""       
    _code = "SENS"
    _name = 'sensitivity'
    _label = "Sensitivity"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

class Recall(TruePositiveRate):
    """Alias for True Positive Rate."""       
    _code = "REC"
    _name = 'recall'
    _label = "Recall"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

class FalseNegativeRate(BaseBinaryClassificationMetric):
    """False negative rate computed as false negatives / positive condition."""       
    _code = "FNR"
    _name = 'false_negative_rate'
    _label = "False Negative Rate"
    _best = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        fn = FalseNegative()(y, y_pred, *args, **kwargs)
        p = PositiveCondition()(y, y_pred, *args, **kwargs)
        return fn / p

class TrueNegativeRate(BaseBinaryClassificationMetric):
    """True negative rate computed as true negatives / negative condition."""       
    _code = "TNR"
    _name = 'true_negative_rate'
    _label = "True Negative Rate (Specificity)"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tn = TrueNegative()(y, y_pred, *args, **kwargs)
        n = NegativeCondition()(y, y_pred, *args, **kwargs)
        return tn / n

class Specificity(TrueNegativeRate):
    """Alias for TrueNegativeRate."""       
    _code = "TNR"
    _name = 'specificity'
    _label = "Specificity (True Negative Rate)"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

class FalsePositiveRate(BaseBinaryClassificationMetric):
    """False positive rate computed as false positives / negative condition."""       
    _code = "FPR"
    _name = 'false_positive_rate'
    _label = "False Positive Rate"
    _best = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        fp = FalsePositive()(y, y_pred, *args, **kwargs)
        n = NegativeCondition()(y, y_pred, *args, **kwargs)
        return fp / n
# --------------------------------------------------------------------------- #

class MissclassificationRate(BaseBinaryClassificationMetric):
    """Missclassification rate computed as false classification / negative condition."""       
    _code = "MCR"
    _name = 'missclassification_rate'
    _label = "Missclassification Rate"
    _best = np.min
    _better  = np.less
    _worst  = np.Inf
    _epsilon_factor  = -1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        fc = FalseClassification()(y, y_pred, *args, **kwargs)        
        return fc / y.shape[0]
# -------------------------- 1ST LEVEL METRICS ------------------------------ #

class F1(BaseBinaryClassificationMetric):
    """F1 score computed as 2 * (PPV * TPR) / (PPV + TPR)."""       
    _code = "F1"
    _name = 'f1_score'
    _label = "F1 Score"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        return 2 * (ppv * tpr) / (ppv + tpr)

class F05(BaseBinaryClassificationMetric):
    """F0.5 score computed as (1.25 * PPV * TPR) / (0.25 * PPV + TPR)."""       
    _code = "F0.5"
    _name = 'f0.5_score'
    _label = "F0.5 Score"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        return (1.25 * ppv * tpr) / (0.25 * ppv + tpr)

class F2(BaseBinaryClassificationMetric):
    """F2 score computed as (5 * PPV * TPR) / (4 * PPV + TPR)."""       
    _code = "F2"
    _name = 'f2_score'
    _label = "F2 Score"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        return (5 * ppv * tpr) / (4 * ppv + tpr)

class FBeta(BaseBinaryClassificationMetric):
    """FBeta score computed as ((1+beta^2) * PPV * TPR) / ((1+beta^2) * PPV + TPR)."""       
    _code = "FBeta"
    _name = 'fbeta_score'
    _label = "FBeta Score"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, beta,  *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        return ((1+beta**2) * ppv * tpr) / ((1+beta**2) * ppv + tpr)
# --------------------------------------------------------------------------- #

class Informedness(BaseBinaryClassificationMetric):
    """Informedness is computed as TPR + TNR - 1."""       
    _code = "INFORM"
    _name = 'informedness'
    _label = "Informedness"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        tnr = TrueNegativeRate()(y, y_pred, *args, **kwargs)        
        return tpr + tnr - 1

class Markedness(BaseBinaryClassificationMetric):
    """Markedness is computed as PPV + NPV - 1."""       
    _code = "MARK"
    _name = 'markedness'
    _label = "Markedness"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred,  *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        npv = NegativePredictiveValue()(y, y_pred, *args, **kwargs)        
        return ppv + npv - 1

# --------------------------------------------------------------------------- #
class BalancedAccuracy(BaseBinaryClassificationMetric):
    """Balanced accuracy is computed as (TPR + TNR) / 2."""       
    _code = "BACC"
    _name = 'balanced_accuracy'
    _label = "Balanced Accuracy"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        tnr = TrueNegativeRate()(y, y_pred, *args, **kwargs)        
        return (tpr + tnr) / 2

class FowlkesMallowsIndex(BaseBinaryClassificationMetric):
    """Computes Fowlkes-Mallows index as sqrt(PPV * TPR)."""
    _code = "FM"
    _name = 'fowlkes_mallows_index'
    _label = "Fowlkes-Mallows Index"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        return np.sqrt(ppv * tpr)

class OptimizationPrecision(BaseBinaryClassificationMetric):
    """Computes Optimization Precision as ACC - (|TPR-TNR|)/(TPR+TNR)."""
    _code = "OPR"
    _name = 'optimization_precision'
    _label = "Optimization Precision"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        acc = Accuracy()(y, y_pred, *args, **kwargs)        
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)        
        tnr = TrueNegativeRate()(y, y_pred, *args, **kwargs)        
        return acc - (abs(tpr-tnr)) / (tpr+tnr)


class Jaccard(BaseBinaryClassificationMetric):
    """Computes Jaccard's Similarity Coefficient as TP/(TP+FP+FN)."""
    _code = "JAC"
    _name = 'jaccard_similarity_coefficient'
    _label = "Jaccard's Similarity Coefficient"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)        
        fp = FalsePositive()(y, y_pred, *args, **kwargs)        
        fn = FalseNegative()(y, y_pred, *args, **kwargs)            
        return tp / (tp+fp+fn)


# --------------------------------------------------------------------------- #

class CohensKappa(BaseBinaryClassificationMetric):
    """Computes Cohen's Kappa as (ACC-CKc) / (1-CKc)."""
    _code = "CK"
    _name = 'cohens_kappa'
    _label = "Cohen's Kappa"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        acc = Accuracy()(y, y_pred, *args, **kwargs)        
        ckc = CohensKappaChance()(y, y_pred, *args, **kwargs)        
        return (acc-ckc) / (1-ckc)

# --------------------------------------------------------------------------- #
class GeometricMean(BaseBinaryClassificationMetric):
    """Computes Geometric Mean as sqrt(tp*tn)."""
    _code = "GM"
    _name = 'geometric_mean'
    _label = "Geometric Mean"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)        
        tn = TrueNegative()(y, y_pred, *args, **kwargs)        
        return np.sqrt(tp*tn)

# --------------------------------------------------------------------------- #
class AdjustedGeometricMean(BaseBinaryClassificationMetric):
    """Computes Adjusted Geometric Mean as GM+TNR(FP+TN)/(1+FP+TN) if TPR>0, 0 otherwise."""
    _code = "AGM"
    _name = 'adjusted_geometric_mean'
    _label = "Adjusted Geometric Mean"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        gm = GeometricMean()(y, y_pred, *args, **kwargs)       
        tpr = TruePositiveRate()(y, y_pred, *args, **kwargs)         
        tnr = TrueNegativeRate()(y, y_pred, *args, **kwargs)        
        fp = FalsePositive()(y, y_pred, *args, **kwargs)        
        tn = TrueNegative()(y, y_pred, *args, **kwargs)        
        agm = 0
        if tpr > 0:
            agm = gm+tnr(fp+tn) / (1+fp+tn)
        return agm

# -------------------------- 2nd LEVEL METRIC ------------------------------- #

class MatthewsCorrelationCoefficient(BaseBinaryClassificationMetric):
    """Computes Mathew's Correlation Coefficient as sqrt(INFORM * MARK)."""
    _code = "MCC"
    _name = 'matthews_correlation_coefficient'
    _label = "Matthew's Correlation Coefficient"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        inform = Informedness()(y, y_pred, *args, **kwargs)        
        mark = Markedness()(y, y_pred, *args, **kwargs)        
        return np.sqrt(inform * mark)

# --------------------------- TEST DIAGNOSTICS ------------------------------ #
class Significance(BaseBinaryClassificationMetric):
    """Computes Significance as [(TP*TN-FP*FN)^2 (TP+TN+FP+FN)] / [(TP+FP)(TP+FN)(TN+FP)(TN+FN)]."""
    _code = "SIG"
    _name = 'significance'
    _label = "Significance"
    _best = np.max
    _better  = np.greater
    _worst  = -np.Inf
    _epsilon_factor  = 1
    _is_probability_metric = False

    def __call__(self, y, y_pred, *args, **kwargs):           
        tp = TruePositive()(y, y_pred, *args, **kwargs)        
        tn = TrueNegative()(y, y_pred, *args, **kwargs)        
        fp = FalsePositive()(y, y_pred, *args, **kwargs)        
        fn = FalseNegative()(y, y_pred, *args, **kwargs)        
        numerator = ((tp*tn)-(fp*fn))**2 * (tp+tn+fp+fn)
        denominator = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
        return numerator / denominator