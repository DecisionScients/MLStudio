# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \base.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:26:47 am                        #
# Last Modified : Thursday, July 16th 2020, 2:26:47 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Base classes for performance analytics subclasses. 

The PerformanceAnalytics class occupies the top of the class hierarchy. 

From this class, we derive the Metrics and Metrics subclasses.  Metrics,
primarily used in classification problems, are derived from taking a 
measurement. Examples include the number of true positives. A metric is
a calculation between two or more measures, e.g. true positive rate.
Metrics can be reported; whereas metrics can be used for scoring and
generalized performance estimation.

The RegressionMetrics, ClassificationMetrics and ClassificationMetrics
classes are derived from Metrics and Metrics.
"""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
# --------------------------------------------------------------------------- #
class BasePerformance(ABC, BaseEstimator):
    """Abstract base class for performance analytics."""
    
    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    def _format_results(self, y, y_pred):
        """Formats results into dataframe for evaluation."""
        y = np.array(y)
        y_pred = np.array(y_pred)
        d = {'y': y, 'y_pred': y_pred}
        df = pd.DataFrame(data=d)
        return df                    

    @property
    def id(self):
        return self._id

    @property
    def code(self):
        return self._code         

    @code.setter
    def code(self, x):
        self._code = x        

    @property
    def name(self):
        return self._name         

    @name.setter
    def name(self, x):
        self._name = x

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, x):
        self._label = x     

    @property
    def best(self):
        return self._best

    @property
    def better(self):
        return self._better

    @property
    def worst(self):
        return self._worst

    @property
    def epsilon_factor(self):
        return self._epsilon_factor   

    @property
    def is_probability_metric(self):
        return self._is_probability_metric   

class BaseMetric(BasePerformance):
    """Base class for performance metrics."""

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseRegressionMetric(BaseMetric):
    """Base class for regresssion metrics."""

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")


class BaseBinaryClassificationMetric(BaseMetric):
    """Base class for regresssion metrics."""

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")


class BaseMultiClassificationMetric(BaseMetric):
    """Base class for regresssion metrics."""

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

