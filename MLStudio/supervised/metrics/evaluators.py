# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \evaluators.py                                                    #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 11:26:15 am                       #
# Last Modified : Saturday, July 25th 2020, 11:26:16 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Classes that perform estimator evaluations using panels of metrics."""
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict

import pandas as pd
# --------------------------------------------------------------------------- #
class AbstractEvaluator(ABC):
    """Abstract base class for all evaluators."""

    def __init__(self, estimator, panel, printer=None):
        self._estimator = estimator
        self._panel = panel
        self._printer = printer
        self.scores_ = []

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, x):      
        if not hasattr(x, 'score'):
            msg = "The estimator parameter does not have the required 'score' method."  
        self._estimator = x

    @property
    def panel (self):
        return self._panel

    @panel.setter
    def panel (self, x):
        if not isinstance(x, AbstractPanel):
            msg = "The panel parameter is not a valid Panel object. Should be a subclass of 'AbstractPanel'"
            raise ValueError(msg)
        self._panel = x        
    
    def __call__(self, X, y):
        pass

    def print(self):
        title = self._panel.label + " : " + self._panel.description
        code, label, value = []
        for result in self.results_:
            code.append(result.code)
            label.append(result.label)
            value.append(result.value)
        d = OrderedDict()
        d['Code': code, "Metric": metric, "Value": value]
        df = pd.DataFrame.from_dict(d)
        printer.print_dataframe(content=df, title=title)

# --------------------------------------------------------------------------- #
class RegressionEvaluator(AbstractEvaluator):

    def __call__(self, X, y):

        y_pred = self._estimator.score(X, y)        
        result = OrderedDict()
        
        for code, metric in self._panel.metrics.items():
            self._scores[code] = OrderedDict()
            self._scores[code]['label'] = metric.label
            self._scores[code]['value'] = metric.instance(y, y_pred)

        return self._scores

# --------------------------------------------------------------------------- #
class BinaryClassEvaluator(AbstractEvaluator):        

    def __call__(self, X, y):

        y_pred = self._estimator.score(X, y)                
        
        for code, metric in self._panel.metrics.items():
            self._scores[code] = OrderedDict()
            self._scores[code]['label'] = metric.label
            self._scores[code]['value'] = metric.instance(y, y_pred)

        return self._scores    

# --------------------------------------------------------------------------- #
class MultiClassEvaluator(AbstractEvaluator):        

    def __call__(self, X, y):

        output_encoded = {}
        output_encoded[False] = {}
        output_encoded[True] = {}        

        output_encoded[False]['y'] = y
        output_encoded[False]['y_pred'] = self._estimator.predict(X)
        if self._panel.encoded_output:
            output_encoded[True]['y'] = self._data_processor.fit_transform(y)
            output_encoded[False]['y_pred'] = self._estimator.predict(X)

        y_pred = self._estimator.score(X, y)                
        
        for code, metric in self._panel.metrics.items():

            self._scores[code] = OrderedDict()
            self._scores[code]['label'] = metric.label
            self._scores[code]['value'] = metric.instance(y, y_pred)

        return self._scores    

            
