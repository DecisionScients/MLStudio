#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : early_stop.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, May 15th 2020, 9:16:41 pm                           #
# Last Modified : Sunday, June 14th 2020, 11:49:38 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
"""Early stop classes."""
from abc import ABC, abstractmethod
import collections
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[5]
print(PROJECT_DIR)
site.addsitedir(PROJECT_DIR)

from copy import copy, deepcopy
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mlstudio.supervised.algorithms.optimization.observers.base import Observer
from mlstudio.utils.validation import validate_monitor, validate_int
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
#                               EARLYSTOP                                     #
# --------------------------------------------------------------------------- #
class EarlyStop(Observer,BaseEstimator):
    """Performances and log model performance, critical points and stability. 

    Performance is defined in terms of:

        * Metric : A metric to observe. This can be training error, validation
            score, gradient norm or the like 

        * Epsilon : A mininum amount of relative change in the observed
            metric required to consider the optimization in a productive
            state.

        * Patience : The number of consecutive epochs or iterations of
            non-improvement that is tolerated before considering an
            optimization stabilized.    

    Parameters
    ----------
    metric : str, optional (default='train_cost')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost' : Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.01)
        The amount of relative change in the observed metric considered to be
        a sufficient improvement in performance. 

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that is 
        tolerated before considering the optimization stable.

    """

    def __init__(self, monitor='val_score', epsilon=0.001, patience=5, performance_observer=None): 
        super(EarlyStop, self).__init__()
        self.monitor = monitor
        self.epsilon = epsilon
        self.patience = patience
        self.performance_observer = performance_observer
        self.name = "EarlyStop Observer"        

    def on_train_begin(self, log=None):
        """Logic executed at the beginning of training.
        
        Parameters
        ----------
        log: dict
            Currently not used
        """
        self._performance_observer = deepcopy(self.performance_observer)
        super(EarlyStop, self).on_train_begin(log)
        if self._performance_observer:
            self._performance_observer.set_model(model=self.model)
            self._performance_observer.monitor = self.monitor
            self._performance_observer.epsilon = self.epsilon
            self._performance_observer.patience = self.patience
        else:
            raise Exception("EarlyStop requires a PerformanceObserver object.")

        self._performance_observer.on_train_begin(log)
        super(EarlyStop, self).on_train_begin(log=log)

    def on_epoch_end(self, epoch, log=None):
        """Logic executed at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                  
        super(EarlyStop, self).on_epoch_end(epoch=epoch, log=log)
        self._performance_observer.on_epoch_end(epoch, log)
        if self._performance_observer.stabilized:
            self.model.converged = True       


#%%


# %%
