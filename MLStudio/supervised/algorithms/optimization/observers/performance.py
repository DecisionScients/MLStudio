# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \performance.py                                                   #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, July 28th 2020, 9:18:00 pm                         #
# Last Modified : Tuesday, July 28th 2020, 9:18:38 pm                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Module containing functionality that acts upon existing performance data."""
from abc import ABC, abstractmethod, ABCMeta
import warnings
warnings.filterwarnings("once", category=UserWarning, module='base')

import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mlstudio.utils.validation import validate_int, validate_zero_to_one
from mlstudio.utils.validation import validate_monitor
# --------------------------------------------------------------------------- #
#                             PERFORMANCE OBSERVER                            #
# --------------------------------------------------------------------------- #
class PerformanceObserver(Observer):
    """Base class for performance observers."""

    def __init__(self): 
        super(PerformanceObserver, self).__init__()               
        self._metric = None
        self._epsilon = None
        self._patience = None

    @property
    def name(self):
        return "Performance Base Observer"

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, x):
        self._metric = x

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, x):
        self._epsilon = x

    @property
    def patience(self):
        return self._patience

    @patience.setter
    def patience(self, x):
        self._patience = x        

    @property
    def stabilized(self):
        return self._stabilized   
       
    def _validate(self):     
        validate_zero_to_one(param=self.epsilon, param_name='epsilon',
                             left='closed', right='closed')       
        validate_int(param=self.patience, param_name='patience',
                     minimum=0, left='open', right='open')

    def on_train_begin(self, log=None):                
        """Sets key variables at beginning of training.        
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(PerformanceObserver, self).on_train_begin(log=log)        
        log = log or {}        
        self._validate()
        
        self._baseline = None        
        self._iter_no_improvement = 0
        self._stabilized = False

        # If score metric is designated, obtain what constitutes a better
        # or best scores from the model's metric object. 
        if 'score' in self.metric:
            self._best = self.model.scorer.best
            self._better = self.model.scorer.better
        # Otherwise, the metric is cost and best and better costs are min and
        # less, respectively
        else:
            self._best = np.min            
            self._better = np.less            

        # Validation
        validate_monitor(self.metric)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon',
                             left='open', right='open')
        validate_int(param=self.patience, param_name='patience')

        # log data
        self.performance_log_ = {}

    def _update_log(self, epoch, log):
        """Creates log dictionary of lists of performance results."""
        log['epoch'] = epoch
        log['baseline']= self._baseline        
        log['iter_no_improvement']= self._iter_no_improvement
        log['stabilized'] = self._stabilized
        for k,v in log.items():
            self.performance_log_.setdefault(k,[]).append(v)

    def _improved(self, current):   
        """Returns true if relative change is greater than epsilon."""  
        return self._better(current, self._baseline) and \
            (abs(current-self._baseline) / abs(self._baseline)) > \
                self.epsilon

    def _process_improvement(self, current, log=None):
        """Sets values of parameters and attributes if improved."""
        self._iter_no_improvement = 0            
        self._stabilized = False
        self._baseline = current
        self._best_results = log

    def _process_no_improvement(self, current, log=None):
        """Sets values of parameters and attributes if no improved."""    
        self._iter_no_improvement += 1  
        if self._iter_no_improvement == self.patience:
            self._stabilized = True
            # We reset iter_no_improvement and baseline to better of the 
            # current value and prior baseline. This gives the estimator
            # another 'patience' epochs to achieve real improvement from
            # new baseline.  
            self._iter_no_improvement = 0
            self._baseline = self._best((current, self._baseline)) 
        else:
            self._stabilized = False               

    def _get_current_value(self, log):
        """Obtain the designated metric or fallback metric from the log."""
        current = log.get(self.metric)
        if not current:
            current = log.get('train_score')            
            msg = self.metric + " not evaluated for this estimator. Using\
                train_score instead."
            warnings.warn(msg, UserWarning)
        return current

    def on_epoch_end(self, epoch, log=None):
        """Logic executed at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                  
        log = log or {}   
        
        # Obtain current performance
        current = self._get_current_value(log)

        # Handle first iteration as an improvement by default
        if self._baseline is None:       
            log['improved'] = True
            self._process_improvement(current, log)    

        # Otherwise, evaluate the direction and magnitude of the change        
        elif self._improved(current):
            log['improved'] = True
            self._process_improvement(current, log)
        else:
            log['improved'] = False
            self._process_no_improvement(current, log)

        # Log results 
        self._update_log(epoch, log)
        


        
