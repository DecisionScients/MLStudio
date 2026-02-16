#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : observers.py                                                       #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 7:27:16 pm                          #
# Last Modified : Sunday, March 15th 2020, 7:37:00 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Module containing functionality called during the training process.

Note: The ObserverList and Observer abstract base classes were inspired by
the Keras implementation.  
"""
from abc import ABC, abstractmethod, ABCMeta
from copy import copy, deepcopy
import warnings
warnings.filterwarnings("once", category=UserWarning, module='base')

import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mlstudio.utils.validation import validate_int, validate_zero_to_one
from mlstudio.utils.validation import validate_monitor
# --------------------------------------------------------------------------- #
#                             OBSERVER LIST                                   #
# --------------------------------------------------------------------------- #
class ObserverList(ABC, BaseEstimator):
    """Container of observers."""

    def __init__(self, observers=None):
        """ObserverList constructor
        
        Parameters
        ----------
        observers : list
            List of 'Observer' instances.        
        """
        self._observers = observers or []                        
        self.params = {}
        self.model = None


    def append(self, observer):
        """Appends observer to list of observers.
        
        Parameters
        ----------
        observer : Observer instance            
        """
        self._observers.append(observer)

    def set_params(self, params):
        """Sets the parameters variable, and in list of observers.
        
        Parameters
        ----------
        params : dict
            Dictionary containing model parameters
        """
        self.params = params
        for observer in self._observers:
            observer.set_params(params)

    def set_model(self, model):
        """Sets the model variable, and in the list of observers.
        
        Parameters
        ----------
        model : Estimator or subclass instance 
        
        """
        self.model = model
        for observer in self._observers:
            observer.set_model(model)

    def on_batch_begin(self, batch, log=None):
        """Calls the `on_batch_begin` methods of its observers.

        Parameters
        ----------
        batch : int
            Current training batch

        log: dict
            Currently no data is set to this parameter for this class. This may
            change in the future.
        """
        log = log or {}
        for observer in self._observers:
            observer.on_batch_begin(batch, log)

    def on_batch_end(self, batch, log=None):
        """Calls the `on_batch_end` methods of its observers.
        
        Parameters
        ----------
        batch : int
            Current training batch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """
        log = log or {}
        for observer in self._observers:
            observer.on_batch_end(batch, log)

    def on_epoch_begin(self, epoch, log=None):
        """Calls the `on_epoch_begin` methods of its observers.

        Parameters
        ----------        
        epoch: integer
            Current training epoch

        log: dict
            Currently no data is passed to this argument for this method
            but that may change in the future.
        """
        log = log or {}
        for observer in self._observers:
            observer.on_epoch_begin(epoch, log)

    def on_epoch_end(self, epoch, log=None):
        """Calls the `on_epoch_end` methods of its observers.
        This function should only be called during train mode.

        Parameters
        ----------
        epoch: int
            Current training epoch
        
        log: dict
            Metric results for this training epoch, and for the
            validation epoch if validation is performed.
        """
        log = log or {}
        for observer in self._observers:
            observer.on_epoch_end(epoch, log)

    def on_train_begin(self, log=None):
        """Calls the `on_train_begin` methods of its observers.

        Parameters
        ----------
        log: dict
            Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for observer in self._observers:
            observer.on_train_begin(log)

    def on_train_end(self, log=None):
        """Calls the `on_train_end` methods of its observers.

        Parameters
        ----------
        log: dict
            Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for observer in self._observers:
            observer.on_train_end(log)

    def __iter__(self):
        return iter(self._observers)

# --------------------------------------------------------------------------- #
#                             OBSERVER CLASS                                  #
# --------------------------------------------------------------------------- #
class Observer(ABC, BaseEstimator):
    """Abstract base class used to build new observers."""
    def __init__(self):
        """Observer class constructor."""
        self.params = None
        self.model = None

    def set_params(self, params):
        """Sets parameters from estimator.

        Parameters
        ----------
        params : dict
            Dictionary containing estimator parameters
        """ 
        self.params = params

    def set_model(self, model):
        """Stores model in Observer object.

        Parameters
        ----------
        model : Estimator
            Estimator object
        """
        self.model = model

    def on_batch_begin(self, batch, log=None):
        """Logic executed at the beginning of each batch.

        Parameters
        ----------
        batch : int
            Current training batch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """        
        pass

    def on_batch_end(self, batch, log=None):   
        """Logic executed at the end of each batch.
        
        Parameters
        ----------
        batch : int
            Current training batch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        pass

    def on_epoch_begin(self, epoch, log=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        pass

    def on_epoch_end(self, epoch, log=None):
        """Logic executed at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                      
        pass

    def on_train_begin(self, log=None):
        """Logic executed at the beginning of training.
        
        Parameters
        ----------        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                      
        pass

    def on_train_end(self, log=None):
        """Logic executed at the end of training.
        
        Parameters
        ----------        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """               
        pass
# --------------------------------------------------------------------------- #
#                             PERFORMANCE BASE                                #
# --------------------------------------------------------------------------- #
class PerformanceObserver(Observer):
    """Base class for performance observers."""

    def __init__(self): 
        super(PerformanceObserver, self).__init__()               

    @property
    def name(self):
        return "Performance Base Observer"

    @property
    def monitor(self):
        return self._monitor

    @monitor.setter
    def monitor(self, x):
        self._monitor = x

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

        # If a variable has been designated for monitoring, obtain what constitutes a better
        # or best scores from the model's scorer object. 
        if 'score' in self._monitor:
            self._best = self.model.scorer.best
            self._better = self.model.scorer.better
        # Otherwise, cost is being monitored and best and better costs are min and
        # less, respectively
        else:
            self._best = min            
            self._better = np.less            

        # Validation
        validate_monitor(self.monitor)
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
            self._baseline = self._best(current, self._baseline)
        else:
            self._stabilized = False               

    def _get_current_value(self, log):
        """Obtain the designated metric or fallback metric from the log."""
        current = log.get(self.monitor)
        if not current:
            current = log.get('train_score')            
            msg = self.monitor + " not evaluated for this estimator. Using\
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
        


        
