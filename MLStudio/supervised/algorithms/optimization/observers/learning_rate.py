#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : learning_rate.py                                                  #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, May 15th 2020, 9:48:31 pm                           #
# Last Modified : Friday, May 15th 2020, 9:48:31 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Learning rate schedules."""
from abc import abstractmethod
from copy import deepcopy
import math
import numpy as np
from tabulate import tabulate

from mlstudio.supervised.algorithms.optimization.observers.base import Observer, PerformanceObserver
from mlstudio.supervised.metrics.regression import MeanSquaredError
from mlstudio.utils.validation import validate_bool
from mlstudio.utils.validation import validate_int
from mlstudio.utils.validation import validate_objective, validate_optimizer
from mlstudio.utils.validation import validate_monitor, validate_string
from mlstudio.utils.validation import validate_zero_to_one, validate_int
from mlstudio.utils.validation import validate_monitor
# --------------------------------------------------------------------------  #
class LearningRateSchedule(Observer):
    """Base class for learning rate schedules. 
    
    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate

    decay_factor : float (default=0.5) 
        The factor by which the learning rate is decayed

    """

    @abstractmethod
    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor=0.5):    
        super(LearningRateSchedule, self).__init__()
        self.eta0 = eta0
        self.eta_min = eta_min  
        self.decay_factor = decay_factor        

    def _validate(self):
        """Performs validation of hyperparameters."""
        validate_zero_to_one(param=self.eta0, 
                             param_name='eta0', 
                             left='open', right='open')
        validate_zero_to_one(param=self.eta_min, 
                             param_name='eta_min', 
                             left='open', right='open')
        validate_zero_to_one(param=self.decay_factor, 
                             param_name='decay_factor',
                             left='closed', right='open')

    @abstractmethod
    def _compute_learning_rate(self, epoch, log):
        pass

    def on_train_begin(self, log=None):
        super(LearningRateSchedule, self).on_train_begin(log)        
        self._validate()
        self.model.eta = self.eta0

    def on_epoch_end(self, epoch, log=None):
        super(LearningRateSchedule, self).on_epoch_begin(epoch=epoch, log=log)   
        self.model.eta = max(self.eta_min,\
            self._compute_learning_rate(epoch=epoch, log=log))
    
        

# --------------------------------------------------------------------------  #
class StepDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:
    .. math:: \eta_0 \times \gamma^{\text{floor((1+epoch)/decay_steps)}}
    
    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        
         
    decay_factor : float (default= 0.5)
        The factor used as the base of the polynomial used to compute
        the decayed learning rate.

    step_size : int (default=100)
        The total number of steps taken between drops in the 
        learning rate.

    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor=0.5,
                       step_size=10):        
        super(StepDecay, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)              

        self.name = "Step Decay Learning Rate Schedule"
        self.step_size = step_size

    def _validate(self):
        """Performs hyperparameter validation """
        super(StepDecay, self)._validate()
        validate_int(param=self.step_size, 
                     param_name='step_size',
                     minimum=1, left='closed', right='open')                

    def _compute_learning_rate(self, epoch, log):        
        exponent = (1 + epoch) // self.step_size
        return self.eta0 * \
            np.power(self.decay_factor, exponent)

    def on_train_begin(self, log=None):
        super(StepDecay, self).on_train_begin(log=log)        
        self.step_size = min(self.step_size, self.model.epochs)                    

# --------------------------------------------------------------------------  #
class TimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot t} 

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    decay_factor : float or 'optimal' (default= 'optimal')
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of epochs

    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor='optimal'):        
        super(TimeDecay, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)

        self.name = "Time Decay Learning Rate Schedule"              

    def _compute_optimal_decay_factor(self):
        return 1 - (self.eta0 - self.eta_min)\
            / self.model.epochs

    def _compute_learning_rate(self, epoch, log):
        return self.eta0 / \
            (1 + self.decay_factor * epoch)

    def on_train_begin(self, log=None):
        if self.decay_factor == 'optimal':
            self.decay_factor = self._compute_optimal_decay_factor()            
        super(TimeDecay, self).on_train_begin(log)        

# --------------------------------------------------------------------------  #
class SqrtTimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot \sqrt{t}} 

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    decay_factor : float (default= 0.5)
        The factor used as to decay the learning rate.
    """
    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor=0.5):        
        super(SqrtTimeDecay, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)              

        self.name = "Sqrt Time Decay Learning Rate Schedule"

    def _compute_learning_rate(self, epoch, log):
        return self.eta0 / \
            (1 + self.decay_factor * np.sqrt(epoch))       
               


# --------------------------------------------------------------------------  #
class ExponentialDecay(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    decay_factor : float (default= 0.1)
        The decay factor used in the exponent for the learning rate computation

    step_size : int(default-100)
        The number of steps between each drop in learning rate.
    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor=0.96,
                       step_size=100,
                       staircase=False):        
        super(ExponentialDecay, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)

        self.step_size = step_size
        self.staircase = staircase
        self.name = "Exponential Decay Learning Rate Schedule"

    def _compute_learning_rate(self, epoch, log):
        exponent = epoch // self.step_size if self.staircase else \
                    epoch / self.step_size
        return self.eta0 * \
            np.power(self.decay_factor, exponent)


# --------------------------------------------------------------------------  #
class ExponentialSchedule(LearningRateSchedule):
    """ Exponential scheduling learning rate schedule as:
    .. math:: \eta_t=\eta_0 * 10^{-t/r}

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    step_size : int(default-100)
        The number of steps between each drop in learning rate.
    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       step_size=100):        
        super(ExponentialSchedule, self).__init__(
            eta0=eta0,
            eta_min=eta_min)

        self.step_size = step_size
        self.name = "Exponential Schedule Learning Rate Schedule"

    def _compute_learning_rate(self, epoch, log):
        return self.eta0 *  10 ** (-epoch / self.step_size)
            
# --------------------------------------------------------------------------  #
class PolynomialDecay(LearningRateSchedule):
    """ Polynomial decay learning rate schedule given by:
    .. math:: \eta_t=\eta_0 \cdot (1 - \frac{t}{\text{epochs}})^p

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    power : float (default=1)
        The power to which the polynomial is decayed 

    decay_steps : int(default=100)
        The number of times the learning rate is decayed from the initial 
        learning rate to its minimum.
    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       power=1.0,
                       decay_steps=100,
                       cycle=False):        
        super(PolynomialDecay, self).__init__(
            eta0=eta0,
            eta_min=eta_min)   

        self.name = "Polynomial Decay Learning Rate Schedule"
        self.power = power
        self.decay_steps = decay_steps
        self.cycle = cycle

    def _validate(self):
        """Performs hyperparameter validation"""
        super(PolynomialDecay, self)._validate()
        validate_zero_to_one(param=self.power, 
                             param_name='power',
                             left='open', right='closed')                                

    def _compute_learning_rate(self, epoch, log):
        if self.cycle:
            self.decay_steps = self.decay_steps * np.ceil(epoch / self.decay_steps)
        else:
            epoch = min(epoch, self.decay_steps)
        learning_rate_range = self.eta0 - self.eta_min
        base = (1 - epoch / self.decay_steps)
        return learning_rate_range * np.power(base, self.power) + self.eta_min

# --------------------------------------------------------------------------  #
class PowerSchedule(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 * (1+\frac{t}{r})^{-c}

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    power : float (default=-1)
        The power to which the polynomial is decayed 
    """
    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_steps=100,
                       power=1.0):            
        super(PowerSchedule, self).__init__(
            eta0=eta0,
            eta_min=eta_min)
        self.name = "Power Schedule Learning Rate Schedule"              
        self.decay_steps = decay_steps
        self.power = power        

    def _validate(self):
        """Performs hyperparameter """
        super(PowerSchedule, self)._validate()
        validate_zero_to_one(param=self.power, 
                             param_name='power',
                             left='closed', right='closed')    
        
        validate_int(param=self.decay_steps, 
                     param_name='decay_steps',
                     minimum=1, left='open', right='open') 

    def _compute_learning_rate(self, epoch, log):
        return self.eta0 * (1 + epoch/self.decay_steps)**-self.power

# --------------------------------------------------------------------------  #
class BottouSchedule(LearningRateSchedule):
    """ Learning rate schedule as described in:
    https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf
    
    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate        

    decay_factor : float (default=0.5)
        The factor by which the learning rate is decayed.
    
    """

    def __init__(self, eta0=0.1, 
                       eta_min=1e-4,
                       decay_factor=0.5):    
        super(BottouSchedule, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)
        
        self.name = "Bottou Schedule Learning Rate Schedule"

    def _compute_learning_rate(self, epoch, log):
        return self.eta0 * (1 + self.eta0 * \
            self.decay_factor * epoch)**(-1)

# --------------------------------------------------------------------------- #
#                               ADAPTIVE                                      #
# --------------------------------------------------------------------------- #
class Adaptive(LearningRateSchedule):
    """Decays the learning rate if performance plateaus.

    Parameters
    ----------
    eta0 : float (default=0.1)
        The initial learning rate that is decayed during optimization. This 
        will override the eta0 parameter on the estimator.

    eta_min : float(default=1e-4)
        The minimum allowable learning rate

    decay_factor : float (default=0.5)
        The factor by which the learning rate is decayed when stabilization
        is encountered.

    monitor : str, optional (default='train_cost')
        Specifies the variable that we are monitoring for performance evaluation purposes.
        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=10)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, eta0=0.1, eta_min=1e-4,
                 decay_factor=0.5, monitor='train_cost',  epsilon=0.001, 
                 patience=10, performance_observer=None):
        super(Adaptive, self).__init__(
            eta0=eta0,
            eta_min=eta_min,
            decay_factor=decay_factor)

        self.name = "Adaptive Learning Rate Schedule"        
        self.monitor = monitor
        self.epsilon = epsilon
        self.patience = patience  
        self.performance_observer = performance_observer

    def on_train_begin(self, log=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """        
        self._performance_observer = deepcopy(self.performance_observer)
        super(Adaptive, self).on_train_begin(log)
        try:
            self._performance_observer.set_model(self.model)
            self._performance_observer.monitor = self.monitor
            self._performance_observer.epsilon = self.epsilon
            self._performance_observer.patience = self.patience
        except:
            raise ValueError("The performance observer object is required for this object.")
        
        self._performance_observer.on_train_begin(log)        

    def _compute_learning_rate(self, epoch, log):
        lr = log.get('eta') * self.decay_factor \
            if self._performance_observer.stabilized else log.get('eta')
        return lr


    def on_epoch_end(self, epoch, log=None):
        """Determines whether convergence has been achieved.
        Parameters
        ----------
        epoch : int
            The current epoch number
        log : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  
        """        
        log = log or {}                        
        self._performance_observer.on_epoch_end(epoch, log)
        super(Adaptive, self).on_epoch_end(epoch, log)        
        
           