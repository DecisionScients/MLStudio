#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : gradient_descent_optimizers.py                                    #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 9:13:15 pm                         #
# Last Modified : Saturday, May 16th 2020, 9:13:16 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Gradient descent optimization algorithms."""
from abc import ABC, abstractmethod
import math

import numpy as np
from sklearn.base import BaseEstimator  

from mlstudio.utils.data_manager import GradientScaler   
# --------------------------------------------------------------------------  #
class Optimizer(ABC, BaseEstimator):
    """Base class for all optimizers."""

    @abstractmethod        
    def __call__(self, gradient, learning_rate, theta, **kwargs):   
        """Computes the parameter updates.
        
        Parameters
        ----------
        gradient : func
            The function that performs the gradient computation 

        learning_rate : float
            The learning rate from the estimator object.

        theta : array-like
            The model parameters

        **kwargs : dict
            Arbitrary parameters used for computing the gradient.

        Returns
        -------
        theta : array-like
            The updated parameters of the model

        grad : array-like
            The gradient of the objective function w.r.t. parameters theta.
        """
        pass



# --------------------------------------------------------------------------  #
class GradientDescentOptimizer(Optimizer):
    """Standard gradient descent optimizer."""

    def __init__(self):
        self.name = "Gradient Descent"
    
    def __call__(self, gradient, learning_rate, theta, **kwargs):        
        grad = gradient(theta, **kwargs)
        theta = theta - learning_rate * grad
        return theta, grad

# --------------------------------------------------------------------------  #
class Momentum(Optimizer):
    """Standard gradient descent optimizer."""

    def __init__(self, gamma=0.9):
        self.name = "Momentum"
        self.gamma = gamma
        self._velocity = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):             
        grad = gradient(theta, **kwargs)
        self._velocity = self.gamma * self._velocity + learning_rate * grad
        theta = theta - self._velocity
        return theta, grad

# --------------------------------------------------------------------------  #
class Nesterov(Optimizer):
    """Nesterov accelerated gradient optimizer."""

    def __init__(self, gamma=0.9, ):
        self.name = "Nesterov"
        self.gamma = gamma
        self._velocity = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):
        next_position = theta - self.gamma * self._velocity        
        grad = gradient(next_position, **kwargs)
        self._velocity = self.gamma * self._velocity + learning_rate * grad
        theta = theta - self._velocity
        return theta, grad

# --------------------------------------------------------------------------  #
class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(self, epsilon=1e-8):
        self.name = "Adagrad"
        self.epsilon = epsilon
        self.sum_squared_gradients = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):
        grad = gradient(theta, **kwargs)
        # Accumulate the square of gradients up to time t
        self.sum_squared_gradients += np.square(grad)    
        # Convert to diagonal matrix
        Gt_ii = np.diag(self.sum_squared_gradients)        
        # Add an epsilon to diagonal matrix to avoid division by zero
        epsilon = np.diag(np.full(len(grad),self.epsilon))      
        Gt_e = np.add(Gt_ii,epsilon)
        # Take square root of sum of GT and e
        Gt_e_sqrt = np.sqrt(Gt_e)
        # Create its inverse
        Gt_e_sqrt_inv = np.linalg.inv(Gt_e_sqrt)
        # Perform update
        theta = theta - learning_rate * Gt_e_sqrt_inv.dot(grad)
        return theta, grad        

# --------------------------------------------------------------------------  #
class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(self, gamma=0.9, epsilon=1e-8):
        self.name = "Adadelta"
        self.gamma = gamma
        self.epsilon = epsilon
        self.avg_sqr_gradient = 0
        self.avg_sqr_delta_theta = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):                
        grad = gradient(theta, **kwargs)     

        self.avg_sqr_gradient = self.gamma * self.avg_sqr_gradient + \
            (1 - self.gamma) * np.square(grad)            
        rms_grad = np.sqrt(np.array(np.square(np.array(self.avg_sqr_gradient, ndmin=1)) + np.array(self.epsilon, ndmin=1), ndmin=1))
        
        delta_theta = -learning_rate / rms_grad  * grad            

        self.avg_sqr_delta_theta = self.gamma * self.avg_sqr_delta_theta + \
            (1 - self.gamma) * np.square(np.array(delta_theta, ndmin=1))
        rms_delta_theta = np.sqrt(np.array(np.square(np.array(self.avg_sqr_delta_theta, ndmin=1)) + np.array(self.epsilon, ndmin=1),ndmin=1))

        delta_theta = - (rms_delta_theta / rms_grad).dot(grad)

        theta = theta + delta_theta
        return theta, grad

# --------------------------------------------------------------------------  #
class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, gamma=0.9, epsilon=1e-8):     
        self.name = "RMSprop"   
        self.gamma = gamma
        self.epsilon = epsilon
        self.avg_sqr_gradient = 0
        self.avg_sqr_delta_theta = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):                
        grad = gradient(theta, **kwargs)        
        self.avg_sqr_gradient = self.gamma * self.avg_sqr_gradient + \
            0.1 * np.square(grad)
        rms_grad = np.sqrt(np.array(np.square(np.array(self.avg_sqr_gradient, ndmin=1)) + np.array(self.epsilon, ndmin=1),ndmin=1))
        theta = theta - (learning_rate / rms_grad) * grad
        
        return theta, grad

# --------------------------------------------------------------------------  #
class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, beta_one=0.9, beta_two=0.999, epsilon=10e-8):
        self.name = "Adam"
        self.beta_one = beta_one
        self.beta_two = beta_two        
        self.epsilon = epsilon


        self.t = 0
        self.m = 0        
        self.v = 0
    
    def __call__(self, gradient, learning_rate, theta, **kwargs):                
        self.t += 1
        grad = gradient(theta, **kwargs)        
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)
        # Bias corrected moment estimates
        m_hat = self.m / (1 - self.beta_one**self.t)
        v_hat = self.v / (1 - self.beta_two**self.t)

        theta = theta - learning_rate / (np.sqrt(np.array(v_hat)) + np.array(self.epsilon)) * m_hat        
        
        return theta, grad        

# --------------------------------------------------------------------------  #
class AdaMax(Optimizer):
    """AdaMax optimizer."""

    def __init__(self, beta_one=0.9, beta_two=0.999):    
        self.name = "AdaMax"   
        self.beta_one = beta_one
        self.beta_two = beta_two        
        self.t = 0
        self.m = 0
        self.u = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):                
        self.t += 1
        grad = gradient(theta, **kwargs)                
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        m_hat = self.m / (1 - self.beta_one**self.t)
        self.u = np.maximum(self.beta_two * self.u, np.linalg.norm(grad,1))    
        theta = theta - (learning_rate / (1-self.beta_one**self.t)) * m_hat / self.u
        
        return theta, grad              

# --------------------------------------------------------------------------  #
class Nadam(Optimizer):
    """Nadam optimizer."""

    def __init__(self, beta_one=0.9, beta_two=0.999, epsilon=10e-8):
        self.name = "Nadam"
        self.beta_one = beta_one        
        self.beta_two = beta_two        
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)        
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad   
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)
        # Bias corrected moment estimates
        m_hat = self.m / (1 - self.beta_one**self.t)
        v_hat = self.v / (1 - self.beta_two**self.t)
     
        # Nadam update
        theta = theta - (learning_rate / (np.sqrt(np.array(v_hat,ndmin=1)) + self.epsilon)) * \
            (self.beta_one * m_hat + ((1-self.beta_one)* grad)/(1-self.beta_one**self.t))

        return theta, grad                                    

# --------------------------------------------------------------------------  #
class AMSGrad(Optimizer):
    """AMSGrad optimizer."""

    def __init__(self, beta_one=0.9, beta_two=0.999, epsilon=10e-8):
        self.name = "AMSGrad"
        self.beta_one = beta_one        
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        self.v_hat = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)        
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)
        self.v_hat = np.array(np.maximum(self.v_hat, self.v),ndmin=1)
        theta = theta - (learning_rate / (np.sqrt(np.array(self.v_hat)) + np.array(self.epsilon))) * self.m
        
        return theta, grad                        

# --------------------------------------------------------------------------  #
class AdamW(Optimizer):
    """AdamW optimizer."""

    def __init__(self, beta_one=0.9, beta_two=0.999, decay_rate=1e-4, epsilon=10e-8):
        self.name = "AdamW"
        self.beta_one = beta_one        
        self.beta_two = beta_two
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        self.v_hat = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)        
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)
        self.v_hat = np.array(np.maximum(self.v_hat, self.v))
        theta = theta - (learning_rate / (np.sqrt(np.array(self.v_hat,ndmin=1)) + np.array(self.epsilon, ndmin=1))) * \
            self.m + self.decay_rate * theta
        
        return theta, grad          

# --------------------------------------------------------------------------  #
class QHAdam(Optimizer):
    """Quasi-Hyperbolic Adam"""

    def __init__(self, beta_one=0.9, beta_two=0.999, decay_rate=1e-4, epsilon=10e-8,
                 v1=1, v2=1):
        self.name = "QH-Adam"
        self.v1 = v1
        self.v2 = v2
        self.beta_one = beta_one        
        self.beta_two = beta_two
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        self.v_hat = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)        
        self.m = self.beta_one * self.m + (1 - self.beta_one) * grad
        self.v = self.beta_two * self.v + (1 - self.beta_two) * np.square(grad)
        self.v_hat = np.maximum(self.v_hat, self.v)
        theta = theta - learning_rate * (((1-self.v1)*grad + self.v1 * self.m) /\
            (np.sqrt(np.array(1-self.v2,ndmin=1))*np.square(np.array(grad,ndmin=1))+self.v2*self.v+self.epsilon))
        
        return theta, grad                  

# --------------------------------------------------------------------------  #
class AggMo(Optimizer):
    """Aggregated Momentum."""

    def __init__(self, k=3, betas=[0, 0.9, 0.999], decay_factor=0, epsilon=10e-8):
        self.name = "AggMo"
        self.k = k
        self.betas = betas
        self.t = 0
        self.v = 0

        
    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)       
        if self.v == 0:
            self.v = {}            
            for beta in self.betas:
                self.v[beta] = np.zeros_like(theta)
        buf = np.zeros_like(theta)
        for beta in self.betas:
            self.v[beta] = beta * self.v[beta] - grad
            buf = buf + self.v[beta]
        buf = buf / self.k

        theta = theta - learning_rate * buf 
        
        return theta, grad        

# --------------------------------------------------------------------------  #
class QuasiHyperbolicMomentum(Optimizer):
    """Quasi-Hyperbolic Momentum"""

    def __init__(self, v=0.7, beta=0.999, epsilon=10e-8):
        self.name = "QH-Momentum"       
        self.v = v 
        self.beta = beta
        self.t = 0
        self.m = 0

    
    def __call__(self, gradient, learning_rate, theta, **kwargs):    
        self.t += 1
        grad = gradient(theta, **kwargs)
        self.m = self.beta * self.m + (1 - self.beta) * grad
        theta = theta - learning_rate * ((1-self.v) * grad + self.v * self.m)
        
        return theta, grad                 

         