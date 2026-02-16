#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : debug.py                                                      #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, March 23rd 2020, 4:06:07 pm                         #
# Last Modified : Monday, March 23rd 2020, 4:35:57 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Call back that performs gradient checking."""
import copy
import numpy as np
import pandas as pd

from scipy.sparse import issparse
from mlstudio.supervised.algorithms.optimization.observers.base import Observer
from mlstudio.supervised.algorithms.optimization.services.loss import Loss
from mlstudio.utils.validation import is_valid_array_size

# --------------------------------------------------------------------------- #
class GradientCheck(Observer):
    """Performs gradient checking."""

    def __init__(self, iterations=50, epsilon=10e-7, verbose=False):
        super(GradientCheck, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.verbose = verbose
        self.name = "GradientCheck"
        
    def on_train_begin(self, log=None):        
        """Initializes gradient check parameters.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(GradientCheck, self).on_train_begin()        
        self._n = 0
        self._iteration = []
        self._theta = []
        self._cost = []
        self._cost_plus = []
        self._cost_minus = []
        self._gradients = []
        self._grad_norm = []
        self._grad_approx_norm = []
        self._approximations = []
        self._abs_differences = []
        self._rel_differences = []
        self._results = []    
        # Obtain a copy of the loss function and turn gradient_scaling off
        try:
            self._objective = copy.deepcopy(self.model.task.loss)
        except:
            self._objective = copy.deepcopy(self.model.objective)
        self._objective.gradient_scaling = False

    def _check_cost_functions(self, log):
        """Computes gradient and approximation for cost functions."""
        X = self.model.X_train
        y = self.model.y_train
        theta = log.get('theta')               

        cost = []
        cost_plus = []
        cost_minus = []
        grad_approx = []
        # Compute cost for current parameters
        y_pred = self.model.task.compute_output(theta, X)
        J = self._objective(theta, y, y_pred)        
        for i in np.arange(len(theta)):
            # Compute theta differentials
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] = theta_plus[i] + self.epsilon
            theta_minus[i] = theta_minus[i] - self.epsilon
            # Compute associated costs
            y_pred = self.model.task.compute_output(theta_plus, X)
            J_plus = self._objective(theta_plus, y, y_pred)
            y_pred = self.model.task.compute_output(theta_minus, X)
            J_minus = self._objective(theta_minus, y, y_pred)
            # Store costs
            cost.append(J)
            cost_plus.append(J_plus)
            cost_minus.append(J_minus)        
            # Estimate the gradient
            grad_approx_i = (J_plus - J_minus) / (2 * self.epsilon)         
            grad_approx.append(grad_approx_i)
        
        # Compute gradient via back-propagation
        y_pred = self.model.task.compute_output(theta, X)
        grad = self._objective.gradient(theta, X, y, y_pred)         
        return grad, grad_approx, cost, cost_plus, cost_minus

    def _check_benchmark_functions(self, log):
        """Computes gradient and approximation for benchmark functions."""
        theta = log.get('theta') 
        cost = None
        cost_plus = None
        cost_minus = None             
        grad_approx = []
        for i in np.arange(len(theta)):

            # Compute theta differentials
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] = theta_plus[i] + self.epsilon
            theta_minus[i] = theta_minus[i] - self.epsilon
            # Compute associated costs            
            J_plus = self._objective(theta_plus)            
            J_minus = self._objective(theta_minus)
            # Estimate the gradient
            grad_approx_i = (J_plus - J_minus) / (2 * self.epsilon)         
            grad_approx.append(grad_approx_i)
        
        # Compute gradient via back-propagation        
        grad = self._objective.gradient(theta)         
        return grad, grad_approx, cost, cost_plus, cost_minus

    def on_epoch_end(self, epoch, log=None):
        """Checks gradient each self.iterations number of iterations.

        Parameters
        ----------
        epoch : int
            The current epoch number

        log : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        """        
        if log.get('epoch') % self.iterations == 0:                                         
            if isinstance(self._objective, Loss):
                grad, grad_approx, cost, cost_plus, cost_minus = \
                    self._check_cost_functions(log)
            else:
                grad, grad_approx, cost, cost_plus, cost_minus = \
                    self._check_benchmark_functions(log)               

            grad = np.array(grad)
            grad_approx = np.array(grad_approx)

            # Check norms and bail if gradients are rediculous.
            if is_valid_array_size(grad) and is_valid_array_size(grad_approx):
                # Evaluate
                numerator = np.linalg.norm((grad-grad_approx), ord=1)
                denominator = max(np.linalg.norm((grad), ord=1), \
                                     np.linalg.norm((grad_approx),ord=1))
                difference = numerator / denominator
                result = difference < self.epsilon

                # Update results
                self._n += 1
                self._iteration.append(self._n)
                self._theta.append(log.get('theta'))
                self._cost.append(cost)
                self._cost_plus.append(cost_plus)
                self._cost_minus.append(cost_minus)
                self._gradients.append(grad)
                self._grad_norm.append(np.linalg.norm(grad, ord=1))
                self._grad_approx_norm.append(np.linalg.norm(grad_approx, ord=1))
                self._approximations.append(grad_approx)
                self._abs_differences.append(numerator)
                self._rel_differences.append(difference)                
                self._results.append(result)
            else:
                return

    def _print(self, results):
        """Prints results."""
        print("\n","* ",40*"=", " *")        
        print("Gradient Checks for {desc}".format(desc=self.model.description))
        print("  Num Failures : {failures}".format(failures=results['failures']))
        print(" Num Successes : {successes}".format(successes=results['successes']))
        print(" Pct Successes : {pct}".format(pct=results['successes']/(self._n)*100))
        print("Avg Difference : {diff}".format(diff=results['avg_difference']))        

    def _print_errors(self):
        """Prints gradient approximation errors."""
        # Skipping first iteration with randomly initialized parameters
        problems = None
        def condition(i,x): return x > self.epsilon and i > 0
        problems = [idx for idx, element in enumerate(self._rel_differences) if condition(idx, element)]
        for p in problems:
            print("\n\n  Iteration: {i} Theta: {t}".format(i=str(p), \
                t=str(self._theta[p])))
            print("\n  Iteration: {i} Cost: {c}  CostPlus: {l}   CostMinus: {m}".format(i=str(p), \
                c=str(np.mean(self._cost[p])), l=str(np.mean(self._cost_plus[p])), m=str(np.mean(self._cost_minus[p]))))
            print("\n  Iteration: {i} Gradient: {g}".format(i=str(p), \
                g=str(self._gradients[p])))
            print("\n  Iteration: {i} Approximation {a}".format(i=str(p), \
                a=str(self._approximations[p])))
            print("\n  Iteration: {i} Gradient Norm {a}".format(i=str(p), \
                a=str(self._grad_norm[p])))
            print("\n  Iteration: {i} Gradient Approximation Norm {a}".format(i=str(p), \
                a=str(self._grad_approx_norm[p])))
            print("\n  Iteration: {i} Absolute Difference {a}".format(i=str(p), \
                a=str(self._abs_differences[p])))                                                                                
            print("\n  Iteration: {i} Relative Difference {a}".format(i=str(p), \
                a=str(self._rel_differences[p])))                
        if problems:
            msg = "\nGradient check failed for " + self._objective.name 
            if hasattr(self._objective, 'regularization'):
                msg = msg + ' with ' + self._objective.regularization.name 
            raise Exception(msg)        

    def on_train_end(self, log=None):
        d = {"Iteration": self._iteration, "Theta": self._theta,
             "Absolute Difference": self._abs_differences,
             "Relative Difference": self._rel_differences,
             "Result": self._results, "Gradient": self._gradients,
             "Approximation": self._approximations}
        df = pd.DataFrame(d)
        results = {}
        results['failures'] = len(df[df['Result']== False])
        results['successes'] = len(df[df['Result']== True])
        results['avg_difference'] = df['Relative Difference'].mean(axis=0)
        if self.verbose:
            self._print(results)
        if self._n > 0:
            if results['successes'] / self._n * 100 < 100.0:
                self._print_errors()



        