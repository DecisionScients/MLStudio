# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \model_selection.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, June 30th 2020, 7:20:33 pm                         #
# Last Modified : Wednesday, July 1st 2020, 12:38:36 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Model Development classes"""
#%%
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
import persistent

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, learning_curve, validation_curve
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_array

from mlstudio.utils.format import proper
from mlstudio.utils.data_analyzer import one_sample_ttest, critical_values
# --------------------------------------------------------------------------- #
class ModelBuilder(BaseEstimator, persistent.Persistent):
    """Builds model via nested cross-validation.

    This class performs nested cross validation to evaluate the model 
    selection process. It is comprised of an inner loop for model selection,
    and an outer loop for model selection process evaluation. The output
    of the nested cross-validation is an unbiased estimate of generalization
    performance of the model selection process. 

    Optionally, the model selection process (inner loop) is performed
    on the full training set to select the final model.
    
    Parameters
    ----------
    estimator : Scikit-Learn compatible estimator object
        The algorithm to be evaluated

    parameters : list of dictionaries
        List of one or more dictionaries. Each dictionary represents a 
        parameter set. A parameter set is a collection of hyperparameters 
        and an iterable containing the values to be searched together. 
        Note that a hyperparameter may be an object with its own hyperparameters. 
        In such cases, we use the <object>__<parameter> syntax.

    outer_cv : Scikit-Learn Cross Validation object (default=KFold(n_splits=5))
        The object that controls the outer cross-validation loop 

    inner_cv : Scikit-Learn Cross Validation object (default=KFold(n_splits=5))
        The object that controls the inner cross-validation loop 
    
    scoring : str, callable, list/tuple, or dict, (default=None)
        A single str (see `The scoring parameter: defining model 
        evaluation rules <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) 
        or a callable (see `Defining your scoring strategy from metric 
        functions <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) 
        to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) 
        strings or a dict with names as keys and callables as values.
        
        If None, the estimator’s score method is used.        

    pac_failure_prob : float in (0,1) (default = 0.05)
        The Probably Approximately Correct (PAC) failure probability parameter. The
        maximum allowable probability that the dataset upon which the models
        are trained is unrepresentative of the true data giving probability
        distribution.

    pac_risk_approximation_tol : float (0,1) (default=0.1) 
        The risk approximation tolerance parameter. Given a data set D, 
        drawn identically and independently from a data generating probability 
        distribution P, a learning algorithm A, a learning model is PAC 
        learnable if for a dataset of size >=m, there is a (1-delta) 
        probability that the empirical risk R is greater than epsilon. 

    test_size : float in (0,1) (default=0.3)
        The proportion of the data to held out for testing the final model. 
    
    stratify : array-like (default=None)
        Required for classification data. If not None, then this is the 
        array of class labels.

    scaler : Boolean or Scikit-Learn scaler object (default=StandardScaler())
        If False, no scaling will be applied to the data. If True, the data
        will be standardized with zero mean and unit variance.

    n_jobs : int, (default=None)    
        Number of jobs to run in parallel. None means 1 unless unless in a 
        joblib.parallel_backend context. -1 means using all processors. 

    pre_dispatch : int, or str, (default=n_jobs)
        Controls the number of jobs that get dispatched during parallel execution. 
        Reducing this number can be useful to avoid an explosion of memory consumption 
        when more jobs get dispatched than CPUs can process. This parameter can be:

            None, in which case all the jobs are immediately created and spawned. 
                Use this for lightweight and fast-running jobs, to avoid delays due 
                to on-demand spawning of the jobs

            An int, giving the exact number of total jobs that are spawned

            A str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’        

    return_train_score : bool (default=True)
        Whether to include train scores in the results.
    
    refit : bool (default=True)
        Refit an estimator using the best found parameters on the whole dataset.

    random_state : int (default=None)
        Controls pseudo-randomization and reproducibility.

    Attributes
    ----------
    performance_estimate_ : dict of float arrays of shape (n_splits,)
        A dict containing test scores, train_scores, fit_times, and score_times.

    gridsearches_ : GridSearchCV objects
        One GridSearchCV object for each outer fold.       

    cv_results_ : dict
        Results of cross validation on the entire training set, including:
            * hyperparameter grid
            * train/test scores / fit times
            * mean and standard deviation of train/test scores and fit times.
            * the best estimator     
            * the best hyperparameters  
            * best score   
            * best index   
            * metric (function or dict)
            * n_splits    
            * refit time

    final_test_score_ : float
        The score obtained on the original hold-out test/validation set. 

    final_model_ : estimator
        Best model trained on all available dataset.  

    input_size_ : int
        The size of entire data set

    train_set_size_ : int
        The size of the training set

    test_set_size_ : int
        The size of the test set

    pac_generalization_bound : float
        The upper bound on the risk of any hypothesis computed on a sample S drawn
        from the data generating probability distribution P, for a given
        pac_risk_approximation_tol and pac_failure_prob > 0

    pac_sample_complexity_ : int
        The minimum required data set size required to conclude that the learning
        models achieve a risk <= epsilon with a probability of (1-delta).          

    n_splits_outer_cv : int
        The number of splits in the outer cross validation loop

    n_splits_inner_cv : int
        The number of splits in the inner cross validation loop        
    """    

    def __init__(self, estimator, parameters, 
                 outer_cv=KFold(n_splits=5, shuffle=True), 
                 inner_cv=KFold(n_splits=5, shuffle=True), 
                 scoring=None, delta=0.05, epsilon=0.05, test_size=0.3,
                 stratify=None, scaler=StandardScaler(), n_jobs=None,
                 pre_dispatch='n_jobs', return_train_score=True,
                 refit=True, random_state=None):

        self.estimator = estimator
        self.parameters = parameters
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.delta = delta
        self.epsilon = epsilon
        self.test_size = test_size
        self.stratify = stratify
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self.refit = refit
        self.random_state = random_state

    def _prepare_data(self, X, y):
        """Create training and test data."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(\
            X, y, test_size=self.test_size, stratify=self.stratify,
            random_state=self.random_state)

    def _get_scaler(self):
        """Obtains the scaler object based upon the scaler parameter value."""
        if isinstance(self.scaler, TransformerMixin):
            scaler = deepcopy(self.scaler)
        elif self.scaler:
            scaler = StandardScaler()
        else:
            scaler = None
        return scaler


    def _build_pipeline(self):
        """Creates pipeline object"""        
        scaler = self._get_scaler()
        steps = []
        if scaler:
            steps.append(('std', scaler))
        steps.append(('est', self.estimator))        
        self._pipeline = Pipeline(steps)

    def _format_param_grid(self):
        """Formats param_grid keys in <estimator>__<param> syntax."""
        if isinstance(self.parameters, list):
            self._parameters = []
            for params in self.parameters:
                d = OrderedDict()
                for k, v in params.items():
                    k = "est__" + k
                    d[k] = v
                self._parameters.append(d)
        elif isinstance(self.parameters, dict):
            self._parameters = OrderedDict()
            for k, v in params.items():
                k = "est__" + k
                self._parameters[k] = v                       

    def _build_gridsearch_object(self):
        """Builds a gridsearch object for inner cross-validation."""
        self._gridsearch_object =  \
            GridSearchCV(estimator=self._pipeline,
                         param_grid=self._parameters,
                         scoring=self.scoring,
                         n_jobs=self.n_jobs,
                         cv=self.inner_cv)

    def _evaluate_model_generalization(self):
        """Performs nested cross validation to estimate model building generalization."""        

        cv_results = cross_validate(estimator=self._gridsearch_object,
                                      X=self.X_train_,
                                      y=self.y_train_,
                                      scoring=self.scoring,
                                      cv=self.outer_cv,
                                      error_score='raise',
                                      return_estimator=True,
                                      return_train_score=self.return_train_score,
                                      n_jobs=self.n_jobs)
        
        self.results_ = cv_results        

    def fit(self, X, y):
        """Performs nested cross-validation and final model selection.
        
        Estimate model performance using nested cross-validation and
        select the best model using cross-validation on all available data
        then refit the model with best parameters on the full dataset. 
        """ 

        self._prepare_data(X, y)
        self._build_pipeline()
        self._format_param_grid()
        self._build_gridsearch_object()
        self._evaluate_model_generalization()
         
        
#%%
