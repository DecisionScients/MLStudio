#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : gradient_descent.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Saturday, June 13th 2020, 9:52:07 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
"""Gradient Descent Module"""
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
import copy
import warnings
from pathlib import Path
import site
import time
import tracemalloc
PROJECT_DIR = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_DIR)

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from tabulate import tabulate

from mlstudio.utils.data_manager import unpack_parameters
from mlstudio.utils.data_manager import batch_iterator
from mlstudio.utils import validation
# =========================================================================== #
#                              GRADIENT DESCENT                               #
# =========================================================================== #        
class GradientDescent(ABC, BaseEstimator):
    """Gradient descent abstract base class for all estimators.
    
    Performs gradient descent optimization to estimate the parameters theta
    that best fit the data.

    Parameters
    ----------
    eta0 : float
        The initial learning rate on open interval (0,1) 

    epochs : int
        The number of epochs to execute        

    batch_size : None or int (default=None) 
        The number of observations to include in each batch. This also 
        specifies the gradient descent variant according to the following:

            Batch_Size      Variant
            ----------      -----------------------
            None            Batch Gradient Descent
            1               Stochastic Gradient Descent
            Other int       Minibatch Gradient Descent        


    val_size : float in interval [0,1) (default=0.3)
        The proportion of the training set to allocate a validation set

    theta_init : array_like
        Contains the initial values for the parameters theta. Should include
        the bias parameter in addition to the feature parameters.

    optimizer : An Optimizer object or None
        The optimization algorithm to use. If None, the generic 
        GradientDescentOptimizer will be used.

    metric : a Metric object (default=None)
        Supported Metric object for estimating performance.        

    early_stop : an EarlyStop object or None (default=None)
        Class responsible for stopping the optimization process once
        training has stabilized. 

    learning_rate : LearningRateSchedule object or None (default=None)
        This optional parameter can be a supported LearningRateSchedule
        object.

    observer_list : an ObserverListobject
        Manages observers and subscriptions

    progress : Progress observer object
        Reports optimization statistics each 'verbose' epochs.

    blackblox : Blackbox Object
        Tracks training and validation metrics during optimization. 

    verbose : Bool or Int
        If False, the parameter is ignored. If an integer is provided, this 
        will be the number of epochs between progress reports.

    summary : Summary object
        Reports summary data at end of training.

    random_state : int or None
        If an int, this will be the random state used anywhere pseudo-randomization
        occurs.
    
    """

    def __init__(self, eta0=0.01, epochs=1000,  batch_size=None,  val_size=0.3, 
                 loss=None, data_processor=None, activation=None,
                 theta_init=None, optimizer=None, scorer=None, early_stop=None, 
                 learning_rate=None,  observer_list=None, progress=None,                  
                 blackbox=None, summary=None, verbose=False, random_state=None,
                 check_gradient=False, gradient_checker=None):

        self.eta0 = eta0
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.loss = loss
        self.data_processor = data_processor
        self.activation = activation
        self.theta_init = theta_init
        self.optimizer = optimizer            
        self.scorer = scorer        
        self.early_stop=early_stop            
        self.learning_rate = learning_rate
        self.observer_list = observer_list
        self.progress = progress
        self.blackbox = blackbox
        self.summary = summary
        self.verbose = verbose
        self.random_state = random_state    
        self.check_gradient = check_gradient
        self.gradient_checker = gradient_checker

    # ----------------------------------------------------------------------- #                
    @property
    def variant(self):
        """Returns the gradient descent variant based upon the batch size."""
        if self.batch_size is None:
            variant = "Batch Gradient Descent"
        elif self.batch_size == 1:
            variant = "Stochastic Gradient Descent"   
        else:
            variant = "Minibatch Gradient Descent"   
        return variant

    # ----------------------------------------------------------------------- #                
    @property
    def eta(self):
        return self._eta

    @eta.setter  
    def eta(self, x):
        self._eta = x

    @property
    def converged(self):
        return self._converged

    @converged.setter
    def converged(self, x):
        validation.validate_bool(x)
        self._converged = x      

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        self._loss = x

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        self._activation = x

    @property
    def theta(self):
        return self._theta

    @property
    def train_data_package(self):
        return self._train_data_package

    def get_blackbox(self):
        return self._blackbox

    def get_scorer(self):
        try:
            scorer = self._scorer
        except:
            scorer = self.scorer
        return scorer

    def set_scorer(self, x):
        validation.validate_scorer(self, x)
        self._scorer = x

    # ----------------------------------------------------------------------- #
    def _compile(self, log=None):
        """Makes copies of mutable parameters and makes them private members."""

        self._eta = self.learning_rate.eta0 if self.learning_rate else self.eta0 
        self._loss = copy.deepcopy(self.loss) 
        self._activation = copy.deepcopy(self.activation)
        self._data_processor = copy.deepcopy(self.data_processor)
        self._observer_list = copy.deepcopy(self.observer_list)           
        self._optimizer = copy.deepcopy(self.optimizer)
        self._scorer = copy.deepcopy(self.scorer)
        self._progress = copy.deepcopy(self.progress)
        self._summary = copy.deepcopy(self.summary) 
        self._gradient_checker = copy.deepcopy(self.gradient_checker)
        self._blackbox = copy.deepcopy(self.blackbox)
        self._tracemalloc = tracemalloc

        # Observers
        self._learning_rate = copy.deepcopy(self.learning_rate) if \
            self.learning_rate else self.learning_rate

        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop\
            else self.early_stop        

    # ----------------------------------------------------------------------- #
    def _initialize_state(self, log=None):
        """Initializes variables that represent teh state of the estimator."""
        self._epoch = 0      
        self._batch = 0 
        self._train_data_package = None
        self._theta = None
        self._gradient = None
        self._converged = False
        self._data_prepared = False
        self._performance_log = OrderedDict()
        self._profile_log = OrderedDict()
        self._timer = time
        self._epoch_log = None
        self._start_time = None
        self._end_time = None
        # Attributes
        self.n_features_in_ = None
        self.n_features_out_ = None
        self.classes_ = None
        self.n_classes_ = None
    # ----------------------------------------------------------------------- #  
    def _unpack_data(self, data):
        """Unpacks the data into attributes"""
        data_sets = {'X_train_': False, 'y_train_': False,
                     'X_val_': False, 'y_val_' : False, 
                     'X_test_': False, 'y_test_' : False}
        for k,v in data_sets.items():
            if data.get(k):
                if data[k].get('data') is not None:                    
                    data_sets[k] = True
                    setattr(self, k, data[k]['data'])                    
        
        if data.get('X_train_'):
            self.n_features_in_ = data['X_train_']['metadata']['orig']['n_features']
            self.n_features_out_ = data['X_train_']['metadata']['processed']['n_features']

        self.train_data_package_ = data

    # ----------------------------------------------------------------------- #    
    def _prepare_train_data(self, X, y=None, random_state=None):
        """Prepares training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The independent variables from the training set.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The dependent variable from the training set.

        Returns
        -------
        data : dict
            dictionary containing data and metadata    

        """
        data = self._data_processor.process_train_data(X, y, random_state)
        self._unpack_data(data)

    # ----------------------------------------------------------------------- #    
    def _prepare_train_val_data(self, X, y=None, val_size=None, random_state=None):
        """Prepares training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The independent variables from the training set.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The dependent variable from the training set.

        val_size : float in [0,1)
            The proportion of data to allocate to the validation set.

        random_state : int or None (default=None)
            Seed for pseudo-randomization

        Returns
        -------
        data : dict
            Dictionary containing data and metadata
        """

        data = self._data_processor.process_train_val_data(X, y, val_size, 
                                                            random_state)
        self._unpack_data(data) 
    # ----------------------------------------------------------------------- #    
    def _prepare_data(self, X, y=None):
        """Prepares data for training and creates data and metadata attributes."""     
        self._data_prepared = True   
        if self.val_size:
            self._prepare_train_val_data(X, y, self.val_size, self.random_state)
        else:
            self._prepare_train_data(X, y, self.random_state)

    # ----------------------------------------------------------------------- #
    def _initialize_observers(self, log=None):
        """Initialize remaining observers. Create and initialize observer list."""        
        log = log or {}        

        self._observer_list.append(self._blackbox)
        self._observer_list.append(self._summary)

        if self.verbose:
            self._observer_list.append(self._progress)

        if self._learning_rate:
            self._observer_list.append(self._learning_rate)

        if self._early_stop:
            self._observer_list.append(self._early_stop)

        if self.check_gradient:
            self._observer_list.append(self._gradient_checker)
        
        # Publish model parameters and estimator instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            
        self._observer_list.on_train_begin(log)

    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Compiles the estimator, initializes weights, observers, and state"""
        log = log or {}        
        validation.validate_estimator(self)
        self._compile(log)    
        self._initialize_state(log)
        self._prepare_data(log.get('X'), log.get('y'))
        self._initialize_observers(log)
        self._theta = self._init_weights(self.theta_init)
    # ----------------------------------------------------------------------- #
    def _on_train_end(self, log=None):
        """Finalizes training and posts model parameter attributes."""
        log = log or {}
        self.n_iter_ = self._epoch         
        self.intercept_, self.coef_ = unpack_parameters(self._theta)
        self._observer_list.on_train_end()                
    # ----------------------------------------------------------------------- #
    def _on_epoch_begin(self, log=None):
        """Initializes the epoch and notifies observers."""
        log = log or {}      
        self._epoch_log = self._performance_snapshot(log)
        self._start_time = self._timer.perf_counter()          
        self._tracemalloc.start()
        self._observer_list.on_epoch_begin(epoch=self._epoch, log=log)
    # ----------------------------------------------------------------------- #
    def _on_epoch_end(self, log=None):
        """Finalizes epoching and notifies observers."""
        log = log or {}

        self._end_time = self._timer.perf_counter()
        elapsed_time = self._end_time - self._start_time
        
        current, peak = self._tracemalloc.get_traced_memory()
        self._tracemalloc.stop()                
        
        self._epoch_log['cpu_time'] = elapsed_time
        self._epoch_log['current_memory'] = current
        self._epoch_log['peak_memory'] = peak
        
        self._observer_list.on_epoch_end(epoch=self._epoch, log=self._epoch_log)
        self._epoch += 1
    # ----------------------------------------------------------------------- #            
    def _on_batch_begin(self, log=None):
        """Initializes the batch and notifies observers."""
        log = log or {}
        self._observer_list.on_batch_begin(batch=self._batch, log=log)        
    # ----------------------------------------------------------------------- #            
    def _on_batch_end(self, log=None):
        """Wraps up the batch and notifies observers."""
        log = log or {}
        self._observer_list.on_batch_end(batch=self._batch, log=log)            
        self._batch += 1 

    # ----------------------------------------------------------------------- #            
    def _init_weights(self, theta_init=None):
        """Initializes parameters to theta_init or to random values.
        
        Parameters
        ----------
        theta_init : array-like of shape (n_features,) or (n_features, n_classes) Optional
            Optional initial values for the model parameters.

        Raises
        ------
        Exception if data has not been processed

        Returns
        ------        
        theta : array-like of shape (n_features,) or (n_features, n_classes)
        """
        if not self._data_prepared:
            raise Exception("Data must be prepared before weights are initialized.")

        if theta_init is not None:
            if theta_init.shape != (self.n_features_out_,):
                msg = "Initial parameters theta must have shape (n_features,)."
                raise ValueError(msg)
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self.n_features_out_) 
            # Set the bias initialization to zero
            theta[0] = 0
        return theta        
    # ----------------------------------------------------------------------- #            
    def _compute_output(self, theta, X):
        """Computes output of the current iteration.

        For linear regression, this is the linear combination of the inputs
        and the weights. For binary classification the output is the sigmoid
        probability of the positive class. For the multiclass case,
        the output is the softmax probabilities. 

        Parameters
        ----------
        theta : array-like (n_features,) or (n_features, n_classes)
            The model parameters at the current iteration

        X : array-like (n_samples, n_features)
            The features including a constant bias term.
        
        Returns
        -------
        y_out : float
        """

        return np.array(X.dot(theta), dtype=np.float32)

    # ----------------------------------------------------------------------- #            
    def _compute_loss(self, theta, y, y_out):
        """Computes the average loss of the model.

        Parameters
        ----------
        theta : array-like (n_features,) or (n_features, n_classes)
            The model parameters at the current iteration

        y : array-like of shape (n_samples,)
            True target values

        y_out : array-like of shape (n_samples,)
            The real-valued output of the model.

        Returns
        -------
        J : float
        """
        return self._loss.cost(theta, y, y_out)
    # ----------------------------------------------------------------------- #            
    def _gradient(self, theta, X, y, y_out):
        """Computes the gradient."""

        return self._loss.gradient(theta, X, y, y_out)        
    # ----------------------------------------------------------------------- #            
    def _performance_snapshot(self, log=None):
        """Computes loss and scores for the current set of parameters."""
        log = log or {}
        log['epoch'] = self._epoch
        log['eta'] = self._eta
        log['theta'] = self._theta

        y_out = self._compute_output(self._theta, self.X_train_)
        log['train_cost'] = self._compute_loss(self._theta, self.y_train_,
                                                    y_out)
        log['train_score'] = self.score(self.X_train_, self.y_train_)

        # Check not only val_size but also for empty validation sets 
        if self.val_size:
            if hasattr(self, 'X_val_'):
                if self.X_val_.shape[0] > 0:                
                    y_out_val = self._compute_output(self._theta, self.X_val_)
                    log['val_cost'] = self._compute_loss(self._theta, self.y_val_, y_out_val)                                
                    log['val_score'] = self.score(self.X_val_, self.y_val_)
        # Store the gradient and its magnitude
        log['gradient'] = self._gradient
        log['gradient_norm'] = None
        if self._gradient is not None:
            log['gradient_norm'] = np.linalg.norm(self._gradient) 

        return log           

    # ----------------------------------------------------------------------- #            
    def train_epoch(self):
        """Trains a single epoch."""
        self._on_epoch_begin()
        
        log = {}
        log['epoch'] = self._epoch

        for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):
            self._on_batch_begin()

            y_out = self._compute_output(self._theta, X_batch)     
            cost = self._compute_loss(self._theta, y_batch, y_out)
            # Grab theta for the batch log before it is updated
            log = {'batch': self._batch,'theta': self._theta, 
                    'train_cost': cost}
            # Update the model parameters and return gradient for monitoring purposes.
            self._theta, self._gradient = self._optimizer(gradient=self._loss.gradient, \
                learning_rate=self._eta, theta=copy.copy(self._theta),  X=X_batch, y=y_batch,\
                    y_out=y_out)                       
            
            log['gradient_norm'] = np.linalg.norm(self._gradient) 
            self._on_batch_end(log=log)           
        
        self._on_epoch_end()


    # ----------------------------------------------------------------------- #    
    def fit(self, X, y):
        """Trains model until stop condition is met.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : numpy array, shape (n_samples,)
            Target values 
        Returns
        -------
        self : returns instance of self
        """        
        log = {'X': X, 'y': y}
        self._on_train_begin(log)        

        while (self._epoch < self.epochs and not self._converged):            
            self.train_epoch()

        self._on_train_end()
        return self 
    
    # ----------------------------------------------------------------------- #    
    def _check_X(self, X, theta):
        """Checks X to ensure that it has been processed for training/prediction."""
        X = validation.check_X(X)        
        if X.shape[1] != theta.shape[0]:                
            data = self._data_processor.process_X_test_data(X)                    
            X = data['X_test_']['data']
        return X
    
    # ----------------------------------------------------------------------- #    
    def _check_y_pred(self, y_pred):
        if y_pred.ndim > 1:
            msg = self.__class__.__name__ + " doesn't support multioutput."
            warnings.warn(msg, UserWarning)        
        else:
            return y_pred        

    # ----------------------------------------------------------------------- #    
    @abstractmethod
    def predict(self, X):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        theta : array-like of shape (n_features) or (n_features, n_classes)
            The model parameters
        
        Returns
        -------
        y_pred : prediction
        """
        pass

    # ----------------------------------------------------------------------- #    
    def score(self, X, y):
        """Default behavior for scoring predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        score based upon the metric object.
        
        """                
        y_pred = self.predict(X)
        return self._scorer(y, y_pred, n_features=self.n_features_in_)               

    # ----------------------------------------------------------------------- #    
    def summarize(self):  
        """Prints and optimization report. """
        self._summary.report()      

# --------------------------------------------------------------------------- #
#                       GRADIENT DESCENT REGRESSOR                            #
# --------------------------------------------------------------------------- #
class GDRegressor(GradientDescent, RegressorMixin):
    """Gradient Descent Regressor."""

    @property
    def description(self):        
        return "Linear Regression by " + self.variant     

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_regression_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_regression_data_processor(x)
        self._data_processor = x        

    def _get_tags(self):
        tags = {}
        tags['X_types'] = ['2darray']
        tags['poor_score'] = True
        return tags    
    
    # --------------------------------------------------------------------------- #
    def predict(self, X):
        """Predicts the output class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, )
        
        """                        
        X = self._check_X(X, self._theta)
        y_pred = self._compute_output(self._theta, X)
        y_pred = self._check_y_pred(y_pred)
        return y_pred
    
    # --------------------------------------------------------------------------- #
    def predict_proba(self, theta, X):
        raise NotImplementedError("predict_proba is not implemented for the GDRegression class.")        

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT CLASSIFIER (BINARY)                    #
# --------------------------------------------------------------------------- #
class GDBinaryclass(GradientDescent, ClassifierMixin):
    """Gradient Descent Regressor."""

    @property
    def description(self):        
        return "Binary Classification by " + self.variant 

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_binaryclass_loss(x)
        self._loss = x
    
    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_binaryclass_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_binaryclass_activation(x)
        self._activation = x        
    
    # --------------------------------------------------------------------------- #        
    def _get_tags(self):
        tags = {}
        tags['binary_only'] = True
        if self.learning_rate or self.loss.regularizer:
            tags['poor_score'] = True
        return tags
        
    def _unpack_data(self, data):
        """Unpacks the data into attributes."""
        super(GDMulticlass, self)._unpack_data(data)
        self.classes_ = data['y_train_']['metadata']['orig']['classes']
        self.n_classes_ = data['y_train_']['metadata']['orig']['n_classes']                

    # --------------------------------------------------------------------------- #        
    def compute_output(self, theta, X):
        """Computes output as a probability of the positive class.

        The logit or linear combination of inputs and parameters is passed
        through a sigmoid function to create the probability of the 
        positive class.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_out
        """
        z = super(GDBinaryclass, self)._compute_output(theta, X)        
        return self._activation(z)

    # --------------------------------------------------------------------------- #      
    def _check_y(self, y):
        """Confirms y has been encoded."""
        if not validation.is_binary(y):
            data = self._data_processor.process_y_test_data(y)
            return data['y_test_']['data']
        return y

    def predict(self, X):
        """Predicts the output class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, )
        
        """                
        return self.predict(X, self._theta)

    def predict_proba(self, X, theta):
        """Predicts the probability of the positive class

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class probability
        """
        X = self._check_X(X, theta)
        y_pred = self._compute_output(theta, X)     
        y_pred = self._check_y_pred(y_pred)
        return y_pred       

    def score(self, X, y):
        """Computes scores for test data after training.

        Calls the predict function based upon whether the metric for the scorer
        takes a probability or a predicted class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        score based upon the metric object.
        
        """        
        if self._scorer.is_probability_metric:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)
        return self._scorer(y, y_pred, n_features=self.n_features_in_)
        

# --------------------------------------------------------------------------- #
#                 GRADIENT DESCENT MULTICLASS CLASSIFIER                      #
# --------------------------------------------------------------------------- #
class GDMulticlass(GradientDescent, ClassifierMixin):
    """Gradient Descent Multiclass Classifier."""

    @property
    def description(self):
        return "Multiclass Classification by " + self.variant     

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_multiclass_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_multiclass_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_multiclass_activation(x)
        self._activation = x        
    
    # --------------------------------------------------------------------------- #
    def _get_tags(self):
        return {'binary_only': True}    
    
    # --------------------------------------------------------------------------- #
    def init_weights(self, theta_init=None):
        """Initializes parameters to theta_init or to random values.
        
        Parameters
        ----------
        theta_init : array-like of shape (n_features,) or (n_features, n_classes) Optional
            Optional initial values for the model parameters.

        Raises
        ------
        Exception if data has not been processed

        Returns
        ------        
        theta : array-like of shape (n_features,) or (n_features, n_classes)
        """
        if not self._data_prepared:
            raise Exception("Data must be prepared before weights are initialized.")

        if theta_init is not None:
            assert theta_init.shape == (self.n_features_out_, self.n_classes_),\
                "Initial parameters theta must have shape (n_features,n_classes)."
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self.n_features_out_, self.n_classes_) 
            # Set the bias initialization to zero
            theta[0] = 0
        return theta 
    # --------------------------------------------------------------------------- #
    def _unpack_data(self, data):
        """Unpacks the data into attributes."""
        super(GDMulticlass, self)._unpack_data(data)
        self.classes_ = data['y_train_']['metadata']['orig']['classes']
        self.n_classes_ = data['y_train_']['metadata']['orig']['n_classes']                
    
    # --------------------------------------------------------------------------- #
    def _compute_output(self, theta, X):
        """Computes output as a vector of class probabilities.

        The unnormalized linear combination of inputs and parameters is passed
        through a softmax function to create a vector of probabilities.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_out
        """      
        z = super(GDMulticlass, self)._compute_output(theta, X)
        return self._activation(z)     
    # --------------------------------------------------------------------------- #
    def _check_y(self, y):
        """Confirms y has been one-hot encoded."""
        if not validation.is_one_hot(y):
            data = self._data_processor.process_y_test_data(y)
            y = data['y_test_']['data'] 
        return y
    
    # --------------------------------------------------------------------------- #
    def predict(self, X):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data        

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class
        """
                   
        o = self.predict_proba(X, self._theta)
        return o.argmax(axis=1)

    # --------------------------------------------------------------------------- #
    def predict_proba(self, X):
        """Predicts the probability of the positive class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, )
        
        """        
        X = self._check_X(X, theta)
        return self._compute_output(theta, X)  

    def score(self, X, y):
        """Computes scores for test data after training.

        Calls the predict function based upon whether the metric for the scorer
        takes a probability or a predicted class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) or (n_samples, n_classes) 
            The target variable.
        
        Returns
        -------
        score based upon the metric object.
        
        """        
        if self._scorer.is_probability_metric:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)
        return self._scorer(y, y_pred, n_features=self.n_features_in_)

# =========================================================================== #
#                    GRADIENT DESCENT PURE OPTIMIZER                          #
# =========================================================================== #
class GD(BaseEstimator):
    """Performs pure optimization of an objective function."""

    def __init__(self, eta0=0.01, epochs=1000, theta_init=None,
                 objective=None,  optimizer=None,  learning_rate=None,
                 blackbox=None, verbose=False, random_state=None):

        self.eta0 = eta0
        self.learning_rate=learning_rate
        self.epochs = epochs
        self.objective = objective
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.verbose = verbose
        self.random_state = random_state               

    # ----------------------------------------------------------------------- #
    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            if self.theta_init.shape[0] != 2:
                raise ValueError("Parameters theta must have shape (2,)")
            else:
                self._theta = self.theta_init
        else:            
            rng = np.random.RandomState(self.random_state)         
            self._theta = rng.randn(2)    

    # ----------------------------------------------------------------------- #            
    def fit(self, X=None, y=None):
        """Performs the optimization of the objective function..
        
        Parameters
        ----------
        objective : object derived from Objective class
            The objective function to be optimized

        Returns
        -------
        self
        """
        
        self._on_train_begin()

        while (self._epoch < self.epochs and not self._converged):

            self._on_epoch_begin()

            cost = self._objective(self._theta)

            self._theta, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

            self._on_epoch_end()

        self._on_train_end()
        return self   


    def predict(self, X):
        """Predicts output as linear combination of inputs and weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, )
        
        """                        
        return self._objective(X, self._theta)        
# %%
