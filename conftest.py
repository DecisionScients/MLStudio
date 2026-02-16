#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : conftest.py                                                        #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 10:02:50 pm                         #
# Last Modified : Sunday, March 15th 2020, 10:02:50 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# %%
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.datasets import make_multilabel_classification
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from mlstudio.supervised.algorithms.optimization.services.optimizers import Adagrad
from mlstudio.supervised.algorithms.optimization.services.benchmarks import StyblinskiTank
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import TimeDecay
from mlstudio.supervised.metrics.regression import MeanSquaredError
from mlstudio.utils.data_manager import StandardScaler


homedir = str(Path(__file__).parents[0])
datadir = os.path.join(homedir, "tests\\test_supervised\\test_data")
sys.path.append(homedir)
sys.path.append(datadir)
# ---------------------------------------------------------------------------- #
#                               FILES TO SKIP                                  #
# ---------------------------------------------------------------------------- #

collect_ignore_glob = ["*ions.py", "performance*","*panels.py", "*ic_regression.py",
                       "*ss_regression.py", "*metrics.py", "*_test.py",
                       "*stop.py", "*IoC.py", "*validation.py",                       
                       "*test_cross_validation.py", "test_pure*.py"]

# ---------------------------------------------------------------------------- #
#                                  DATA                                        #
# ---------------------------------------------------------------------------- #  
@fixture(scope="session")
def get_target_2d_vector():
    y = np.array([0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,0], ndmin=2).reshape(-1,1)
    return y

@fixture(scope="session")
def get_binary_target_numeric():
    y = np.array([0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,0])
    return y

@fixture(scope="session")
def get_binary_target_char():
    y = np.array(["blue","green","green","blue","green","blue","green","blue","blue","blue","green","blue","green","green","blue","green","blue","green","blue","blue"])
    return y

@fixture(scope="session")
def get_multiclass_target_num():
    y = np.array([0,1,1,0,1,0,1,2,0,0,1,0,9,1,17,1,0,1,0,0])
    return y    

@fixture(scope="session")
def get_multiclass_target_char():
    y = np.array(["blue", "red", "green","green","blue","green","blue","green","blue","blue","blue","green","blue","green","green","blue","green","blue","green","blue","blue","blue","green"])
    return y    

@fixture(scope="session")
def get_one_hot():
    y = np.array([[0,0,0,1],
             [0,0,1,0],
             [0,1,0,0],
             [1,0,0,0]])
    return y

@fixture(scope="session")
def get_multilabel_numeric():
    y = np.arange(0,100)
    y = y.reshape(20,-1)
    return y

@fixture(scope="session")
def get_multilabel_char():
    labels = np.array(["red", "green", "blue", "orange"])
    y = np.random.choice(a=labels, size=(20,5))
    return y

@fixture(scope="session")
def get_data_management_data(get_target_2d_vector,
                             get_binary_target_numeric,
                             get_binary_target_char,
                             get_multiclass_target_num,
                             get_multiclass_target_char,
                             get_one_hot,
                             get_multilabel_numeric,
                             get_multilabel_char):
    d = {}
    d['target_2d_vector'] = get_target_2d_vector
    d['binary_target_numeric'] = get_binary_target_numeric
    d['binary_target_char'] = get_binary_target_char
    d['multiclass_target_num'] = get_multiclass_target_num
    d['multiclass_target_char'] = get_multiclass_target_char
    d['one_hot'] = get_one_hot
    return d

@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler(scale_std=False)    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_regression_data_unscaled():
    X, y = datasets.load_boston(return_X_y=True)
    return X, y    

@fixture(scope="session")
def get_regression_data_features():
    data = datasets.load_boston()
    return data['feature_names']

@fixture(scope="session")
def get_regression_data_split(get_regression_data):
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test        


@fixture(scope="session")
def get_logistic_regression_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y   

@fixture(scope="session")
def get_logistic_regression_data_categorical():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    y = np.where(y==0, "good", "bad")    
    return X, y   

@fixture(scope="session")
def get_logistic_regression_data_split():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test                

@fixture(scope="session")
def get_logistic_regression_data_features():
    data = datasets.load_breast_cancer()
    return data['feature_names']

@fixture(scope="session")
def get_multiclass_data():
    X, y = datasets.load_iris(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y   

@fixture(scope="session")
def get_multiclass_data_categorical():
    X, y = datasets.load_iris(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    y = np.where(y == 0, 'class_0', y)
    y = np.where(y == '1', 'class_1', y)
    y = np.where(y == '2', 'class_2', y)
    return X, y       


@fixture(scope="session")
def get_multiclass_data_split(make_multiclass_data):
    X, y = make_multiclass_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test                

@fixture(scope="session")
def make_multiclass_data_features():
    data = datasets.load_iris()
    return data['feature_names']    

@fixture(scope="session")
def get_regression_prediction():
    X, y = make_regression()
    X, y_pred = make_regression()
    return X, y, y_pred

@fixture(scope="session")
def get_logistic_regression_prediction():
    X, y = make_classification()
    X, y_pred = make_classification()
    return X, y, y_pred    

@fixture(scope='session')
def get_log():
    filepath = "tests/test_data/test_monitor.csv"
    df = pd.read_csv(filepath)
    log = []
    for idx, row in df.iterrows():
        epoch_log = {'epoch': row['epoch'],
                    'train_cost': row['train_cost'],
                    'train_score': row['train_score'],
                    'val_cost': row['val_cost'],
                    'val_score': row['val_score'],                    
                    'gradient_norm': row['gradient_norm']                    
                    }
        log.append(epoch_log)    
    return log
# ---------------------------------------------------------------------------- #
#                              SIMULATED DATA                                  #
# ---------------------------------------------------------------------------- #
@fixture(scope="session")
def make_regression_data():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=5)    
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    return X, y

@fixture(scope="session")
def make_classification_data():
    X, y, = make_classification(n_samples=1000, n_features=10, random_state=5)        
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    return X, y

@fixture(scope="session")
def make_multiclass_data():
    X, y = make_classification(n_samples=10000,n_features=10, n_informative=3,
                n_classes=4, random_state=5)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)    
    return X, y

@fixture(scope="session")
def make_regression_data_split():
    X, y = make_regression(n_samples=10000, n_features=10, random_state=5)    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    scaler = StandardScaler()    
    X_train = scaler.fit_transform(X_train)    
    X_test = scaler.transform(X_test)    
    return X_train, X_test, y_train, y_test

@fixture(scope="session")
def make_classification_data_split():
    X, y, = make_classification(n_samples=10000, n_features=10, random_state=5)        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    scaler = StandardScaler()    
    X_train = scaler.fit_transform(X_train)    
    X_test = scaler.transform(X_test)    
    return X_train, X_test, y_train, y_test

@fixture(scope="session")
def make_multiclass_data_split():
    X, y = make_classification(n_samples=10000,n_features=10, n_informative=3,
                n_classes=4, random_state=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    scaler = StandardScaler()    
    X_train = scaler.fit_transform(X_train)    
    X_test = scaler.transform(X_test)    
    return X_train, X_test, y_train, y_test

@fixture(scope="session")
def make_regression_data_unscaled():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=5)    
    return X, y

@fixture(scope="session")
def make_classification_data_unscaled():
    X, y, = make_classification(n_samples=1000, n_features=10, random_state=5)        
    return X, y

@fixture(scope="session")
def make_multiclass_data_unscaled():
    X, y = make_classification(n_samples=1000,n_features=10, n_informative=3,
                n_classes=4, random_state=5)
    return X, y

# ---------------------------------------------------------------------------- #
#                                   STUBS                                      #
# ---------------------------------------------------------------------------- #
class MockBlackBox:
    def __init__(self):
        self.epoch_log = {}

    def on_epoch_end(self, epoch, log=None):
        log = log or {}
        for k,v in log.items():
            self.epoch_log.setdefault(k,[]).append(v)        

# ---------------------------------------------------------------------------- #  
class MockEstimator:
    """Mocks gradient descent estimator class."""
    def __init__(self, eta0=0.01, epochs=1000, objective=None, 
                 theta_init=None, optimizer=None,  
                 observers=None, verbose=False, random_state=None):

        self.eta0 = eta0
        self.epochs = epochs
        self.objective  = objective
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.observers = observers
        self.verbose = verbose
        self.random_state = random_state        
        self.blackbox = MockBlackBox()
        self._eta = eta0
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
        self._converged = x  

    # ----------------------------------------------------------------------- #
    def fit(self, X=None, y=None):    

        # Initialize observers
        for observer in self.observers:
            setattr(observer, 'model', self)
            observer.on_train_begin()
        
        for i in range(self.epochs):
            for observer in self.observers:
                observer.on_epoch_end(epoch=i, log=None)
            log = {'epoch': i, 'eta': self._eta}            
            self.get_blackbox().on_epoch_end(epoch=i,log=log)            

@fixture(scope="session")
def get_mock_estimator():
    return MockEstimator            

@fixture(scope='session')
def get_regression_estimator():
    return GDRegressor()
# ---------------------------------------------------------------------------- #
#                                   TEST PACKAGES                              #
# ---------------------------------------------------------------------------- #
def get_objective_test_package(filepath):
    # Get X data
    d = {}
    xlsx = pd.ExcelFile(filepath)
    d['X'] = pd.read_excel(xlsx, sheet_name='X', header=0, usecols="B:F").to_numpy()
    d['y'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="B").to_numpy()    
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='y_pred', header=0, usecols="B").to_numpy()   
    d['theta'] = pd.read_excel(xlsx, sheet_name='theta', header=0, usecols="B").to_numpy()   
    d['cost'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="A").values
    d['cost_l1'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="B").values
    d['cost_l2'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="C").values
    d['cost_l1_l2'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="D").values
    d['grad'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="A").to_numpy()
    d['grad_l1'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="B").to_numpy()
    d['grad_l2'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="C").to_numpy()
    d['grad_l1_l2'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="D").to_numpy()
    return d

def get_objective_test_package_cxe(filepath):
    # Get X data
    d = {}
    xlsx = pd.ExcelFile(filepath)
    d['X'] = pd.read_excel(xlsx, sheet_name='X', header=0, usecols="B:F").to_numpy()
    d['y'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="B:E").to_numpy()    
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='y_pred', header=0, usecols="B:E").to_numpy()   
    d['theta'] = pd.read_excel(xlsx, sheet_name='theta', header=0, usecols="B:E").to_numpy()   
    d['cost'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="A").values
    d['cost_l1'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="B").values
    d['cost_l2'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="C").values
    d['cost_l1_l2'] = pd.read_excel(xlsx, sheet_name='cost', header=0, usecols="D").values
    d['grad'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="A:D").to_numpy()
    d['grad_l1'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="F:I").to_numpy()
    d['grad_l2'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="K:N").to_numpy()
    d['grad_l1_l2'] = pd.read_excel(xlsx, sheet_name='gradient', header=0, usecols="P:S").to_numpy()
    return d    

def get_regularization_test_package(filepath):
    d = {}
    xlsx = pd.ExcelFile(filepath)
    d['l1_cost'] = pd.read_excel(xlsx, sheet_name='lasso', header=0, usecols="A").to_numpy()
    d['l1_grad'] = pd.read_excel(xlsx, sheet_name='lasso_grad', header=0, usecols="A").to_numpy().flatten()
    d['l2_cost'] = pd.read_excel(xlsx, sheet_name='ridge', header=0, usecols="A").to_numpy()
    d['l2_grad'] = pd.read_excel(xlsx, sheet_name='ridge_grad', header=0, usecols="A").to_numpy().flatten()
    d['l1_l2_cost'] = pd.read_excel(xlsx, sheet_name='elasticnet', header=0, usecols="A").to_numpy()
    d['l1_l2_grad'] = pd.read_excel(xlsx, sheet_name='elasticnet_grad', header=0, usecols="A").to_numpy().flatten()
    return d

@fixture(scope='session')
def get_objective_mse_package():
    filename = "test_objective_cost_functions_mse.xlsx"
    filepath = os.path.join(datadir, filename)
    return get_objective_test_package(filepath)

@fixture(scope='session')
def get_objective_xe_package():
    filename = "test_objective_cost_functions_xe.xlsx"
    filepath = os.path.join(datadir, filename)
    return get_objective_test_package(filepath)    

@fixture(scope='session')
def get_objective_cxe_package():
    filename = "test_objective_cost_functions_cxe.xlsx"
    filepath = os.path.join(datadir, filename)
    return get_objective_test_package_cxe(filepath)  

@fixture(scope='session')
def get_regularization_package():
    filename = "test_regularization.xlsx"
    filepath = os.path.join(datadir, filename)
    return get_regularization_test_package(filepath)  

@fixture(scope="session")
def get_optimization_momentum_test_package():
    d = {}
    filename = "test_optimizer_momentum.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['v_0'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="A:D").to_numpy().flatten()
    d['alpha'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="E").to_numpy().flatten()
    d['lambda'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="F").to_numpy().flatten()
    d['theta_init'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="G:J").to_numpy().flatten()
    d['theta'] = pd.read_excel(xlsx, sheet_name='results', header=0, usecols="E:H").to_numpy()
    return d

@fixture(scope="session")
def get_optimization_nesterov_test_package():
    d = {}
    filename = "test_optimizer_nesterov.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['v_0'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="A:D").to_numpy().flatten()
    d['alpha'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="E").to_numpy().flatten()
    d['lambda'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="F").to_numpy().flatten()
    d['theta_init'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="G:J").to_numpy().flatten()
    d['theta'] = pd.read_excel(xlsx, sheet_name='results', header=0, usecols="E:H").to_numpy()
    return d

@fixture(scope="session")
def get_optimization_adagrad_test_package():
    d = {}
    filename = "test_optimizer_adagrad.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['v_0'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="A:D").to_numpy().flatten()
    d['alpha'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="E").to_numpy().flatten()
    d['lambda'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="F").to_numpy().flatten()
    d['theta_init'] = pd.read_excel(xlsx, sheet_name='params', header=0, usecols="G:J").to_numpy().flatten()
    d['theta'] = pd.read_excel(xlsx, sheet_name='results', header=0, usecols="A:D").to_numpy()
    return d

@fixture(scope="session")
def get_classification_metric_test_package():
    d = {}
    filename = "test_classification_metrics.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['y'] = pd.read_excel(xlsx, sheet_name='data', header=0, usecols="A").to_numpy().flatten()
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='data', header=0, usecols="B").to_numpy().flatten()
    d['metrics'] = pd.read_excel(xlsx, sheet_name='metrics', header=0, usecols="A:B", \
        names=['metric', 'value'])    
    return d    

@fixture(scope="session")
def get_multiclass_task_package():
    d = {}
    filename = "test_task_multiclass.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['X'] = pd.read_excel(xlsx, sheet_name='X', header=0, usecols="A:E").to_numpy()
    d['theta'] = pd.read_excel(xlsx, sheet_name='theta', header=0, usecols="A:C").to_numpy()    
    d['y_prob'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="M:O").to_numpy()    
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="Q").to_numpy().flatten()    
    return d        

@fixture(scope="session")
def get_logistic_regression_task_package():
    d = {}
    filename = "test_task_logistic_regression.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['X'] = pd.read_excel(xlsx, sheet_name='X', header=0, usecols="A:E").to_numpy()
    d['theta'] = pd.read_excel(xlsx, sheet_name='theta', header=0, usecols="A").to_numpy().flatten()    
    d['y_prob'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="C").to_numpy().flatten()    
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="E").to_numpy().flatten()    
    return d            

@fixture(scope="session")
def get_regression_task_package():
    d = {}
    filename = "test_task_regression.xlsx"
    filepath = os.path.join(datadir, filename)
    xlsx = pd.ExcelFile(filepath)
    d['X'] = pd.read_excel(xlsx, sheet_name='X', header=0, usecols="A:E").to_numpy()
    d['theta'] = pd.read_excel(xlsx, sheet_name='theta', header=0, usecols="A").to_numpy().flatten()        
    d['y_pred'] = pd.read_excel(xlsx, sheet_name='y', header=0, usecols="A").to_numpy().flatten()    
    return d                