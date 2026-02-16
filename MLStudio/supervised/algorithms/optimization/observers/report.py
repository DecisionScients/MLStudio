# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \report.py                                                        #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, May 15th 2020, 9:16:41 pm                           #
# Last Modified : Tuesday, July 28th 2020, 9:09:28 pm                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Module containing observers that monitor and report on optimization."""
from abc import ABC, abstractmethod
from collections import OrderedDict, ChainMap
import datetime
import itertools
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

from mlstudio.supervised.algorithms.optimization.observers import base, debug
from mlstudio.supervised.metrics.base import BaseMetric, BaseMetric
from mlstudio.supervised.algorithms.optimization.observers.history import BlackBox
from mlstudio.data_services.preprocessing import RegressionDataProcessor
from mlstudio.data_services.preprocessing import BinaryClassDataProcessor
from mlstudio.data_services.preprocessing import MultiClassDataProcessor
from mlstudio.utils.data_manager import AddBiasTerm, LabelEncoder, OneHotLabelEncoder
from mlstudio.utils.data_manager import DataSplitter
from mlstudio.utils.format import proper
from mlstudio.utils.print import Printer

# --------------------------------------------------------------------------- #
#                                PROGRESS                                     #
# --------------------------------------------------------------------------- #              
class Progress(base.Observer):
    """Class that reports progress at designated points during training."""

    def __init__(self):
        super(Progress, self).__init__()
        self.name = "Progress"    
    
    def on_epoch_begin(self, epoch, log=None):
        """Reports progress at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch

        log : Dict
            Statistics obtained at end of epoch
        """
        if self.model.verbose:
            if not isinstance(self.model.verbose, int):
                raise TypeError("Verbose must be False or an integer. The \
                    integer indicates the number of epochs between each \
                    progress update.")            
            else:
                if epoch % self.model.verbose == 0:
                    items_to_report = ('epoch', 'train', 'val')
                    log = {k:v for k,v in log.items() if k.startswith(items_to_report)}
                    progress = "".join(str(key) + ': ' + str(np.round(value,4)) + ' ' \
                        for key, value in log.items())
                    print(progress)

# --------------------------------------------------------------------------- #
#                                SUMMARY                                      #
# --------------------------------------------------------------------------- #           
class Summary(base.Observer):
    """Optimization summary class."""

    def __init__(self, printer=None):
        super(Summary, self).__init__()
        self.printer = printer
        self.name = "Summary"

    def _optimization_summary(self):
        """Prints optimization summary information."""
        
        bb = self.model.get_blackbox()

        optimization_summary = {'Name': self.model.description,
                                'Start': str(bb.start),
                                'End': str(bb.end),
                                'Duration': str(bb.duration) + " seconds.",
                                'Epochs': str(bb.total_epochs),
                                'Batches': str(bb.total_batches)}
        self.printer.print_dictionary(optimization_summary, "Optimization Summary")             

    def _data_summary(self):
        """Prints summary of the data used for training (and validation)."""

        X_train_metadata = self.model.train_data_package_['X_train_']['metadata']['processed']                
        y_train_metadata = self.model.train_data_package_['y_train_']['metadata']['processed']
        
        if self.model.train_data_package_.get('X_val_'):       
            X_val_metadata = self.model.train_data_package_['X_val_']['metadata']['processed']            
            y_val_metadata = self.model.train_data_package_['y_val_']['metadata']['processed']

            headers = ["n_Observations", "n_Features", "Size (Bytes)"]
            df_X = {'X_train_': X_train_metadata, "X_val_": X_val_metadata}
            self.printer.print_title("Training and Validation Input Data (X)")
            df_X = pd.DataFrame(data=df_X)
            print(tabulate(df_X.T, headers, tablefmt="simple"))

            headers = ["n_Observations", "Data Type","Data Class", "n_Classes", "Size (Bytes)"]
            df_y = {'y_train_': y_train_metadata, "y_val_": y_val_metadata}
            self.printer.print_title("Training and Validation Target Data (y)")
            df_y = pd.DataFrame(data=df_y)
            print(tabulate(df_y.T, headers, tablefmt="simple"))
        else:
            self.printer.print_title("Training Set (X)")
            self.printer.print_dictionary(X_train_metadata)
            self.printer.print_title("Training Set (y)")
            self.printer.print_dictionary(y_train_metadata)                        
            
    def _resource_summary(self):
        """Produces data on memory and CPU consumption."""
        bb = self.model.get_blackbox()
        log = bb.epoch_log
        d = OrderedDict()
        d['Total CPU Time (s)'] = round(np.sum(log['cpu_time']), 4)
        d['Average Peak Memory (bytes)'] = round(np.mean(log['peak_memory']), 4)
        d['Average Current Memory (bytes)'] = round(np.mean(log['current_memory']), 4)
        self.printer.print_dictionary(d, "Resource Consumption")

    def _performance_summary(self):        
        """Renders early stop optimization data."""

        bb = self.model.get_blackbox()
        log = bb.epoch_log
        datasets = {'train': 'Training', 'val': 'Validation'}
        keys = ['train', 'val']
        metrics = ['cost', 'score']
        print_data = []
        # Format labels and data for printing from result parameter
        for performance in list(itertools.product(keys, metrics)):
            d = {}
            key = performance[0] + '_' + performance[1]
            if log.get(key):
                label = datasets[performance[0]] + ' ' + proper(performance[1]) 
                d['label'] = label
                if performance[1] == 'score':                    
                    d['data'] = str(np.round(log.get(key)[-1],4)) + " " + self.model.get_scorer().label
                else:
                    d['data'] = str(np.round(log.get(key)[-1],4)) 
                print_data.append(d) 

        # Print performance statistics
        performance_summary = OrderedDict()
        for i in range(len(print_data)):
            performance_summary[print_data[i]['label']] = print_data[i]['data']
        performance_summary['Max Gradient Norm'] = \
            np.round(np.nanmax(np.array(log.get('gradient_norm'),dtype=np.float64)),4)
        performance_summary['Min Gradient Norm'] = \
            np.round(np.nanmin(np.array(log.get('gradient_norm'),dtype=np.float64)),4)
        title = "Performance Summary"
        self.printer.print_dictionary(performance_summary, title)    
        
    _hdf = pd.DataFrame()
    _posted = [] 
    def _update_params(self, level, k, v):
        """Adds a k,v pair of params to the hyperparameter dataframe."""          
    
        if v is not None:
            obj = k.split("__")[0]                        
            d = {'level': level, 'object':obj, 'Hyperparameter': k, 'Value': v}
            df = pd.DataFrame(data=d, index=[0])
            self._hdf = pd.concat([self._hdf, df])        
        else:
            obj = k.split("__")[0]                        
            d = {'level': level, 'object':obj, 'Hyperparameter': k, 'Value': None}
            df = pd.DataFrame(data=d, index=[0])
            self._hdf = pd.concat([self._hdf, df])            
            


    def _get_params(self, obj):
        """Gets the hyperparameters for an object."""        
        self._implicit_dependencies = (debug.GradientCheck, BlackBox, Printer, 
                              Progress, base.ObserverList, BaseMetric, base.PerformanceObserver,
                              AddBiasTerm, OneHotLabelEncoder, LabelEncoder,
                              DataSplitter, RegressionDataProcessor, MultiClassDataProcessor,
                              BinaryClassDataProcessor, debug.GradientCheck,
                              BaseMetric, self.__class__)        
        object_name = obj.__class__.__name__        
        params = obj.get_params()
        for k, v in params.items():
            if k not in self._posted:
                if isinstance(v, (str, dict, bool, int, float, np.ndarray, \
                    np.generic, list)) or v is None:                                
                    level = k.count("__")
                    k = object_name + "__" + k                                
                    self._update_params(level, k, v)                        
                    self._posted.append(k)
                else:
                    if not isinstance(v, self._implicit_dependencies):                    
                        self._update_params(level=0, k=" ", v=" ")                        
                        k = v.__class__.__name__
                        level = k.count("__")
                        self._update_params(level, k, v=" ")                        
                        self._get_params(v)        
                        self._posted.append(k)

    def _hyperparameter_summary(self):        
        """Displays model hyperparameters."""        
        
        #self.printer.print_title("Model Hyperparameters")
        self._get_params(self.model)
        hdf = self._hdf.loc[self._hdf['level'] == 0][['Hyperparameter', 'Value']]
        hd = {}
        for idx, row in hdf.iterrows():
            if row['Hyperparameter'] == " ":
                pass
            else:
                hd[row['Hyperparameter']] = row['Value']
        self.printer.print_dictionary(hd, "Model Hyperparameters")    

    def report(self):
          
        self._optimization_summary()
        self._resource_summary()
        self._data_summary()
        self._performance_summary()
        self._hyperparameter_summary()
