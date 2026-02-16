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
import datetime
from pathlib import Path
import persistent
from sklearn.datasets import load_boston
from uuid import uuid4
homedir = str(Path(__file__).parents[2])
sys.path.append(homedir)

from mlstudio.supervised.machine_learning.gradient_descent import GDRegressor
from mlstudio.supervised.model.centre import SModel
from mlstudio.supervised.model.development import ModelBuilder
from mlstudio.supervised.model.data import DataFactory
# --------------------------------------------------------------------------- #
class ModelEvaluation(persistent.Persistent):
    """Contains generalization error statistics for a NestedCV Search."""
    def __init__(self, typ):
        
        self._id = uuid4()
        self._name = "John"
        self._desc = "Model Selection Demo"                       
        self._created = datetime.datetime.now()
        self._updated = datetime.datetime.now()
        self._filename = "model_selection_demo_" + str(self._created) + ".fs"
        self._data_object = DataFactory()(typ)
        self._eval_data = {}
    
    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._desc            

    @property
    def filename(self):
        return self._filename

    def fit(self):
        """Fits the nested CV model."""
        X, y = load_boston(return_X_y=True)
        self._estimator = GDRegressor()
        param_set = [
            {"epochs": [100,200,500,1000],
             "batch_size": [32, 64]}
        ]
        self._estimator = ModelBuilder(estimator=self._estimator, parameters=param_set) 
        self._estimator.fit(X,y)

    def inner_loop(self, gscv):
        for k, v in gscv.cv_results_.items():

        name = k
        desc = "Model evaluation " + k
        filename = "model_evaluation_demo_2020-07-21_" + k + '_.fs'
        self._data_object(name=name, desc=desc, filename=filename)

    def extract(self):
        """Extracts results from nested CV."""
        #db = ZODB.config.databaseFromString('mlstudio.fs')
        
        for k, v in self._estimator.results_.items():
            name = k
            desc = "Model evaluation " + k
            filename = "model_evaluation_demo_2020-07-21_" + k + '_.fs'
            self._data_object(name=name, desc=desc, filename=filename)
            self._data_object.value = v
            self._eval_data[k] = self._data_object
            if not isinstance(v, list):
                self.inner_loop(v)

                
                

            # with db.transaction() as connection:
            #     connection.root.foo = 1            
        print(self._eval_data)

ms = ModelEvaluation(typ='stat')
ms.fit()
ms.extract()
#%%
            