# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \component.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Sunday, July 12th 2020, 5:17:47 am                          #
# Last Modified : Sunday, July 12th 2020, 5:17:47 am                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Domain classes for the Model Selection and Evaluation Modules"""
from abc import ABC, abstractmethod
import uuid
# --------------------------------------------------------------------------- #
class BaseModel(ABC):
    """Abstract base class machine learning models.

    Models are mathematical representation of real-world processes. Distinct
    from a machine learning algorithm, machine learning models are specific
    incantations of a process and represents phenomena.

    The lifecycle of a model begins when it 'wins' a model selection 
    search. GridSearchCV, RandomSearchCV are two cross validation processes
    that can produce a model instance.
    
    Parameters
    ----------
    selection : A Scikit-Learn *SearchCV object.
        A random or grid search cross-validation object.

    name : str
        Suggested a 35 character name for the object.

    desc : str (default=name of Estimator object.)
        Up to 80 character description of the object. 

    model_type : str
        The model type: This pertains to the machine learning task for which
        the Model was created, for instance, 'regression'.        

    filepath : str
        The relative filename for object persistence.  

    precedent : object
        The Model object from which this model was derived.

    """    

    def __init__(self, selection, name, desc, model_type, filepath, precedent=None):       
        
        self.id = uuid()
        self.name = name
        self.desc = desc
        self.created = datetime.datetime.now()
        self.updated = datetime.datetime.now()
        self.filepath = filepath
        self.precedent = precedent
        self._selection = {}


    @property
    def algorithm_name(self):
        return self._selection.cv_results_.best_estimator_.name        

    @property
    def parameters(self):
        return self._selection.cv_results_.best_estimator_.theta_

    @property
    def hyperparameters(self):
        return self._selection.cv_results_.best_estimator_.get_params()

    @property
    def best_score(self):
        return self._selection.cv_results_.best_score_
    
    @property
    def mean_test_score(self):
        return self._selection.cv_results_

    @property
    def mean_train_score(self):
        return self.method.cv_results_.mean_train_score        

    def fit(self, X, y):
        """Fits the model on the data."""
        self.estimator.fit(X, y)

    def predict(self, X):
        """Computes predictions."""
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

# --------------------------------------------------------------------------- #
class SModel(BaseModel):
    """Supervised Learning Model."""

    def __init__(self, name, desc, model_type, filepath=None, precedent=None):       
        super(SModel, self).__init__(name=name,
                                     desc=desc, 
                                     model_type=model_type,                                      
                                     filepath=filepath,
                                     precedent=precedent)


# --------------------------------------------------------------------------- #
class UModel(BaseModel):
    """Unsupervised Learning Model."""

    def __init__(self, name, desc, model_type, filepath=None, precedent=None):       
        super(UModel, self).__init__(name=name,
                                     desc=desc, 
                                     model_type=model_type,                                      
                                     filepath=filepath,
                                     precedent=precedent)

# --------------------------------------------------------------------------- #
class TModel(BaseModel):
    """Transfer Learning Model."""

    def __init__(self, name, desc, model_type, filepath=None, precedent=None):       
        super(TModel, self).__init__(name=name,
                                     desc=desc, 
                                     model_type=model_type,                                      
                                     filepath=filepath,
                                     precedent=precedent)
# --------------------------------------------------------------------------- #
class RModel(BaseModel):
    """Reinforcement Learning Model."""

    def __init__(self, name, desc, model_type, filepath=None, precedent=None):       
        super(TModel, self).__init__(name=name,
                                     desc=desc, 
                                     model_type=model_type,                                      
                                     filepath=filepath,
                                     precedent=precedent)
