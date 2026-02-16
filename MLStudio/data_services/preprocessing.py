# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \preprocessing.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 3:17:33 pm                        #
# Last Modified : Saturday, July 25th 2020, 3:17:34 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Data preprocessing for machine learning and model development."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy

from sklearn.base import BaseEstimator
from mlstudio.utils.data_analyzer import get_feature_info, get_target_info
# --------------------------------------------------------------------------- #
class AbstractDataProcessor(ABC, BaseEstimator):
    """Defines interface for DataProcessor classes."""

    def __init__(self, add_bias_transformer=None, 
                 split_transformer=None, label_encoder=None, 
                 one_hot_label_encoder=None):
        
        self.add_bias_transformer = add_bias_transformer  
        self.split_transformer = split_transformer
        self.label_encoder = label_encoder
        self.one_hot_label_encoder = one_hot_label_encoder

    def _compile(self):
        """Compiles the class for scikit-learn compatibility."""
        self.add_bias_transformer = deepcopy(self.add_bias_transformer)
        self.split_transformer = deepcopy(self.split_transformer)
        self.label_encoder = deepcopy(self.label_encoder)
        self.one_hot_label_encoder = deepcopy(self.one_hot_label_encoder)

    def _transform_X(self, X):
        """Adds bias term to X."""
        return self.add_bias_transformer.fit_transform(X)

    def _transform_X_ysplit(self, X, y, val_size=None, stratify=False, 
                                random_state=None):
        """Splits the data."""        
        X_train, X_val, y_train, y_val = self.split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=stratify, 
            random_state=random_state)        
        return X_train, X_val, y_train, y_val
    
    def process_X_test_data(self, X, y=None):     
        data = OrderedDict()
        data['X_test_'] = OrderedDict()  
        data['X_test_']['metadata'] = OrderedDict()

        data['X_test_']['metadata']['orig'] = get_feature_info(X)

        X = self._transform_X(X)
        
        data['X_test_']['data'] = X
        data['X_test_']['metadata']['processed'] = get_feature_info(X)               
        return data
    
    @abstractmethod
    def process_y_test_data(self, y):
        """Default behavior for processing y_test metadata."""
        # Default behavior does nothing to y variable.
        data = OrderedDict()
        data['y_test_'] = OrderedDict()
        data['y_test_']['metadata'] = OrderedDict()
        data['y_test_']['metadata']['orig'] = get_target_info(y)        
        data['y_test_']['data'] = y
        data['y_test_']['metadata']['processed'] = get_target_info(y)        
        
        return data
        

    def process_train_data(self, X, y=None):
        """Default behavior for processing training metadata."""
        # Default behavior adds bias to X, does nothing to y
        # Copies mutable parameters to private variables for scikit-learn
        self._compile()

        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()
        data['X_train_']['metadata']['orig'] = get_feature_info(X)        
        data['y_train_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)        
        
        data['X_train_']['data'] = X
        data['X_train_']['metadata']['processed'] = get_feature_info(X)
        data['y_train_']['metadata']['processed'] = get_target_info(y)
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        """Default behavior for processing training and validation metadata."""
        # Copies mutable parameters to private variables for scikit-learn        
        self._compile()

        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_val_'] = OrderedDict()
        data['y_val_'] = OrderedDict()        
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()        
        data['X_val_']['metadata'] = OrderedDict()
        data['y_val_']['metadata'] = OrderedDict()                

        data['X_train_']['metadata']['orig'] = get_feature_info(X)
        data['y_train_']['metadata']['orig'] = get_target_info(y)
        data['X_val_']['metadata']['orig'] = get_feature_info(X)
        data['y_val_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_ysplit(X=X, y=y, val_size=val_size, 
                                      stratify=False, random_state=random_state)

        data['X_train_']['data'] = X_train_xform
        data['y_train_']['data'] = y_train_xform
        data['X_train_']['metadata']['processed'] = get_feature_info(X_train_xform)
        data['y_train_']['metadata']['processed'] = get_target_info(y_train_xform)            

        # Confirm split was successful
        if X_val_xform is not None:
            data['X_val_']['data'] = X_val_xform
            data['y_val_']['data'] = y_val_xform
            data['X_val_']['metadata']['processed'] = get_feature_info(X_val_xform)            
            data['y_val_']['metadata']['processed'] = get_target_info(y_val_xform)                                
        return data
    
# --------------------------------------------------------------------------- #
class RegressionDataProcessor(AbstractDataProcessor):
    """Performs preprocessing of metadata for regression."""

    def process_train_data(self, X, y=None):
        return super(RegressionDataProcessor, self).process_train_data(X, y)

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        return super(RegressionDataProcessor, self).process_train_val_data(X, y, \
            val_size, random_state)

    def process_X_test_data(self, X, y=None):     
        return super(RegressionDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        return super(RegressionDataProcessor, self).process_y_test_data(y)

        
# --------------------------------------------------------------------------- #
class BinaryClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):

        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()
                
        data['X_train_']['metadata']['orig'] = get_feature_info(X)
        data['y_train_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)        
        y = self.label_encoder.fit_transform(y)
        
        data['X_train_']['data'] = X
        data['y_train_']['data'] = y

        data['X_train_']['metadata']['processed'] = get_feature_info(X)
        data['y_train_']['metadata']['processed'] = get_target_info(y)
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):

        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_val_'] = OrderedDict()
        data['y_val_'] = OrderedDict()        
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()        
        data['X_val_']['metadata'] = OrderedDict()
        data['y_val_']['metadata'] = OrderedDict()      

        data['X_train_']['metadata']['orig'] = get_feature_info(X)
        data['y_train_']['metadata']['orig'] = get_target_info(y)
        data['X_val_']['metadata']['orig'] = get_feature_info(X)
        data['y_val_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)        

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_ysplit(X=X, y=y, val_size=val_size, 
                                     stratify=True, random_state=random_state)

        y_train_xform = self.label_encoder.fit_transform(y_train_xform)
        
        data['X_train_']['data'] = X_train_xform
        data['y_train_']['data'] = y_train_xform
        data['X_train_']['metadata']['processed'] = get_feature_info(X_train_xform)        
        data['y_train_']['metadata']['processed'] = get_target_info(y_train_xform)


        # Check if split was successful
        if X_val_xform is not None:
            data['X_val_']['data'] = X_val_xform
            data['y_val_']['data'] = y_val_xform
            data['X_val_']['metadata']['processed'] = get_feature_info(X_val_xform)        
            data['y_val_']['metadata']['processed'] = get_target_info(y_val_xform)        
        return data           

    def process_X_test_data(self, X, y=None):     
        return super(BinaryClassDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        data = OrderedDict()
        data['y_test_'] = OrderedDict()
        data['y_test_']['metadata'] = OrderedDict()
        data['y_test_']['metadata']['orig'] = get_target_info(y)        
        data['y_test_']['data'] = self.label_encoder.transform(y)
        data['y_test_']['metadata']['processed'] = get_target_info(y)          
        return data

        
# --------------------------------------------------------------------------- #
class MultiClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):


        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()
                
        data['X_train_']['metadata']['orig'] = get_feature_info(X)
        data['y_train_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)        
        y = self.label_encoder.fit_transform(y)
        y = self.one_hot_label_encoder.fit_transform(y)
        
        data['X_train_']['data'] = X
        data['y_train_']['data'] = y

        data['X_train_']['metadata']['processed'] = get_feature_info(X)
        data['y_train_']['metadata']['processed'] = get_target_info(y)
        
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):

        data = OrderedDict()
        data['X_train_'] = OrderedDict()
        data['y_train_'] = OrderedDict()
        data['X_val_'] = OrderedDict()
        data['y_val_'] = OrderedDict()        
        data['X_train_']['metadata'] = OrderedDict()
        data['y_train_']['metadata'] = OrderedDict()        
        data['X_val_']['metadata'] = OrderedDict()
        data['y_val_']['metadata'] = OrderedDict()      

        data['X_train_']['metadata']['orig'] = get_feature_info(X)
        data['y_train_']['metadata']['orig'] = get_target_info(y)
        data['X_val_']['metadata']['orig'] = get_feature_info(X)
        data['y_val_']['metadata']['orig'] = get_target_info(y)

        X = self._transform_X(X)        

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_ysplit(X=X, y=y, val_size=val_size, 
                                     stratify=True, random_state=random_state)

        y_train_xform = self.label_encoder.fit_transform(y_train_xform)
        y_train_xform = self.one_hot_label_encoder.fit_transform(y_train_xform)
        
        data['X_train_']['data'] = X_train_xform
        data['y_train_']['data'] = y_train_xform
        data['X_train_']['metadata']['processed'] = get_feature_info(X_train_xform)        
        data['y_train_']['metadata']['processed'] = get_target_info(y_train_xform)


        # Check if split was successful
        if X_val_xform is not None:
            data['X_val_']['data'] = self.label_encoder.transform(y_val_xform)
            data['y_val_']['data'] = self.one_hot_label_encoder.transform(y_val_xform)
            data['X_val_']['metadata']['processed'] = get_feature_info(X_val_xform)        
            data['y_val_']['metadata']['processed'] = get_target_info(y_val_xform)        
        return data        

    def process_X_test_data(self, X, y=None):     
        return super(MultiClassDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        data = OrderedDict()
        data['y_test_'] = OrderedDict()
        data['y_test_']['metadata'] = OrderedDict()
        data['y_test_']['metadata']['orig'] = get_target_info(y)        
        y = self.label_encoder.transform(y)
        data['y_test_']['data'] = self.one_hot_label_encoder.transform(y)
        data['y_test_']['metadata']['processed'] = get_target_info(y)          
        return data
