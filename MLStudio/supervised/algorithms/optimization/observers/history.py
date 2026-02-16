# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \history.py                                                       #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, July 28th 2020, 8:54:52 pm                         #
# Last Modified : Tuesday, July 28th 2020, 9:09:41 pm                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Module containing observers that monitor and report on optimization."""
import datetime
import numpy as np
from mlstudio.supervised.algorithms.optimization.observers.base import Observer
# --------------------------------------------------------------------------- #
#                                BLACKBOX                                     #
# --------------------------------------------------------------------------- #
class BlackBox(Observer):
    """Repository for data obtained during optimization."""

    def __init__(self):
        super(BlackBox, self).__init__()
        self.name = "BlackBox"
        self.total_epochs = 0
        self.total_batches = 0
        self.start = None 
        self.end = None
        self.duration = None
        self.epoch_log = {}
        self.batch_log = {}            

    def on_train_begin(self, log=None):
        """Sets instance variables at the beginning of training.
        
        Parameters
        ----------
        log : Dict
            Dictionary containing the X and y data
        """ 
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now()
        self.end = None
        self.duration = None
        self.epoch_log = {}
        self.batch_log = {}        

    def on_train_end(self, log=None):        
        """Sets instance variables at end of training.
        
        Parameters
        ----------
        log : Dict
            Not used 
        """
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 

    def on_batch_end(self, batch, log=None):
        """Updates data and statistics relevant to the training batch.
        
        Parameters
        ----------
        batch : int
            The current training batch
        
        log : dict
            Dictionary containing batch statistics, such as batch size, current
            weights and training cost.
        """
        log = log or {}
        self.total_batches += 1
        for k,v in log.items():
            self.batch_log.setdefault(k,[]).append(v)            

    def on_epoch_end(self, epoch, log=None):
        """Updates data and statistics relevant to the training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch
        
        log : dict
            Dictionary containing data and statistics for the current epoch,
            such as weights, costs, and optional validation set statistics
            beginning with 'val_'.
        """
        log = log or {}
        self.total_epochs += 1
        for k,v in log.items():
            self.epoch_log.setdefault(k,[]).append(v)
