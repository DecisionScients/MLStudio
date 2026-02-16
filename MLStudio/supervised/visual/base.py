#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : base.py                                                            #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Tuesday, March 17th 2020, 7:15:23 pm                         #
# Last Modified : Tuesday, March 17th 2020, 7:15:23 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Base classes for visualizations."""
import os
from abc import ABC, abstractmethod

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio 
from sklearn.base import BaseEstimator
from mlstudio.utils.file_manager import save_plotly_figure, save_plotly_animation
# --------------------------------------------------------------------------- #
#                         BASE VISUALIZER                                     #
# --------------------------------------------------------------------------- #
class BaseVisualizer(ABC, BaseEstimator):
    """Abstact base class for static visualizations of a single model.

    Parameters
    ----------
    estimator : An unfitted MLStudio estimator object
        The estimator object being visualized

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'    

    kwargs : dict
        Additional keyword arguments to be passed to the underlying
        plotly object.

    """
    
    def __init__(self, estimator, title=None, height=450, width=900, 
                 template='plotly_white', **kwargs):        
        self.estimator = estimator
        self.title = title
        self.height = height
        self.width = width
        self.template = template
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X, y):
        """Fits the visualizer to the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features)
            Target relative to X for classification or regression
        """
        self.estimator.fit(X,y)    
    
    def show(self, **kwargs):
        """Renders the visualization"""
        self.fig.show()

    def save(self, filepath):
        """Saves image to filepath

        Parameters
        ----------
        filepath : str
            Relative filepath including file name and extension
        """
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        save_plotly_figure(fig=self.fig, directory=directory, filename=filename)

# --------------------------------------------------------------------------- #
#                       BASE MODEL SELECTION VISUALIZER                       #
# --------------------------------------------------------------------------- #
class BaseModelSelectioniVisualizer(ABC, BaseEstimator):
    """Abstact base class for static visualizations of multiple models.

    Parameters
    ----------
    estimators : list of MLStudio estimator objects
        The estimators to be visualized

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'    

    kwargs : dict
        Additional keyword arguments to be passed to the underlying
        plotly object.

    """
    
    def __init__(self, estimators, title=None, height=450, width=900, 
                 template='plotly_white', **kwargs):        
        self.estimators = estimators
        self.title = title
        self.height = height
        self.width = width
        self.template = template
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X, y):
        """Fits the visualizer to the data."""
        self.models_ = []
        for estimator in self.estimators:
            self.models_.append(estimator.fit(X, y))
                
    
    def show(self, **kwargs):
        """Renders the visualization"""
        self.fig.show()

    def save(self, filepath):
        """Saves image to filepath

        Parameters
        ----------
        filepath : str
            Relative filepath including file name and extension
        """
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        save_plotly_figure(fig=self.fig, directory=directory, filename=filename)

# --------------------------------------------------------------------------- #
#                           BASE ANIMATOR                                     #
# --------------------------------------------------------------------------- #
class BaseAnimator(ABC, BaseEstimator):
    """Abstract base class for animations for a single model.

    Parameters
    ----------
    estimator : MLStudio estimator object
        The estimator being visualized

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'    

    kwargs : dict
        Additional keyword arguments to be passed to the underlying
        plotly object.

    """    
    def __init__(self, estimator, title=None, height=450, width=900, 
                 template='plotly_white', **kwargs):        
        self.estimator = estimator
        self.title = title
        self.height = height
        self.width = width
        self.template = template
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X, y):
        """Fits the visualizer to the data."""
        self.estimator.fit(X, y)    
    
    def show(self, **kwargs):
        """Renders the visualization"""
        pio.renderers.default = "browser"
        self.fig.show()

    def save(self, filepath):
        """Saves image to filepath

        Parameters
        ----------
        filepath : str
            Relative filepath including file name and extension
        """
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        save_plotly_animation(fig=self.fig, directory=directory, filename=filename)    

# --------------------------------------------------------------------------- #
#                      BASE MODEL SELECTION ANIMATOR                          #
# --------------------------------------------------------------------------- #
class BaseModelSelectioniAnimator(ABC, BaseEstimator):
    """Abstract base class for animations for a single model.

    Parameters
    ----------
    estimators : List
        A list of MLStudio estimator objects

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'    

    kwargs : dict
        Additional keyword arguments to be passed to the underlying
        plotly object.

    """    
    def __init__(self, estimators, title=None, height=450, width=900, 
                 template='plotly_white', **kwargs):        
        self.estimators = estimators
        self.title = title
        self.height = height
        self.width = width
        self.template = template
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X, y):
        """Fits the visualizer to the data."""
        self.models_ = []
        for estimator in self.estimators:
            self.models_.append(estimator.fit(X, y))

    
    def show(self, **kwargs):
        """Renders the visualization"""
        pio.renderers.default = "browser"
        self.fig.show()

    def save(self, filepath):
        """Saves image to filepath

        Parameters
        ----------
        filepath : str
            Relative filepath including file name and extension
        """
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        save_plotly_animation(fig=self.fig, directory=directory, filename=filename)           

# --------------------------------------------------------------------------- #
#                             BASE SUBPLOT                                    #
# --------------------------------------------------------------------------- #
class BaseSubplot(ABC, BaseEstimator):        
    """Abstract base class for all subplot classes.
    
    Subplot classes add traces and updates to a figure object passed as a
    parameter to the class. These objects do not render plots or animations
    in and of themselves. The purpose is to create modularization in the
    way that plots having 2 or more subplots are built.

    Parameters
    ----------
    fig : plotly Figure object
        The object upon which the subplot is built

    row : int
        The row in which the subplot is to be rendered

    col : int
        The column in which the subplot is to be rendered

    xaxis_label : str or None (default=None)
        The label for the xaxis of the subplot

    yaxis_label : str or None (default=None)
        The label for the yaxis of the subplot        

    kwargs : dict
        Additional keyword arguments required by subclasses.

    """

    @abstractmethod
    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None,
                 **kwargs):
        self.fig = fig
        self.row = row
        self.col = col
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X, y=None):
        """Adds the traces and updates to the fig object.
        
        Parameters
        ----------
        X : array-like in 1 dimension 
            The x-axis data

        y : array-like of 1 dimension (optional)
            The y-axis data

        """ 
        pass       

