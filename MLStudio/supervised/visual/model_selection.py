#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : model_selection.py                                                 #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Tuesday, March 17th 2020, 7:25:56 pm                         #
# Last Modified : Tuesday, March 17th 2020, 7:25:56 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
""" Model selection visualizations and animations.    

    Visualizations
    --------------    
    * LearningCurves: Training and validation metricS by training set sizes.
    * ValidationCurve: Training and validation scores by parameter values.
    * ScalabilityCurve: Fit times by training set sizes.
    * PerformanceCurve: Scores by fit times.

    Generally speaking, visualizations take a single estimator as a parameter.

    Animations
    ----------
    * OptimizationAnimation: Training errors by epoch projected onto surface plot
    * RegressionLineAnimation: 2D Regression lines projected onto scatterplot
    * OptimizationRegressionAnimation: Dual Optimization and Regression Line animations. 

    Animations accept one or more estimators as parameters. 

"""
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from plotly.subplots import make_subplots
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve
from sklearn.utils.validation import check_X_y, check_array

from mlstudio.utils.format import proper
from mlstudio.supervised.visual import COLORS
from mlstudio.supervised.visual.base import BaseVisualizer, BaseModelSelectioniVisualizer
from mlstudio.supervised.visual.base import BaseAnimator, BaseModelSelectioniAnimator
# ---------------------------------------------------------------------------- #
class LearningCurves(BaseVisualizer):
    """Plots learning, stability, and performance curves for an estimator.

    Parameters
    ----------
    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.        

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. 
    
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'  

    """

    def __init__(self, estimator, title=None, cv=None, n_jobs=None, 
                 train_sizes=np.linspace(0.1, 1.0, 5), height=450,
                 width=1200, template='plotly_white'):
        super(LearningCurves, self).__init__(
            estimator=estimator,
            title=title,
            height=height,
            width=width,
            template=template
        )
        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes
        self.title = title or "Learning Curves <br>" + \
            self.estimator.description 

    def _fit_learning_curve(self, train_sizes, train_scores, test_scores):
        """Creates traces for learning curve."""
        # Extract statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Create confidence band by wrapping and reversing the data 
        # X line
        x = train_sizes
        x_rev = x[::-1]

        # Training Data
        train_upper = train_scores_mean + train_scores_std
        train_lower = train_scores_mean - train_scores_std
        train_lower = train_lower[::-1]

        # Validation Data
        test_upper = test_scores_mean + test_scores_std
        test_lower = test_scores_mean - test_scores_std
        test_lower = test_lower[::-1]                 

        # Plot training scores confidence band
        self.fig.add_trace(go.Scatter(
            name='train_band',     
            mode='lines',       
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((train_upper,train_lower), axis=0),
            fillcolor=COLORS['blue'],
            line_color=COLORS['transparent'],
            fill='toself',
            opacity=0.15,
            showlegend=False
        ), row=1, col=1)

        self.fig.add_trace(go.Scatter(
            name='test_band',
            mode='lines',
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((test_upper,test_lower), axis=0),
            fillcolor=COLORS['green'],
            line_color=COLORS['transparent'],
            fill='toself',
            opacity=0.15,
            showlegend=False
        ), row=1, col=1)

        # Render training and validation score traces
        self.fig.add_trace(go.Scatter(
            name='Training Scores',
            mode='lines+markers',
            x=x, y=train_scores_mean,
            line=dict(color=COLORS['blue']),            
            showlegend=True
        ), row=1, col=1)

        self.fig.add_trace(go.Scatter(
            name='Validation Scores',
            mode='lines+markers',
            x=x, y=test_scores_mean,
            line=dict(color=COLORS['green']),             
            showlegend=True
        ), row=1, col=1)        

        # Format axis properties
        self.fig.update_xaxes(title_text="Training Samples", row=1, col=1)
        self.fig.update_yaxes(title_text=self.estimator.metric.label, row=1, col=1)

    def _fit_scalability_curve(self, train_sizes, fit_times):
        """Renders traces for scalability curve."""
        # Extract statistics
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Create confidence band by wrapping and reversing the data 
        # X line
        x = train_sizes
        x_rev = x[::-1]

        # Fit times
        fit_times_upper = fit_times_mean + fit_times_std
        fit_times_lower = fit_times_mean - fit_times_std
        fit_times_lower = fit_times_lower.clip(min=0)
        fit_times_lower = fit_times_lower[::-1]

        # Create trace for fit times confidence band
        self.fig.add_trace(go.Scatter(
            name='fit_times_band',     
            mode='lines',       
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((fit_times_upper, fit_times_lower), axis=0),
            fillcolor=COLORS['red'],
            line_color=COLORS['transparent'],
            fill='toself',
            opacity=0.15,
            showlegend=False
        ), row=1, col=2)

        # Create trace for fit_times
        self.fig.add_trace(go.Scatter(
            name='Fit Times',
            mode='lines+markers',
            x=x, y=fit_times_mean,
            line=dict(color=COLORS['red']),            
            showlegend=True
        ), row=1, col=2)

        # Format axis properties
        self.fig.update_xaxes(title_text="Training Samples", row=1, col=2)
        self.fig.update_yaxes(title_text="Fit Times", row=1, col=2)        

    def _fit_performance_curve(self, fit_times, train_scores, test_scores):
        """Renders traces for performance curve."""
        # Extract statistics
        fit_times_mean = np.mean(fit_times, axis=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Create confidence band by wrapping and reversing the data 
        # X line
        x = np.round(fit_times_mean,4)
        x_rev = x[::-1]

        # Training Data
        train_upper = train_scores_mean + train_scores_std
        train_lower = train_scores_mean - train_scores_std
        train_lower = train_lower[::-1]

        # Validation Data
        test_upper = test_scores_mean + test_scores_std
        test_lower = test_scores_mean - test_scores_std
        test_lower = test_lower[::-1]

        # Render training scores confidence band traces
        self.fig.add_trace(go.Scatter(
            name='train_band',     
            mode='lines',       
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((train_upper,train_lower), axis=0),
            fillcolor=COLORS['blue'],
            line_color=COLORS['transparent'],
            fill='toself',
            opacity=0.15,
            showlegend=False
        ), row=1, col=3)

        self.fig.add_trace(go.Scatter(
            name='test_band',
            mode='lines',
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((test_upper,test_lower), axis=0),
            fillcolor=COLORS['green'],
            line_color=COLORS['transparent'],
            fill='toself',
            opacity=0.15,
            showlegend=False
        ), row=1, col=3)                     

        # Render training and validation score traces
        self.fig.add_trace(go.Scatter(
            name='Training Scores',
            mode='lines+markers',
            x=x, y=train_scores_mean,
            line=dict(color=COLORS['blue']),            
            showlegend=False
        ), row=1, col=3)

        self.fig.add_trace(go.Scatter(
            name='Validation Scores',
            mode='lines+markers',
            x=x, y=test_scores_mean,
            line=dict(color=COLORS['green']),             
            showlegend=False
        ), row=1, col=3)    

        # Format axis properties
        self.fig.update_xaxes(title_text="Fit Times", row=1, col=3)
        self.fig.update_yaxes(title_text=self.estimator.metric.label, row=1, col=3)

    def fit(self, X, y):
        """Fits the estimator to the data and creates the figure object."""    

        self.fig = make_subplots(rows=1, cols=3, 
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
                subplot_titles=("Learning Curve", "Scalability Curve",
                                "Performance Curve"))

        train_sizes, train_scores, test_scores, fit_times, _  = \
            learning_curve(self.estimator, X, y, cv=self.cv, 
                           n_jobs=self.n_jobs, train_sizes=self.train_sizes,
                           return_times=True)     
                            

        self._fit_learning_curve(train_sizes, train_scores, test_scores)
        self._fit_scalability_curve(train_sizes, fit_times)
        self._fit_performance_curve(fit_times, train_scores, test_scores)

        self.fig.update_layout(
            legend=dict(orientation="h",x=0.3, y=-0.3),
            title=self.title,
            height=self.height,
            width=self.width,
            title_x=0.5,
            title_y=0.95,
            template=self.template)        

