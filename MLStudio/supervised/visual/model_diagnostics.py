#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : model_validation.py                                               #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 5:21:32 am                      #
# Last Modified : Thursday, March 19th 2020, 7:18:57 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Model Diagnostic Plots."""
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from .base import ModelVisualizer
from mlstudio.analysis.model_validation import standardized_residuals
from mlstudio.analysis.model_validation import studentized_residuals, quantile
from mlstudio.analysis.model_validation import leverage, cooks_distance
from mlstudio.supervised.visual import COLORS
from mlstudio.supervised.machine_learning.linear_regression import LinearRegression
from mlstudio.utils.format import proper        
# --------------------------------------------------------------------------  #
#                              RESIDUALS                                      #
# --------------------------------------------------------------------------  #
class Residuals(ModelVisualizer):        
    """Plots residuals versus predicted values.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    hist : bool, 'density', 'frequency', default: True

        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.       
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, hist=True, **kwargs):
        super(Residuals, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.title = self.title or str(estimator.description + "<br>Residuals vs. Predicted")

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute residuals
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred
        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Create lowess smoothing line
        z1 = lowess(residuals, y_pred, frac=1./3, it=0, is_sorted=False)

        # Create scatterplot traces
        data = [
            go.Scattergl(x=y_pred, y=residuals,
                        mode='markers',
                        marker=dict(color=COLORS['blue'],
                            line=dict(
                                color='white',
                                width=1
                            )),
                        name="Residual Plot",
                        showlegend=False),

            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        showlegend=False)                                                
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted",
                        yaxis_title="Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(y_pred),
                y0=0,
                x1=np.max(y_pred),
                y1=0,
                line=dict(
                    color=COLORS['green']
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

        # Specify existence and type of histogram 
        if self.hist is True:
            self.hist = ""
        if self.hist in ["density", ""]:
            self.fig.add_trace(go.Histogram(y=residuals,
                                name="y density train",
                                showlegend=False,
                                xaxis="x2",
                                orientation="h",
                                marker=dict(color=COLORS['green'],
                                    line=dict(
                                        color='white',
                                        width=1
                                    )),
                                histnorm=self.hist))

# --------------------------------------------------------------------------  #
#                          STANDARDIZED RESIDUALS                             #
# --------------------------------------------------------------------------  #
class StandardizedResiduals(ModelVisualizer):        
    """Standardized Residual plot.

    Renders a standardized residual plot showing the residuals on the vertical 
    axis and the predicted values on the horizontal access.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    hist : bool, 'density', 'frequency', default: True

        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.     
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, hist=True, **kwargs):
        super(StandardizedResiduals, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.title = self.title or str(estimator.description + \
            "<br>Standardized Residuals vs. Predicted")

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute standardized residuals
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred
        residuals = standardized_residuals(X, y, y_pred)

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Create lowess smoothing line
        z1 = lowess(residuals, y_pred, frac=1./3, it=0, is_sorted=False)

        # Create scatterplot traces
        data = [
            go.Scattergl(x=y_pred, y=residuals,
                        mode='markers',
                        marker=dict(color=COLORS['blue'],
                            line=dict(
                                color='white',
                                width=1
                            )),
                        name="Residual Plot",
                        showlegend=False),

            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        showlegend=False)                                                
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted",
                        yaxis_title="Standardized Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(y_pred),
                y0=0,
                x1=np.max(y_pred),
                y1=0,
                line=dict(
                    color=COLORS['green']
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

        # Specify existence and type of histogram 
        if self.hist is True:
            self.hist = ""
        if self.hist in ["density", ""]:
            self.fig.add_trace(go.Histogram(y=residuals,
                                name="y density train",
                                showlegend=False,
                                xaxis="x2",
                                orientation="h",
                                marker=dict(color=COLORS['green'],
                                    line=dict(
                                        color='white',
                                        width=1
                                    )),
                                histnorm=self.hist))                                

# --------------------------------------------------------------------------  #
#                          STANDARDIZED RESIDUALS                             #
# --------------------------------------------------------------------------  #
class StudentizedResiduals(ModelVisualizer):        
    """Studentized Residual plot.

    Renders a studentized residual plot showing the residuals on the vertical 
    axis and the predicted values on the horizontal access.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    hist : bool, 'density', 'frequency', default: True

        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.     
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, hist=True, **kwargs):
        super(StudentizedResiduals, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.title = self.title or str(estimator.description + \
            "<br>Studentized Residuals vs. Predicted")

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute studentized residuals
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred
        residuals = studentized_residuals(X, y, y_pred)

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Create lowess smoothing line
        z1 = lowess(residuals, y_pred, frac=1./3, it=0, is_sorted=False)

        # Create scatterplot traces
        data = [
            go.Scattergl(x=y_pred, y=residuals,
                        mode='markers',
                        marker=dict(color=COLORS['blue'],
                            line=dict(
                                color='white',
                                width=1
                            )),
                        name="Residual Plot",
                        showlegend=False),

            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        showlegend=False)                                                
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted",
                        yaxis_title="Studentized Residuals",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        # Create and add shapes
        self.fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=np.min(y_pred),
                y0=0,
                x1=np.max(y_pred),
                y1=0,
                line=dict(
                    color=COLORS['green']
                )
            )
        )
        self.fig.update_shapes(dict(xref='x', yref='y'))

        # Specify existence and type of histogram 
        if self.hist is True:
            self.hist = ""
        if self.hist in ["density", ""]:
            self.fig.add_trace(go.Histogram(y=residuals,
                                name="y density train",
                                showlegend=False,
                                xaxis="x2",
                                orientation="h",
                                marker=dict(color=COLORS['green'],
                                    line=dict(
                                        color='white',
                                        width=1
                                    )),
                                histnorm=self.hist))                                   

# --------------------------------------------------------------------------  #
#                            SCALE LOCATION                                   #
# --------------------------------------------------------------------------  #
class ScaleLocation(ModelVisualizer):        
    """Scale location plot.

    Renders a scale location plot showing the the square root of the 
    standardized residuals and predicted values. 

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    hist : bool, 'density', 'frequency', default: True

        Draw a histogram showing the distribution of the residuals on the 
        right side of the figure. If set to 'density', the probability
        density will be plotted. If set to 'frequency', the frequency will
        be plotted.     
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, hist=True, **kwargs):
        super(ScaleLocation, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.title = self.title or str(estimator.description + \
            "<br>Scale Location")

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute standardized residuals
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred
        # Note: residuals parameter is the square root of the standardized residuals.
        residuals = np.sqrt(standardized_residuals(X, y, y_pred))

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Create lowess smoothing line
        z1 = lowess(residuals, y_pred, frac=1./3, it=0, is_sorted=False)

        # Create scatterplot traces
        data = [
            go.Scattergl(x=y_pred, y=residuals,
                        mode='markers',
                        marker=dict(color=COLORS['blue'],
                            line=dict(
                                color='white',
                                width=1
                            )),
                        name="Residual Plot",
                        showlegend=False),

            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        showlegend=False)                                                
        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Predicted",
                        yaxis_title=r"$\sqrt{\text{Standardized Residuals}}$",
                        xaxis=dict(domain=[0,0.85],  zeroline=False),
                        yaxis=dict(domain=[0,0.85],  zeroline=False),
                        xaxis2=dict(domain=[0.85,1], zeroline=False),
                        yaxis2=dict(domain=[0.85,1], zeroline=False),                        
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)                        

        self.fig.update_shapes(dict(xref='x', yref='y'))

        # Specify existence and type of histogram 
        if self.hist is True:
            self.hist = ""
        if self.hist in ["density", ""]:
            self.fig.add_trace(go.Histogram(y=residuals,
                                name="y density train",
                                showlegend=False,
                                xaxis="x2",
                                orientation="h",
                                marker=dict(color=COLORS['green'],
                                    line=dict(
                                        color='white',
                                        width=1
                                    )),
                                histnorm=self.hist))          

# --------------------------------------------------------------------------  #
#                               QQNORM                                        #
# --------------------------------------------------------------------------  #
class QQPlot(ModelVisualizer):        
    """QQ Plot.

    The quantile-quantile (QQ) plot is used to show if two data sets come from 
    the same distribution. Concretely, one data set's quantiles are plotted 
    along the x-axis and the quantiles for the second distribution are plotted 
    on the y-axis. Typically, data sets are compared to the normal distribution.
    Considered the reference distribution, it is plotted along the X-axis as the
    "Theoretical Quantiles" while the sample is plotted along the Y-axis as
    the "Sample Quantiles".

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    a : float
        Offset for the plotting position of an expected order
        statistic, for example. The plotting positions are given
        by (i - a)/(nobs - 2*a + 1) for i in range(0,nobs+1)
    
    loc : float
        Location parameter for dist
    
    scale : float
        Scale parameter for dist
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, a=0, loc=0, scale=1, fig=None, hist=True, 
                 **kwargs):
        super(QQPlot, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.a = a
        self.loc = loc
        self.scale = scale
        self.title = self.title or str(estimator.description + \
            "<br>Normal Q-Q")

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute standardized residuals and sort
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred
        residuals = standardized_residuals(X, y, y_pred)
        residuals = np.sort(residuals)

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()        
        residuals = residuals.ravel()

        # Compute theoretical quantiles
        theoretical_quantiles = quantile(residuals)       

        # Create scatterplot traces
        data = [
            go.Scattergl(x=theoretical_quantiles, 
                         y=residuals,
                         mode='markers',
                         marker=dict(color=COLORS['blue']),
                         name="Theoretical Quantiles",
                         showlegend=True),
            go.Scattergl(x=theoretical_quantiles,
                         y=theoretical_quantiles,
                        mode='lines',
                        marker=dict(color='green'),
                        name="Theoretical Quantiles Line",
                        showlegend=False)                            

        ]

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Standardized Residuals",                       
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)        


# --------------------------------------------------------------------------  #
#                         RESIDUALS VS LEVERAGE                               #
# --------------------------------------------------------------------------  #
class ResidualsLeverage(ModelVisualizer):        
    """Residuals vs Leverage Plot.

    The Residuals vs Leverage plot helps to illuminate data points that may
    be influential. Observations that lie in the upper and lower right 
    corners of the plot may be outside of the Cook's distance line, and 
    would require further investigation.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    inner_threshold : float (Default = 0.5)
        The threshold for the cooks distance inner contour line

    outer_threshold : float (Default = 1.0)
        The threshold for the cooks distance outer contour line        

    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, inner_threshold=0.5, outer_threshold=1.0,
                **kwargs):
        super(ResidualsLeverage, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold
        self.title = self.title or str(estimator.description + \
            "<br>Residuals vs Leverage")

    def _cooks_contour(self, X, leverage, distance, sign):
        """Computes the leverage v standardized residuals, given Cooks Distance."""        
        # Compute F-distribution degrees of freedom
        n = X.shape[0]
        p = X.shape[1]

        # Designate the leverage range being plotted
        min_leverage = np.min(leverage)
        max_leverage = np.max(leverage)
        leverage_range = np.linspace(start=min_leverage, stop=max_leverage, num=n)

        # Compute standard residuals
        std_resid = sign * np.sqrt(distance * ((1-leverage_range)/leverage_range) * (p+1))        

        return leverage_range, std_resid         

    def _get_influence_points(self, cooks, distance):
        """Gets indices for influential points."""
        points = np.argwhere(np.greater(cooks,distance)).ravel()
        return points

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute standardized residuals
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        residuals = y - y_pred        
        residuals = standardized_residuals(X, y, y_pred)

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()
        residuals = residuals.ravel()

        # Compute Leverage and cooks distances
        lev = leverage(X) 
        cooks = cooks_distance(X, y, y_pred)

        # Create lowess smoothing line
        z1 = lowess(residuals, lev, frac=1./3, it=0, is_sorted=False, 
                    return_sorted=True)

        # Draw Cooks Distance contour lines 
        l1u_leverage, l1u_residual = \
            self._cooks_contour(X, lev, distance=self.inner_threshold, sign=1)
        l1l_leverage, l1l_residual = \
            self._cooks_contour(X, lev, distance=self.inner_threshold, sign=-1)
        l2u_leverage, l2u_residual = \
            self._cooks_contour(X, lev, distance=self.outer_threshold, sign=1)
        l2l_leverage, l2l_residual = \
            self._cooks_contour(X, lev, distance=self.outer_threshold, sign=-1)

        # Grab influential points
        influential_points = []
        distances = [self.inner_threshold, self.outer_threshold]
        for distance in distances:
            influential_points.extend(self._get_influence_points(cooks, distance))
        non_influential_points = [point for point in np.arange(len(residuals)) if \
            point not in influential_points]

        # Create scatterplot traces
        data = [
            go.Scatter(x=lev[non_influential_points], 
                        y=residuals[non_influential_points],
                        mode='markers+text',
                        marker=dict(color=COLORS['blue']),
                        name="Residual vs Leverage",
                        textposition="top center",
                        showlegend=False),
            go.Scattergl(x=lev[influential_points], 
                        y=residuals[influential_points], 
                        mode='markers+text',
                        marker=dict(color='red'),                        
                        text=influential_points,
                        name="High Influence Points",
                        textposition='top center',
                        showlegend=True),  
            go.Scattergl(x=z1[:,0], y=z1[:,1],
                        mode='lines',
                        marker=dict(color='red'),
                        name="Training Set Lowess",
                        showlegend=False),
            go.Scatter(x=l1u_leverage,
                       y=l1u_residual,
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks' D = " + str(self.inner_threshold),
                       showlegend=True),
            go.Scatter(x=l1l_leverage,
                       y=l1l_residual,
                       mode='lines',
                       line=dict(dash='dash'),
                       marker=dict(color='red'),
                       name="Cooks' D = 0.5 (Lower)",
                       showlegend=False),
            go.Scatter(x=l2u_leverage,
                       y=l2u_residual,
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks' D = " + str(self.outer_threshold),
                       showlegend=True),
            go.Scatter(x=l2l_leverage,
                       y=l2l_residual,
                       mode='lines',
                       line=dict(dash='dot'),
                       marker=dict(color='red'),
                       name="Cooks' D = 1 (Lower)",
                       showlegend=False)                                                                                                                                   
       ]

        # Compute x and y axis limits based 110% of the data range
        xmin = np.min(lev) + (0.1 * np.min(lev))
        xmax = np.max(lev) + (0.1 * np.max(lev)) 
        ymin = np.min(residuals) + (0.1 * np.min(residuals))
        ymax = np.max(residuals) + (0.1 * np.max(residuals))

        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Leverage",
                        yaxis_title="Standardized Residuals",
                        xaxis=dict(zeroline=False, range=[xmin, xmax]),
                        yaxis=dict(zeroline=False, range=[ymin, ymax]),                       
                        showlegend=True,
                        legend=dict(bgcolor='white'),
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)  
                                
# --------------------------------------------------------------------------  #
#                             COOKS DISTANCE                                  #
# --------------------------------------------------------------------------  #
class CooksDistance(ModelVisualizer):        
    """Cooks Distance.

    Cook's distance" is a measure of the influence of each observation on the 
    regression coefficients. The Cook's distance statistic is a measure, 
    for each observation in turn, of the extent of change in model 
    estimates when that particular observation is omitted. Any observation 
    for which the Cook's distance is close to 0.5 or more, or that is 
    substantially larger than other Cook's distances 
    (highly influential data points), requires investigation.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    threshold : float (Default - 0.5)
        The Cooks Distance threshold value for the threshold line.  

    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, threshold=0.5, **kwargs):
        super(CooksDistance, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.threshold = threshold
        self.title = self.title or estimator.description + \
            "<br>Cooks Distance" + "<br> Threshold = " + str(threshold)

    def fit(self, X, y):
        """Generates the plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute predictions
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)

        # Flatten arrays (just in case)
        y = y.ravel()
        y_pred = y_pred.ravel()

        # Compute Cooks Distance
        cooks = cooks_distance(X, y, y_pred)

        # Grab influential points above threshold
        points = np.argwhere(np.greater(cooks,self.threshold)).ravel()        

        # Create scatterplot traces
        data = [
            go.Scattergl(x=np.arange(len(cooks)), 
                        y=[round(x,4) for x in cooks],
                        mode='lines',
                        marker=dict(color=COLORS['blue']),
                        name="Cooks Distance",                                                
                        showlegend=False)                                                                                                                        
       ]
        # Designate Layout
        layout = go.Layout(title=self.title, 
                        title_x=0.5,
                        height=self.height,
                        width=self.width,
                        xaxis_title="Observation",
                        yaxis_title="Cooks Distance",
                        showlegend=False,
                        template=self.template)

        # Create figure object
        self.fig = go.Figure(data=data, layout=layout)  

        # Add horizontal line if any appoach Cooks Distance of 0.5
        if len(cooks[cooks > self.threshold]) > 0:
           self.fig.add_shape(
               go.layout.Shape(
                   type="line",
                   x0=0,
                   y0=self.threshold,
                   x1=len(cooks),
                   y1=self.threshold,
                   line=dict(
                       color="darkgrey",
                       width=2
                   )
               )
           )
           self.fig.add_trace(
               go.Scatter(
                   x=points,
                   y=cooks[points],
                   mode='markers+text',
                   marker=dict(color=COLORS['blue']),
                   name='Influential Points',
                   text=points,
                   textposition='top center'
               )
           )                               