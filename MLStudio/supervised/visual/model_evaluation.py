# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \optimization.py                                                  #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Monday, June 29th 2020, 9:30:08 pm                          #
# Last Modified : Monday, June 29th 2020, 9:30:09 pm                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Visualizations and animations of single model optimization processes.

    Visualizations
    --------------
    * Training Optimization Curve: Training and validation losses by epoch.
    * Performance Optimization Curve: Training and validation scores by epoch.

    Both visualizations are supported by the Optimization Curve Class. A
    metric parameter specifies which visualization is rendered. A metric
    parameter value = 'cost' renders the Training Optimization Curve. Whereas
    a metric parameter value != 'cost' renders the Performance Optimization
    Curve displaying training (and validation) scores by epoch.

"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import math

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from plotly.subplots import make_subplots
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve, cross_validate
from sklearn.utils.validation import check_X_y, check_array

from mlstudio.utils.format import proper
from mlstudio.utils.data_analyzer import one_sample_ttest, critical_values
from mlstudio.supervised.visual.base import BaseVisualizer, BaseAnimator
from mlstudio.supervised.visual.base import BaseSubplot
from mlstudio.supervised.visual import COLORS
# ---------------------------------------------------------------------------- #
#                                SUBPLOTS                                      #
# ---------------------------------------------------------------------------- #
class ScoreSubplot(BaseSubplot):
    """Renders a subplot showing training and validation costs."""

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(ScoreSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label
        )

    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        scores = {proper(k):v for (k,v) in X.items() if k in ['train_score', 'test_score']}        
        return dict(sorted(scores.items(), reverse=True))

    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : dict
            Dictionary of cross-validation results from scikit-learn's 
            cross_validate function.
        
        y : None
            Not used
        """
        scores = self._extract_data(X)
        colors = [COLORS['green'], COLORS['blue']]

        for i, (k, v) in enumerate(scores.items()):
            self.fig.add_trace(go.Box(
                name=k,
                y=v,
                marker_color=colors[i],                
                showlegend=False,
            ), row=self.row, col=self.col)        
        
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)
        return self

# ---------------------------------------------------------------------------- #
class ScoreTableSubplot(BaseSubplot):
    """Updates fig object with score descriptive statistics."""

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(ScoreTableSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label
        )


    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        return {proper(k):v for (k,v) in X.items() if 'score' in k}        

    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : dict
            Dictionary from scikit-learn's cross_validate function containing
            train and test scores.
        
        y : None
            Not used
        """    
        scores = self._extract_data(X)  
        df = pd.DataFrame(data=scores)
        describe = df.describe().T

        self.fig.add_trace(
            go.Table(
                header=dict(values=list(describe.columns),
                            align='center'),
                cells=dict(values=list(describe.values),
                           align='center')),
            row=self.row, col=self.col)
        return self
# ---------------------------------------------------------------------------- #
class TimeSubplot(BaseSubplot):
    """Renders a subplot showing fit and score times."""

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(TimeSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label
        )


    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        return {proper(k):v for (k,v) in X.items() if 'time' in k}   

    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : dict
            Dictionary from scikit-learn's cross_validate function containing
            fit_time and score_time.
        
        y : None
            Not used
        """
        times = self._extract_data(X)
        colors = [COLORS['green'], COLORS['blue']]
        
        for i, (k,v) in enumerate(times.items()):
            self.fig.add_trace(go.Box(
                name=k,
                y=v,
                marker_color=colors[i],
                showlegend=False,
            ), row=self.row, col=self.col)        
        
        self.fig.update_yaxes(title_text=self.yaxis_label, row=self.row, col=self.col)
        return self

# ---------------------------------------------------------------------------- #
class TimeTableSubplot(BaseSubplot):
    """Updates fig object with fit and score time descriptive statistics."""

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(TimeTableSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label
        )

    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        return {proper(k):v for (k,v) in X.items() if 'time' in k}  

    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : dict
            Dictionary from scikit-learn's cross_validate function containing
            fit_time and score_time.
        
        y : None
            Not used
        """    
        times = self._extract_data(X)
        df = pd.DataFrame(data=times)
        describe = df.describe().T

        self.fig.add_trace(
            go.Table(
                header=dict(values=list(describe.columns),
                            align='center'),
                cells=dict(values=list(describe.values),
                           align='center')),
            row=self.row, col=self.col)

        return self

# ---------------------------------------------------------------------------- #
class FeatureSubplot(BaseSubplot):
    """Renders a subplot showing feature importances for top_n features.
    
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

    """

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(FeatureSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label
        )

    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        return X['top_features_params']     


    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : list of estimator objects. One estimator per cv split
            Estimators objects returned from scikit-learn's cross_validate 
            function.
        
        y : None
            Not used
        """

        top_features_params = self._extract_data(X)               

        # Create boxplot for each column of top_features_params
        for (feature_name, feature_parameters) in top_features_params.iteritems():
            self.fig.add_trace(go.Box(
                name=feature_name,
                y=feature_parameters,
                showlegend=False
            ), row=self.row, col=self.col)

        return self
# ---------------------------------------------------------------------------- #
class FeatureTableSubplot(BaseSubplot):
    """Updates fig object with fit and score time descriptive statistics."""

    def __init__(self, fig, row, col, xaxis_label=None, yaxis_label=None):
        super(FeatureTableSubplot, self).__init__(
            fig=fig,
            row=row,
            col=col,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label         
        )

    def _extract_data(self, X):
        """Extracts relevant data from the cross-validation results in X."""
        return X['top_features_params_stats']  

    def fit(self, X, y=None):
        """Fits data to the fig object.
        
        Parameters
        ----------
        X : dict
            Dictionary from scikit-learn's cross_validate function containing
            fit_time and score_time.
        
        y : None
            Not used
        """    
        top_features_params_stats = self._extract_data(X)

        self.fig.add_trace(
            go.Table(
                header=dict(values=list(top_features_params_stats.columns),
                            align='center'),
                cells=dict(values=[top_features_params_stats[k].tolist()\
                                    for k in top_features_params_stats.columns],
                           align='center')),
            row=self.row, col=self.col)

        return self

# ---------------------------------------------------------------------------- #
#                            VISUALIZATIONS                                    #            
# ---------------------------------------------------------------------------- #
class ModelSummary(BaseVisualizer):
    """Summarizes a model in terms of scores, fit times, and features.
    
    This visualization summarizes training and test scores, fit times and 
    feature statistics computed using KFold cross-validation.

    The summary is comprised of the following subplots:
        * ScoreSubplot: Boxplot of training and test scores
        * ScoreTableSubplot: A table with training and test score statistics
        * TimeSubplot: Boxplot of training and score times
        * TimeTableSubplot: Table with fit and score time statistics
        * FeatureSubplot: Box plot of the top_n features
        * FeatureTableSubplot: Table with top feature parameter descriptive statistics 

    Parameters
    ----------
    estimator : An unfitted MLStudio estimator object
        The estimator object being visualized

    cv : int (default=5)
        The number of cross-validation folds. 

    top_n : int in interval [1,20] (default=max(1,min(top_n, estimator.n_features_)))
        The number of features to report by feature importance.

    title : str (default=None)
        The title of the visualization. If the title is None, a default
        title will be provided.

    height : int (default=1200)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'     
         
    
    """  

    def __init__(self, estimator, cv=5, feature_names=None, top_n=20, title=None, height=1200, 
                 width=900, template='plotly_white'):
        super(ModelSummary, self).__init__(
            estimator = estimator,
            title = title or "Model Summary<br>" + estimator.description,
            height = height,
            width = width,
            template = template,
            cv=cv,
            top_n = max(1,min(top_n, 20)),
            feature_names = feature_names            
        )

    def _perform_cross_validation(self, X, y):
        """Performs KFold cross-validation using 'cv' folds."""
        cv_results = cross_validate(estimator=self.estimator, X=X, y=y, 
                        cv=self.cv, return_train_score=True, 
                        return_estimator=True)
        return cv_results

    def _create_dummy_feature_names(self, cv_results):
        """Creates dummy features of the format "Xn", where n is the feature index."""
        estimators = cv_results.get('estimator')
        n_features = estimators[0].n_features_
        feature_names = np.array(["X" + str(n) for n in range(n_features)])
        return feature_names

    def _create_parameters_df(self, cv_results):
        """Creates a dataframe of shape (n_splits, n_features) of parameter values."""
        df_parameters = pd.DataFrame()
        estimators = cv_results['estimator']
        for idx, estimator in enumerate(estimators):
            d = {}            
            split = "Split_" + str(idx)
            d[split] = estimator.theta_
            df_theta = pd.DataFrame.from_dict(data=d, orient='index',
                                    columns=self.feature_names) 
            df_parameters = pd.concat([df_parameters, df_theta], axis=0)
        return df_parameters

    def _extract_top_n_features(self, df_params):
        """Returns a dataframe containing the top_n features by importance."""
        # Compute importance as absolute value of mean parameter values
        feature_importances = np.abs(df_params.mean(axis=0).to_numpy())
        # Compute top_n as minimum of top_n parameter and df_params.shape[1]
        top_n = min(self.top_n, df_params.shape[1])
        # Extract an array of indices for top_n columns by importance
        top_n_feature_indices = np.argpartition(feature_importances, \
            -top_n)[-top_n:] 
        # Subset the parameters dataframe by these indices
        df_top_n_features = df_params.iloc[:,list(top_n_feature_indices)]        
        return df_top_n_features

    def _compute_feature_stats(self, feature_name, feature_parameters, alpha=0.05):
        """Computes descriptive statistics, and confidence interval for an individual feature"""
        # Compute degrees of freedom for one sample ttest        
        df = len(feature_parameters) - 1 
        # Compute critical values based upon significance and degrees of freedom
        cv = critical_values(feature_parameters, df, alpha)
        fs = pd.DataFrame()
        fs['feature'] = feature_name
        fs['mean'] = np.mean(feature_parameters)
        fs['std'] = np.std(feature_parameters)
        fs['se'] = fs['std'] / np.sqrt(len(feature_parameters))
        fs['t'], fs['p_value'] = one_sample_ttest(feature_parameters)
        fs['lower_ci'] = fs['mean'] - cv * fs['se']
        fs['upper_ci'] = fs['mean'] + cv * fs['se']
        return fs

    def _compute_parameter_stats(self, top_features):
        """Computes parameter statistics for all features and splits"""
        parameter_stats = pd.DataFrame()
        for (feature_name, feature_parameters) in top_features.iteritems():
            feature_stats = self._compute_feature_stats(feature_name, 
                                    feature_parameters)
            parameter_stats = pd.concat((parameter_stats, feature_stats), axis=0)
        return parameter_stats 

    def _compute_feature_data(self, cv_results):
        if self.feature_names is None:
            self.feature_names = self._create_dummy_feature_names(cv_results)
        
        df_params = self._create_parameters_df(cv_results)
        cv_results['top_features_params'] = self._extract_top_n_features(df_params)   
        cv_results['top_features_params_stats'] = \
            self._compute_parameter_stats(cv_results['top_features_params'])
        return cv_results

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
        cv_results = self._perform_cross_validation(X, y)
        # Add parameter values and statistics to cv_results dictionary
        cv_results = self._compute_feature_data(cv_results)

        self.fig = make_subplots(
            rows=6, cols=2,
            vertical_spacing=0.03,
            specs=[[{"rowspan": 2, "type": "xy"}, {"rowspan": 2, "type": "xy"}],
                   [None, None],
                   [{"type": "table"}, {"type": "table"}],
                   [{"colspan": 2, "rowspan": 2, "type": "xy"}, None],
                   [None, None],
                   [{"colspan": 2, "type": "table"}, None]],
            subplot_titles=("Scores", "Times", "Score Statistics", "Time Statistics",
                            "Top Feature Importances", "Top Feature Parameter Statistics")
        )

        # Create scores subplot
        scores_subplot = ScoreSubplot(fig=self.fig, row=1, col=1, 
                            yaxis_label=self.estimator.metric.label)
        self.fig = scores_subplot.fit(X=cv_results).fig

        # Create fit and score times subplot
        times_subplot = TimeSubplot(fig=self.fig, row=1, col=2,
                            yaxis_label="Time (ms)")
        self.fig = times_subplot.fit(cv_results).fig

        # Create scores table subplot
        scores_table_subplot = ScoreTableSubplot(fig=self.fig, row=3, col=1)
        self.fig = scores_table_subplot.fit(cv_results).fig

        # Create times table subplot
        times_table_subplot = TimeTableSubplot(fig=self.fig, row=3, col=2)
        self.fig = times_table_subplot.fit(cv_results).fig        

        # Create feature importance subplot
        feature_subplot = FeatureSubplot(fig=self.fig, row=4, col=1)
        self.fig = feature_subplot.fit(cv_results).fig

        # Create feature table subplot
        feature_table_subplot = FeatureTableSubplot(fig=self.fig, row=6, col=1)
        self.fig = feature_subplot.fit(cv_results).fig        

        # Set layout 
        self.fig.update_layout(
            legend=dict(orientation="h",x=0.3, y=-0.3),
            title=self.title,
            height=self.height,
            width=self.width,
            title_x=0.5,
            title_y=0.99,
            template=self.template)        




# ---------------------------------------------------------------------------- #
class OptimizationCurve(BaseVisualizer):
    """ Plots training and validation error by epoch

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

    metric : str 'cost' or 'score' (default='cost')
        Determines whether to plot cost or scores

    """
    _default_title = {'cost': 'Training Optimization Curve',
                      'score': 'Performance Optimization Curve'}

    def __init__(self, estimator, title=None, height=450, width=900, 
                 template='plotly_white', metric='cost'): 
        super(OptimizationCurve, self).__init__(
            estimator=estimator,
            title=title or self._default_title[metric],
            height=height,
            width=width,
            template=template
        )    
        self.metric = metric

    def _get_data(self):
        """Extracts data from blackbox and returns a dictionary for plotting."""
        d = {}
        d['Epoch'] = np.arange(1,self.estimator.get_blackbox().total_epochs+1)
        if self.metric == 'cost':
            d['yaxis_title'] = self.estimator.objective.name
            d['Training Loss'] = self.estimator.get_blackbox().epoch_log.get('train_cost')

        else:
            d['yaxis_title'] = self.estimator.metric.label
            d['Training Score'] = self.estimator.get_blackbox().epoch_log.get('train_score')
            if self.estimator.get_blackbox().epoch_log.get('val_score'):
                d['Validation Score'] = self.estimator.get_blackbox().epoch_log.get('val_score')            
        return d

    def fit(self, X, y):
        """Fits the model and creates the figure object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        
        """
        super(OptimizationCurve, self).fit(X, y)        
        # Extract data from the estimator's blackbox for plotting
        data = self._get_data()

        self.fig = go.Figure()

        # Training metric trace
        [training_metric] = [metric for metric in data.keys() if "Training" in metric]        
        self.fig.add_trace(go.Scatter(
            name=training_metric,
            mode='lines',
            x=data['Epoch'], y=data[training_metric],
            line=dict(color='#005b96'),            
            showlegend=True
        ))

        # Validation metric if available
        if self.estimator.val_size and 'score' in self.metric:
            [validation_metric] = [metric for metric in data.keys() if "Validation" in metric]        
            self.fig.add_trace(go.Scatter(
                name=validation_metric,
                mode='lines',
                x=data['Epoch'], y=data[validation_metric],
                line=dict(color='rgb(27,158,119)'),             
                showlegend=True
            ))

        self.fig.update_layout(
            title=self.title,
            height=self.height,
            width=self.width,
            xaxis=dict(title='Epoch'),
            yaxis=dict(title=data['yaxis_title']),
            title_x=0.5,
            template=self.template)

        return self
