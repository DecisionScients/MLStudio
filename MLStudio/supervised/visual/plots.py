#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : static.py                                                         #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, June 10th 2020, 8:34:33 pm                       #
# Last Modified : Wednesday, June 10th 2020, 8:34:33 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Module for static plots"""
from collections import OrderedDict
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio 
import plotly.offline as py
from plotly.subplots import make_subplots
import seaborn as sns

from mlstudio.utils.file_manager import check_directory

def plot_cost_scores(data, filepath=None, show=True):
    """Renders subplots of the optimization and regression line fit.
    
    Parameters
    ----------
    data : dict
        Dictionary containing an error and a score dictionary. Each of
        those dictionaries contain the errors and scores for 
        one or more estimators. 

    filepath : str (default=None)
        The path to which the animation is to be saved.

    show : bool (default=True)
        Indicates whether an animation should be rendered in users browser.

    """        
    # ------------------------------------------------------------------  #
    # Extract plot data
    cost_plot_data = data['cost']
    score_plot_data = data['score']

    # ------------------------------------------------------------------  #
    # Get colors
    colors = px.colors.qualitative.Prism        

    # ------------------------------------------------------------------  #
    # Create subplots
    title_error = "Training Error (MSE)"
    title_score = "Test Scores (" + r'$R^2$' + ")"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title_error, title_score),
                        specs=[[{'type': "xy"}, {"type": "xy"}]])      

    # ------------------------------------------------------------------  #
    # Add Training Error Plot
    # Subplot 1, Trace 0: Bar Plot
    fig.add_trace(
        go.Bar(x=list(cost_plot_data.keys()), y=list(cost_plot_data.values()), 
                marker=dict(color=colors), showlegend=False),
                row=1, col=1)

    # ------------------------------------------------------------------  #
    # Add Test Score Plot
    # Subplot 2, Trace 1: Bar Plot
    fig.add_trace(
        go.Bar(x=list(score_plot_data.keys()), y=list(score_plot_data.values()), 
                marker=dict(color=colors), showlegend=False,
                text=[round(num,3) for num in list(score_plot_data.values())], 
                textposition='auto'),
                row=1, col=2)
    # ------------------------------------------------------------------  #
    # Set layout title, font, template, etc...
    fig.update_layout(
        height=600,
        title=dict(xanchor='center', yanchor='top', x=0.5, y=0.9),        
        font=dict(family="Open Sans"),                
        showlegend=True,            
        template='plotly_white');                       

    # ------------------------------------------------------------------  #
    # Save and rendor plot
    if filepath:
        fig.write_html(filepath, include_plotlyjs='cdn', include_mathjax='cdn')
    if show:
        pio.renderers.default = "browser"
        fig.show()
