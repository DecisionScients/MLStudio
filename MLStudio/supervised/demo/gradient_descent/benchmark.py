#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : benchmark.py                                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, May 24th 2020, 11:06:10 am                          #
# Last Modified : Sunday, June 14th 2020, 9:48:14 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
from collections import OrderedDict
from datetime import datetime
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
homedir = str(Path(__file__).parents[3])
demodir = str(Path(__file__).parents[1])
sys.path.append(homedir)

from mlstudio.supervised.machine_learning.gradient_descent import GDPureOptimizer
from mlstudio.supervised.visual.animations import animate_optimization
from mlstudio.supervised.algorithms.optimization.services.benchmarks import Adjiman, StyblinskiTank, Wikipedia
from mlstudio.supervised.algorithms.optimization.services.benchmarks import ThreeHumpCamel, Ursem01, Branin02
from mlstudio.supervised.algorithms.optimization.services.optimizers import GradientDescentOptimizer, Momentum, Nesterov
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adagrad, Adadelta, RMSprop
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adam, AdaMax, AdamW
from mlstudio.supervised.algorithms.optimization.services.optimizers import Nadam, AMSGrad, QHAdam
from mlstudio.supervised.algorithms.optimization.services.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.algorithms.optimization.services.optimizers import AggMo
from mlstudio.utils.data_analyzer import cosine
from mlstudio.utils.file_manager import save_df

# --------------------------------------------------------------------------  #
# Designate file locations
figures = os.path.join(demodir, "figures")
# --------------------------------------------------------------------------  #
# Package up the objective functions
optimizers = [Momentum(), Nesterov(), Adagrad(), Adadelta(), RMSprop(), Adam(),
              AdaMax(), Nadam(), AMSGrad(), AdamW(), QHAdam(),
              QuasiHyperbolicMomentum()]

objectives = [Adjiman(), Branin02(), Ursem01(),
              StyblinskiTank(), ThreeHumpCamel(), Wikipedia()]

# --------------------------------------------------------------------------  #
# Train models
solutions = OrderedDict()
results = []
for objective in objectives:
    estimators = OrderedDict()   

    for optimizer in optimizers:
        estimators[optimizer.name] = {}
        model = GDPureOptimizer(learning_rate=0.01,
                                theta_init=objective.start, 
                                epochs=500, objective=objective,
                                optimizer=optimizer)
        model.fit()
        sim = cosine(objective.minimum, model.theta_)
        d = {}
        d['DateTime'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        d['Objective'] = model.objective.name
        d['Optimizer'] = optimizer.name
        d['Epochs'] = model.epochs        
        d['Starting Learning Rate'] = model.learning_rate
        d['Final Learning Rate'] = model.eta
        if model.schedule:
            d['Schedule'] = model.schedule.name              
        else:
            d['Schedule'] = None            
        d['gradient_min'] = np.min(model.get_blackbox().epoch_log.get('gradient_norm'))
        d['gradient_max'] = np.max(model.get_blackbox().epoch_log.get('gradient_norm'))        
        d['gradient_mean'] = np.mean(model.get_blackbox().epoch_log.get('gradient_norm'))        
        d['True'] = objective.minimum
        d['est'] = model.theta_
        d['sim'] = sim
        results.append(d)
        estimators[optimizer.name]['model'] = model
        estimators[optimizer.name]['results'] = d        

    solutions[objective.name] = estimators       

df = pd.DataFrame(results)
df_filename = "Benchmark Optimizations.csv"
save_df(df, figures, df_filename)

# --------------------------------------------------------------------------  #
# Render plots
for title, solution in solutions.items():        
    filename = title + " Optimization.html"
    filepath = os.path.join(figures, title)
    animate_optimization(estimators=solution, filepath=filepath)

#%#%%