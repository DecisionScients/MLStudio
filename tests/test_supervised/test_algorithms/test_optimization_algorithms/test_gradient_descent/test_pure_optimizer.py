#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_optimizer.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, June 16th 2020, 12:02:30 am                        #
# Last Modified : Tuesday, June 16th 2020, 12:02:31 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Integration test for gradient descent pure optimizer class."""
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.algorithms.optimization.gradient_descent import GD
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import TimeDecay, StepDecay
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import ExponentialDecay
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import ExponentialSchedule
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import PolynomialDecay
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import PowerSchedule
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import BottouSchedule
from mlstudio.supervised.algorithms.optimization.observers.learning_rate import Adaptive
from mlstudio.supervised.algorithms.optimization.observers.early_stop import EarlyStop
from mlstudio.supervised.algorithms.optimization.services.optimizers import GradientDescentOptimizer
from mlstudio.supervised.algorithms.optimization.services.optimizers import Momentum
from mlstudio.supervised.algorithms.optimization.services.optimizers import Nesterov
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adagrad
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adadelta
from mlstudio.supervised.algorithms.optimization.services.optimizers import RMSprop
from mlstudio.supervised.algorithms.optimization.services.optimizers import Adam, AdaMax, Nadam
from mlstudio.supervised.algorithms.optimization.services.optimizers import AMSGrad, AdamW, QHAdam
from mlstudio.supervised.algorithms.optimization.services.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.algorithms.optimization.services.benchmarks import Adjiman, BartelsConn
from mlstudio.supervised.algorithms.optimization.services.benchmarks import Himmelblau, Leon
from mlstudio.supervised.algorithms.optimization.services.benchmarks import Rosenbrock, Branin02
from mlstudio.supervised.algorithms.optimization.services.benchmarks import StyblinskiTank
from mlstudio.supervised.algorithms.optimization.services.benchmarks import ThreeHumpCamel, Ursem01
from mlstudio.supervised.algorithms.optimization.services.benchmarks import Wikipedia
# --------------------------------------------------------------------------  #
#                       TEST OBJECTIVE FUNCTIONS                              #
# --------------------------------------------------------------------------  #
@mark.gradient_descent
@mark.pure_optimizer
class PureOptimizerObjectiveTests:

    objectives = [Adjiman(), BartelsConn(), Himmelblau(), Leon(), Rosenbrock(), Branin02(),
                  StyblinskiTank(), ThreeHumpCamel(), Ursem01(), Wikipedia()]

    optimizers = [GradientDescentOptimizer(), Momentum(), Nesterov(),
                  Adagrad(), Adadelta(), RMSprop(), Adam(), AdaMax(), Nadam(),
                  AMSGrad(), AdamW(), QHAdam(), QuasiHyperbolicMomentum()]

    def test_pure_optimizer_core(self):
        epochs = 500
        for objective in self.objectives:
            objective_min_norm = np.linalg.norm(objective.minimum)
            for optimizer in self.optimizers:
                est = GDPureOptimizer(epochs=epochs, 
                                                   optimizer=optimizer,
                                                   objective=objective)
                est.fit()
                bb = est.get_blackbox()
                solution_norm = np.linalg.norm(bb.epoch_log.get('theta')[-1])
                assert len(bb.epoch_log.get('epoch')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('theta')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('train_cost')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('eta')) == epochs, "Epoch log wrong length"
                msg = "\nPoor solution for objective = {o}, optimizer = {p}\n       min_norm = {m}, solution_norm = {s}".format(o = objective.__class__.__name__,
                                                                                  p = optimizer.__class__.__name__,
                                                                                  m = str(objective_min_norm),
                                                                                  s = str(solution_norm))
                if solution_norm - objective_min_norm > 50:
                    print(msg)

    