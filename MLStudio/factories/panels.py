# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \panels.py                                                        #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 30th 2020, 9:09:09 pm                        #
# Last Modified : Thursday, July 30th 2020, 9:09:09 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Factories for metrics panels."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.metrics.panel import RegressionPanel
from mlstudio.supervised.metrics.panel import BinaryClassPanel
from mlstudio.supervised.metrics.panel import MultiClassPanel
from mlstudio.supervised.metrics.panel import RegressionPanelFactory
from mlstudio.supervised.metrics.panel import BinaryClassPanelFactory
from mlstudio.supervised.metrics.panel import MultiClassPanelFactory
# --------------------------------------------------------------------------- #
class PanelFactories(containers.DeclarativeContainer):
    """IoC container for panel factories."""

    regression = providers.Factory(RegressionPanelFactory(panel=RegressionPanel))

    binaryclass = providers.Factory(BinaryClassPanelFactory(panel=BinaryClassPanel))

    multiclass = providers.Factory(MultiClassPanelFactory(panel=MultiClassPanel))

