# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \data.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, July 28th 2020, 2:54:55 am                         #
# Last Modified : Tuesday, July 28th 2020, 2:54:55 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Container for data object factories."""
#%%
import os
from pathlib import Path
import site
import sys
PROJECT_DIR = str(Path(__file__).parents[2])
sys.path.append(PROJECT_DIR)

import dependency_injector.containers as containers
import dependency_injector.providers as providers
from sklearn.datasets import load_boston
from mlstudio.utils import data_manager
# --------------------------------------------------------------------------- #
class DataProcessors(containers.DeclarativeContainer):
    from mlstudio.data_services.preprocessing import RegressionDataProcessor
    from mlstudio.data_services.preprocessing import BinaryClassDataProcessor
    from mlstudio.data_services.preprocessing import MultiClassDataProcessor

    regression = providers.Factory(RegressionDataProcessor,
                                   add_bias_transformer=data_manager.AddBiasTerm(),
                                   split_transformer=data_manager.DataSplitter())

    binaryclass = providers.Factory(BinaryClassDataProcessor,
                                   add_bias_transformer=data_manager.AddBiasTerm(),
                                   split_transformer=data_manager.DataSplitter(),
                                   label_encoder=data_manager.LabelEncoder())                                   

    multiclass = providers.Factory(MultiClassDataProcessor,
                                   add_bias_transformer=data_manager.AddBiasTerm(),
                                   split_transformer=data_manager.DataSplitter(),
                                   label_encoder=data_manager.LabelEncoder(),
                                   one_hot_label_encoder=data_manager.OneHotLabelEncoder())                                   

