# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \diagnostics.py                                                   #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 30th 2020, 6:33:05 pm                        #
# Last Modified : Thursday, July 30th 2020, 6:33:06 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Model diagnostics and configuration."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from tabulate import tabulate

from mlstudio.utils.data_analyzer import describe_numeric_array

def diagnose_gradient(estimator):
    """Produces descriptive statistics for the gradient."""
    gradient_norms = estimator.get_blackbox().epoch_log.get('gradient_norm')    
    df = describe_numeric_array(gradient_norms, fmt='df')            
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    return df


