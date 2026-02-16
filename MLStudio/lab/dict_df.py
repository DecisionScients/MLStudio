#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : dict_df.py                                                         #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 12:34:49 am                      #
# Last Modified : Wednesday, March 18th 2020, 12:34:50 am                      #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
#%%
import numpy as np
import pandas as pd
epochs = np.arange(10).tolist()
params = {'learning_rate': 2, 'batch_size': 32}
cost = np.arange(10).tolist()
params['epochs'] = epochs
params['cost'] = cost
df = pd.DataFrame(params)
print(df)

# %%
