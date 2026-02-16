#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : param_grid.py                                                      #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 2:11:31 am                       #
# Last Modified : Wednesday, March 18th 2020, 2:11:31 am                       #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
#%%
from sklearn.model_selection import ParameterGrid
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_rev = x[::-1]
print(x)
print(x_rev)
print(x+x_rev)

y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y1_lower = y1_lower[::-1]
print(y1_upper+y1_lower)

# param_grid = {'a': [1, 2], 'b': [True, False]}
# grid = ParameterGrid(param_grid)
# for params in grid:
#     print("============")
#     for param, value in params.items():
#         print("----------")
#         print(param)
#         print(value)

# %%
