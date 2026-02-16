#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : format.py                                                          #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 12:39:08 am                      #
# Last Modified : Wednesday, March 18th 2020, 12:39:21 am                      #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Miscellaneous formatting functions."""
#%%
import re
import random
import string
import textwrap

def proper(s):
    """Strips then capitalizes each word in a string.""" 
    s = s.replace("-", " ").title()
    s = s.replace("_", " ").title()
    return s    

def snake(s):
    """Converts string to snake case suitable for filenames."""
    s = re.sub(r"[^a-zA-Z0-9._// ]+", '', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    s = s.replace(" ", "_")
    pattern = '_' + '{2,}'
    s = re.sub(pattern, '_', s)
    return s

def format_text(x):
    x = " ".join(x.split())
    formatted = textwrap.fill(textwrap.dedent(x))
    return formatted        


# %%
