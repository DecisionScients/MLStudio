# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \database.py                                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Wednesday, July 29th 2020, 4:41:55 am                       #
# Last Modified : Wednesday, July 29th 2020, 4:41:55 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Module that controls access to the ZODB database."""
#%%
from abc import ABC, abstractmethod
import os
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PERSISTENCE_DIR = os.path.join(DATA_DIR, "persistence")
site.addsitedir(PROJECT_DIR)
site.addsitedir(DATA_DIR)
site.addsitedir(PERSISTENCE_DIR)

import BTrees.OOBTree
import transaction
from ZODB import DB, FileStorage

from mlstudio.supervised.metrics import panel
class AbstractDataBase(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    def save(self, o):
        filepath = os.path.join(PERSISTENCE_DIR, self._name)
        storage = FileStorage.FileStorage(filepath)
        db = DB(storage)
        connection = db.open()
        root = connection.root()
        root.panels = BTrees._OOBTree.BTree()
        root.panels[self._name] = o

    def load(self):
        filepath = os.path.join(PERSISTENCE_DIR, self._name)
        storage = FileStorage.FileStorage(filepath)
        db = DB(storage)
        connection = db.open()
        root = connection.root()
        root.panels = BTrees._OOBTree.BTree()
        return root.panels[self._name]        

class PanelDB(AbstractDataBase):

    def __init__(self):
        self._name = "panel_db"

class PanelRepoDB(AbstractDataBase):

    def __init__(self):
        self._name = "panel_repository_db"        
    