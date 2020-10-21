#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:52:47 2020

@author: matsumoto
"""
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from chemical.FingerPrint import Hash2FingerPrint


def make_dx(data):
    """
    ake Distance matrix (distance : Tanimoto)
    Duplicate compound rows and columns should be deleted.

    Returns
    -------
    M : DataFrame
        Distance Matrix

    """
    
    X_all = Hash2FingerPrint(data.hash)
    y = 1 - pdist(X_all, metric='jaccard')
    
    M = pd.DataFrame(squareform(y), index=X_all.index, columns=X_all.index) #Distance Matrix
    M = M[~M.index.duplicated()]
    M = M.loc[:, ~M.columns.duplicated()]
    
    return M