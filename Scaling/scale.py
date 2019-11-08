import pandas as pd
import numpy as np
from chemical.FingerPrint import Hash2FingerPrint
from scipy.spatial.distance import cdist
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:36:43 2019

@author: matsumoto
"""

def SimilarPairIndex(dict_data, ts_n, tr_n, hshcol="hash"):
    """
    This function outputs high simirality compound's index
    ts_n : Test assay number / tr_n : Train assay number
    
    index[n]:compound ID in test assay
    columns[n]:compound ID in train assay
    dis.iloc[n,d]:tanimoto similarity between compounds having most similarity
    """
    hsh10 = dict_data[ts_n][hshcol]
    fp10 = Hash2FingerPrint(hsh10)
    
    hsh = dict_data[tr_n][hshcol]
    fp = Hash2FingerPrint(hsh)
    
    dis = pd.DataFrame(1 - cdist(fp10,fp,metric='jaccard'), index=fp10.index, columns=fp.index)
    n, d = np.unravel_index(np.argmax(dis.values), dis.shape)
    
    return dis.index[n], dis.columns[d], dis.iloc[n,d]

def ScalingRatioCalc(dict_data, ts_n, tr_n, ic50col="Standard Value"):
    """
    This finction calculates scaling ratio
    ts_n : Test assay number / tr_n : Train assay number
    """
    ic50ts = dict_data[ts_n][dict_data[ts_n].index == SimilarPairIndex(dict_data, ts_n,tr_n)[0]][ic50col]
    ic50tr = dict_data[tr_n][dict_data[tr_n].index == SimilarPairIndex(dict_data, ts_n,tr_n)[1]][ic50col]
    
    return int(ic50ts)/int(ic50tr)
