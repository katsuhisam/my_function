#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:08:46 2020

@author: matsumoto
"""
import pandas as pd
import itertools

def mutual_assay_del(data, thres=0.8):
    """
    Remove any assay that contains more than 
    80% of the same compounds as a particular assay.
    """
    keys = list(set(data.Assay))
    pa = [[(key, key2) for key2 in keys[(num+1):]] for num, key in enumerate(keys)]
    pairs = list(itertools.chain.from_iterable(pa))
    
    delkeys = []
    for pair in pairs:
        p_o = pair[0]
        p_i = pair[1]
        
        data_o = data.query('Assay == @p_o')
        o_idx = [idx for idx in data_o.index]
        onum = len(data_o)
        
        data_i = data.query('Assay == @p_i')
        i_idx = [idx for idx in data_i.index]
        
        i_o_and = set(o_idx) & set(i_idx)
        andn = len(i_o_and)
        
        if andn / onum > thres:
            delkeys.append(p_i)
    delkeys = list(set(delkeys))
    
    datac = data.query('Assay != @delkeys')
    
    return datac