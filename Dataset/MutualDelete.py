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


def mutual_assay_del_new(data, thres=0.8):
    """
    Remove any assay that contains more than 
    80% of the same compounds as a particular assay.
    modefied fucton of mutual_asssay_del
    """
    
    keys = list(set(data.Assay))
    pa = [[(key, key2) for key2 in keys[(num+1):]] for num, key in enumerate(keys)]
    pairs = list(itertools.chain.from_iterable(pa))
    
    delkeys = []
    for pair in pairs:
        p_o = pair[0]
        p_i = pair[1]
        
        data_o = data.query('Assay == @p_o')
        o_idx  = [idx for idx in data_o.index]
        
        data_i = data.query('Assay == @p_i')
        i_idx  = [idx for idx in data_i.index]
        
        i_o_and = set(o_idx) & set(i_idx)
        andn    = len(i_o_and)
        
        num_min = min([len(data_o), len(data_i)])
        
        mutual = andn / num_min
        if  mutual > thres:
            delkeys.append(p_i)
    delkeys = list(set(delkeys))
    
    datac = data.query('Assay != @delkeys')
    
    return datac


def mutual_assay_detail(data):
    
    datac = mutual_assay_del(data)
    
    keys   = list(set(datac.Assay))
    pd_mut = pd.DataFrame(index=keys, columns=keys) 
    
    for asy1 in keys:
        for asy2 in keys:
            
            data1 = datac.query('Assay == @asy1')
            idx1  = [idx for idx in data1.index]
            num1  = len(data1)
            
            data2 = datac.query('Assay == @asy2')
            idx2  = [idx for idx in data2.index]
            num2  = len(data2)
            
            num_min = min([num1, num2])
            andidx  = set(idx1) & set(idx2)
            andn    = len(andidx)
            
            per = andn / num_min
            pd_mut.at[asy1, asy2] = per
            
    return pd_mut
            
    