#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:36:05 2020

@author: matsumoto
"""
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from chembl.AssayPreparation import AssayPrep


def metric_concat(path, mode='rank'):
    
    path = path
    
    fps  = os.listdir(path)
    fps  = [name for name in fps if 'csv' in name]
    fps.sort()
    
    mets = pd.DataFrame()
    for fp in fps:
        met  = pd.read_csv(os.path.join(path, fp), index_col=0)
        if mode == 'reg':
            mets = pd.concat([mets, met.R2ts], axis=1)
        elif mode == 'rank':
            mets = pd.concat([mets, met.spearmanr], axis=1)
            
        n = fp.split('.')[0].split('_')[1]
        
        if mode == 'reg':
            mets = mets.rename(columns={'R2ts': n})
        elif mode == 'rank':
            mets = mets.rename(columns={'spearmanr': n})
        
    return mets

def median_series(metric_df):

    ddif = pd.DataFrame()
    for n in range(len(metric_df.columns)):
        
        v   = metric_df.val
        met = metric_df.iloc[:,n]
        
        dif  = met - v
        ddif = pd.concat([ddif, dif], axis=1)
    
    ddif.columns = metric_df.columns
    med = ddif.median()
    
    med       = med.drop(index='val')
    med.index = [int(num) for num in med.index]

    med = med.sort_index()
    
    return med


def median_df(asy_list, path, mode='rank'):
    meds = pd.DataFrame()
    
    for asy in asy_list:
        mets = metric_concat(path = os.path.join(path, 'test{}_comb').format(asy), mode=mode)
        med  = median_series(mets)
        
        meds = pd.concat([meds, med], axis=1)
    meds.columns = asy_list
    
    return meds


def range_calc(data, asylist):
    
    rds = np.zeros(shape=(len(asylist), len(asylist)))
    for idx, num in enumerate(asylist):
        for idx_in, num_in in enumerate(asylist):
            if num == num_in:
                r = 0
            else:
                d = data.query('Assay == @num or Assay == @num_in')
                r = d['pot.(log,IC50)'].max() - d['pot.(log,IC50)'].min()
            rds[idx, idx_in] = r
    rds = pd.DataFrame(rds, index=asylist, columns=asylist)
    
    return rds