#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:49:16 2020

@author: matsumoto
"""
import pandas as pd
import numpy as np
import os
import pickle
import copy


def restrict_dnum(dfs, dnum):
    """
    Put a limit on the number of data in the assay.
    """
    
    dfs = copy.deepcopy(dfs)
    
    for key, dic in list(dfs.items()):
        for keyin, dicin in list(dic.items()):
            for keydin, df in list(dicin.items()):
                if df.shape[0] < dnum:
                    del dicin[keydin]
        
            if len(dicin) == 0:
                del dic[keyin]
                
        if len(dic) == 0:
            del dfs[key]
            print('{} is deleted'.format(key))
        else:
            print(key)
        
    return dfs


def restrict_pnum(dfs, pnum):
    """
    Put a limit on the number of assay pairs.
    """
    
    dfs = copy.deepcopy(dfs)
    
    for key, dic in list(dfs.items()):
        for keyin, dicin in list(dic.items()):
            if len(dicin) < pnum:
                del dic[keyin]
        
        if len(dic)  == 0:
           del dfs[key]
           print('{} is deleted'.format(key))
           
    return dfs


def restrict_dpnum(dfs, dnum, pnum):
    """
    Put a limit on the number of assay types and 
    the number of data (compounds) included in those assays.

    Parameters
    ----------
    dfs : TYPE
        DESCRIPTION.
    dnum : int
        Number of data (compounds) included in assays.
    pnum : TYPE
        Number of assay types.

    Returns
    -------
    dfs : TYPE
        DESCRIPTION.

    """
    
    dfs = copy.deepcopy(dfs)
    
    dfs = restrict_dnum(dfs, dnum)
    dfs = restrict_pnum(dfs, pnum)
    
    return dfs


def larged_pull(dfs):
    """
    Pull the data for the most assay types.
    """
    dfs = copy.deepcopy(dfs)
    
    for key, dic in list(dfs.items()):
        if len(dic) == 1:
            continue
        
        main = 0
        for keyin, dicin in list(dic.items()):
            if len(dicin)>main:
                main_d = dicin
            d = {keyin: main_d}
        
        dfs[key] = d
        
    return dfs
            

if __name__ == '__main__':
    
    dfs = pickle.load(open('/Users/matsumoto/Research/Research/object/dup_assays.pickle','rb'))
    
    df_new = restrict_dpnum(dfs=dfs, dnum=30, pnum=5)
    df_new = larged_pull(df_new)
    
    
