#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:46:44 2020

@author: matsumoto
"""

import pandas as pd
from scipy.spatial.distance import pdist, squareform


def nearby_index(test, train, dx, threshold=0.5):
    
    """
    Put index nearby test compounds.
    dx : Similarity matrix (Tanimoto Similarity) 
    """
    
    ts_idx = [idx for idx in test.index]
    tr_idx = [idx for idx in train.index]
    
    trs = []
    for tsidx in ts_idx:
        for tridx in tr_idx:
            n = dx.at[tsidx, tridx]
            if n > threshold:
                trs.append(tridx)
    trs = list(set(trs))
    return trs


    
def cluster_distance_mx(data, x, assay='Assay'):
    """
    Making distance matrix between assay clusters.

    Returns
    Distance Matrix : M

    """
    keys        = list(set(data[assay]))
    keys_sorted = sorted(keys)
    
    means = pd.DataFrame()
    for idx, asy in enumerate(keys_sorted):
        x_portion = x[data[assay] == asy]
        mean      = x_portion.mean(axis=0)
        
        means     = pd.concat([means, mean], axis=1)
        
    means       = means.T
    means.index = keys_sorted
    
    # distance
    y = pdist(means)
    M = squareform(y)
    M = pd.DataFrame(M, index=keys_sorted, columns=keys_sorted)
    
    return M


def nearby_cluster(data, test_key, dx, threhold=4.0):
    
    keys = dx.index
    
    using_idx = []
    for key in keys:
        if key == test_key:
            continue
        
        n = dx.at[test_key, key]
        if n < threhold:
            using_idx.append(key)
    
    data = data.query('Assay == @test_key or Assay == @using_idx')
    return data


def nearby_cluster_out_train(data, test_key, dx, threhold=4.0):
    
    keys = dx.index
    
    using_idx = []
    for key in keys:
        if key == test_key:
            continue
        
        n = dx.at[test_key, key]
        if n < threhold:
            using_idx.append(key)
    
    train_data = data.query('Assay == @using_idx')
    return train_data


def gravity_pointer(data, x, assay='Assay'):
    
    keys = list(set(data[assay]))

    means = pd.DataFrame()
    for idx, asy in enumerate(keys):
        x_portion = x[data[assay] == asy]
        mean      = x_portion.mean(axis=0)
        
        means     = pd.concat([means, mean], axis=1)
        
    means       = means.T
    means.index = keys
    
    df_grav = pd.concat([x, means], axis=0)
    
    return df_grav