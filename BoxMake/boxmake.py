#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:50:48 2020

@author: matsumoto
"""
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from chembl.RestrictData import restrict_dpnum, larged_pull
import matplotlib.pyplot as plt
import seaborn as sns
from chembl.AssayPreparation import AssayPrep
from Dataset.MutualDelete import mutual_assay_del
from scipy import stats


def read_result_cox(keys, thresholds, 
                fpath='/Users/matsumoto/Research/Research/Methods/NearbyLearn/Rank/RankLooCox',
                default=True, scaling=False):
    
    origin_path = fpath
    
    results = pd.DataFrame()
    for test_key in keys:
        if default == True:
            path_val = os.path.join(origin_path, 'thres{}/test{}'.format(thresholds[0], test_key))
            res_val  = pd.read_csv(os.path.join(path_val, 'metric_val.csv'), index_col=0)
            res_val['assay'] = test_key
            results          = pd.concat([results, res_val], axis=0)
        
        for thres in thresholds:
            path = os.path.join(origin_path, 'thres{}/test{}'.format(thres, test_key))
            res  = pd.read_csv(os.path.join(path, 'metric.csv'), index_col=0)
            res['assay'] = test_key
            results      = pd.concat([results, res], axis=0)
        if scaling == True:
            path_sc = os.path.join(origin_path, 'thres{}/test{}'.format(thresholds[0], test_key))
            res_sc  = pd.read_csv(os.path.join(path_sc, 'metric_sc.csv'), index_col=0)
            res_sc['assay'] = test_key
            results         = pd.concat([results, res_sc], axis=0)
            
    return results


def read_other_cox(keys, 
                   path1 = '/Users/matsumoto/Research/Research/Methods/PLSAsyInfo/PLSOHVectorCox',
                   path2 = '/Users/matsumoto/Research/Research/Methods/PLSAsyInfo/PLSIC50InfoCox'):
    
    results = pd.DataFrame()
    for test_key in keys:
        path_1 = os.path.join(path1, 'test{}'.format(test_key))
        path_2 = os.path.join(path2, 'test{}'.format(test_key))
        
        res1 = pd.read_csv(os.path.join(path_1, 'metric.csv'), index_col=0)
        res2 = pd.read_csv(os.path.join(path_2, 'metric.csv'), index_col=0)
        res1['assay'], res2['assay'] = test_key, test_key
        results      = pd.concat([results, res1, res2], axis=0)
        
    return results