import pandas as pd
from chemical.FingerPrint import Smiles2Hash
from sampling.sampling import sampleMaxmin
from chemical.FingerPrint import Hash2FingerPrint
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:00:07 2019

@author: matsumoto
"""

def MakeTrain(dict_data, testid, exid=None):
    """
    devide from dictionary into training and test
    testid : select test data key
    """
    #n_data = max(dict_data.keys())
    dict_key = list(dict_data.keys())
    if exid is not None:
        valid_keys = [key for key in dict_key if not key in exid]
    else:
        valid_keys = dict_key
    concat_list = [i for i in valid_keys if i != testid]
    
    df_tr = pd.DataFrame()
    df_ts = pd.DataFrame()
    
    for i in concat_list:
        df_tr = df_tr.append(dict_data[i])
    df_ts = dict_data[testid]
        
    return df_tr, df_ts


def AllDataDict(path, n_data, index=None, sep=','):
    
    """
    read the all file in a folder and concatenate them as dictionary
    path : must be including number(i) because {}.format(i) is used in fanction
    """
    DF = dict()
    
    for i in range(1, n_data+1):
        df = pd.read_csv(path.format(i), index_col=index, sep=sep)
        DF[i] = df
        
    return DF


def HashAllDataDict(path, n_data, ic50col, smilecol, index=None, sep=','):
    """
    funtion is almost the same as ALLDataDict
    Furthermore, convert Smiles to Hash and concatenate with IC50 values
    """
    n_DF = dict()
    for i in range(1, n_data+1):
        df = pd.read_csv(path.format(i), index_col=index, sep=sep)
        ic50 = df[ic50col]
        smiles = df[smilecol]
        
        hsh = Smiles2Hash(smiles)
        n_df = pd.concat([ic50, hsh], axis=1)
        
        n_df.rename(columns={smilecol:"hash"}, inplace=True)
        n_DF[i] = n_df
        
    return n_DF


def tr_ts_split(data, idx_tr=None, train_size=None, train_num=None, random_state=None):
    """
    split into train and test including specified data in train
    if you include specified data in train, you specify its index using parameter idx_tr
    """
    if train_size is not None:
        sample = round(len(data) * train_size) - len([idx_tr])
        temp = data.drop(index = idx_tr)
        tr_temp = temp.sample(n = sample, random_state = random_state)
        
        tr = tr_temp.append(data.loc[idx_tr])
        ts = data.drop(index = tr.index)
        
    elif train_num is not None:
        temp = data.drop(index = idx_tr)
        tr_temp = temp.sample(n = train_num-len([idx_tr]), random_state = random_state)
        
        tr = tr_temp.append(data.loc[idx_tr])
        ts = data.drop(index = tr.index)
    
    return tr, ts


def tr_ts_split_ks(data, idx_tr=None, train_num=None, random_state=0, fpcol='hash'):
    """
    split into train and test including specified data in train based on kennerdstones
    if you include specified data in train, you specify its index using parameter idx_tr
    """
    datafp = Hash2FingerPrint(data[fpcol])
    temp = datafp.drop(index = idx_tr)
    n = train_num - len([idx_tr])
    
    idx = sampleMaxmin(x_org=temp, n_sample=n, seed=random_state)
    idx = list(idx)
    idx.append(idx_tr)
    
    tr = data.loc[idx]
    ts = data.drop(index = idx)
    
    return tr, ts

