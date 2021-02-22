#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:49:19 2020

@author: matsumoto
"""

import pickle
import pandas as pd
import numpy as np
from svm.svm_light import makePDtableFromSVMinput
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from chemical.FingerPrint import Smiles2FingerPrint
from chemical.FingerPrint import Hash2FingerPrint
#from Dataset.DataPreparation import tr_ts_split


def rank_file(value, rank_col_name='pot.(log,IC50)', ascending=True, method='dense'):
    """
    make dataFrame 
    including Rank column
    """
    r = value.rank(ascending=ascending, method=method)
    value['rank'] = r[rank_col_name]
    
    return value


def put_svmlight(value, rankcol='rank', qid='Assay', fname=None, hsh=False, smiles=None):
    """
    make SVMlight File
    """
    # for Mol2Vec descriptor 
    # x = value.loc[:, value.columns.str.contains('raidus')]
    # for ECFP4 descriptor
    if smiles != None:
        x = Smiles2FingerPrint(value[smiles])
    elif hsh:
        x = Hash2FingerPrint(value.hash)
    else:
        print('Input Error!')
    
    y = value[rankcol]
    qid = value[qid]
    
    if fname:
        dump_svmlight_file(X=x, y=y, f=fname, query_id=qid, zero_based=False, multilabel=False)
    else:
        print('Error : File name is None!')
        

def tr_ts_rank(data, ascending=True, method='dense'):
    rank_d = pd.DataFrame()
    
    assay = set(data['Assay'])
    
    for asy in assay:
        d = data.query('Assay == @asy')
        d_r = rank_file(d, ascending=ascending, method=method)
        
        rank_d = pd.concat([rank_d, d_r])
    return rank_d


