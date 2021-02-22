#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:09:50 2020

@author: matsumoto
"""
import numpy as np
import pandas as pd
from chemical.FingerPrint import Hash2FingerPrint
from chem.fingerprints import hash2bits_pd


def Hash2FingerPrintPlusAsyInfo(df_including_hash):
    """
    Add assay information to fingerprints.
    information : min, 25%, 50%, 75%, max
    """
    fp = Hash2FingerPrint(df_including_hash.hash)
    
    assays = list(set(df_including_hash.Assay))
    
    des_all = pd.DataFrame()
    for assay in assays:
        line = df_including_hash.query('Assay == @assay')
        
        activ  = line['pot.(log,IC50)']
        des    = np.array(_describe_calc(activ))
        des    = np.tile(des, (len(line), 1))
        des_df = pd.DataFrame(des, index=line.index)
        
        des_all = pd.concat([des_all, des_df], axis=0)
    
    fp_fin = pd.concat([fp, des_all], axis=1)
    fp_fin.columns = np.arange(0, fp_fin.shape[1])
    
    return fp_fin

def Hash2FingerPrintPlusAsyInfo_ForTest(df_including_hash_test, df_including_hash_val):
    
    """
    """
    fp_test = Hash2FingerPrint(df_including_hash_test.hash)
    
    activ  = df_including_hash_val['pot.(log,IC50)']
    des    = np.array(_describe_calc(activ))
    des    = np.tile(des, (len(df_including_hash_test), 1))
    des_df = pd.DataFrame(des, index=df_including_hash_test.index)
        
    fp_test_fin = pd.concat([fp_test, des_df], axis=1)
    fp_test_fin.columns = np.arange(0, fp_test_fin.shape[1])
    
    return fp_test_fin
        

def _describe_calc(value_series):
    
    des = value_series.quantile(q=[0, 0.25, 0.5, 0.75, 1])
    des = list(des)
    
    return des
    

# def Hash2FingerPrintSlash(hash_series, sln=20):
#     """
#     convert hash_series to ECFP4 2048bits
#     """
#     bits = hash2bits_pd(hash_series).replace({True:1, False:0})
#     bits = bits.iloc[:, :sln]
    
#     return bits


# def Hash2FingerPrintPlusAsyInfo2(df_including_hash, sln=20):
#     """
#     Add assay information to fingerprints.
#     information : min, 25%, 50%, 75%, max
#     """
#     fp = Hash2FingerPrintSlash(df_including_hash.hash, sln=sln)
    
#     assays = list(set(df_including_hash.Assay))
    
#     des_all = pd.DataFrame()
#     for assay in assays:
#         line = df_including_hash.query('Assay == @assay')
        
#         activ  = line['pot.(log,IC50)']
#         des    = np.array(_describe_calc(activ))
#         des    = np.tile(des, (len(line), 1))
#         des_df = pd.DataFrame(des, index=line.index)
        
#         des_all = pd.concat([des_all, des_df], axis=0)
    
#     fp_fin = pd.concat([fp, des_all], axis=1)
#     fp_fin.columns = np.arange(0, fp_fin.shape[1])
    
#     return fp_fin

# def Hash2FingerPrintPlusAsyInfo_ForTest2(df_including_hash_test, df_including_hash_val, sln=20):
    
#     """
#     """
#     fp_test = Hash2FingerPrintSlash(df_including_hash_test.hash, sln=sln)
    
#     activ  = df_including_hash_val['pot.(log,IC50)']
#     des    = np.array(_describe_calc(activ))
#     des    = np.tile(des, (len(df_including_hash_test), 1))
#     des_df = pd.DataFrame(des, index=df_including_hash_test.index)
        
#     fp_test_fin = pd.concat([fp_test, des_df], axis=1)
    
#     return fp_test_fin