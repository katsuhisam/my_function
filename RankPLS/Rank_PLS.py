# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:39:14 2020

@author: matsumoto
"""
import numpy as np
import pandas as pd
import sys

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from evaluation.criteria import r2_rmse_mae

from PLS.plswrappers import PLS_CV
from scipy.stats import spearmanr


def CV_RankPLS_Loo(x_val, y_val, x_train=None, y_train=None, integration=True):
    
    cv_x_train = x_val
    cv_y_train = y_val
    
    if isempty(x_train):
        cv_x_train = pd.concat([x_val, x_train], axis=0)
        cv_y_train = pd.concat([y_val, y_train], axis=0)
    
    pls         = PLS_CV(nf='loo')
    pls.fit(cv_x_train, cv_y_train)
    cv_py_train = pls.predict(cv_x_train)
    cv_py_train = pd.DataFrame(cv_py_train, index=cv_y_train.index)
    
    ys = pd.concat([y_val, cv_py_train], axis=1)
    ys = ys.dropna()
    
    true_label = ys.iloc[:,0].rank(ascending=True, method='dense')
    pred_label = ys.iloc[:,1].rank(ascending=True, method='dense')
    correlation, pval = spearmanr(true_label, pred_label)
    rcv = correlation
    
    print('||| Cross Validation is finished! |||')
    
    return rcv

            
def CV_RankPLS_Loo2(x_val, y_val, x_train=None, y_train=None):
    
    cv = LeaveOneOut()
    
    score = pd.DataFrame()
    # cv main
    for cv_train_index, cv_test_index in cv.split(x_val):
        cv_x_train, cv_x_test = x_val.iloc[cv_train_index,:], x_val.iloc[cv_test_index,:]
        cv_y_train, cv_y_test = y_val[cv_train_index], y_val[cv_test_index]
        
        if isempty(x_train):
            cv_x_train = pd.concat([cv_x_train, x_train], axis=0)
            cv_y_train = pd.concat([cv_y_train, y_train], axis=0)
            
        pls = PLSRegression()
        pls.fit(cv_x_train, cv_y_train)
        cv_pred_y_test = pls.predict(cv_x_test)
        cv_pred_y_test = pd.DataFrame(cv_pred_y_test)
        score = pd.concat([score, cv_pred_y_test], axis=0)
    
    true_label = y_val.rank(ascending=True, method='dense')
    pred_label = score.rank(ascending=True, method='dense')
    correlation, pval = spearmanr(true_label, pred_label)
    rcv = correlation
    
    print('||| Cross Validation is finished! |||')
    
    return rcv


def CV_RankPLS_Loo3(x_val, y_val, x_train=None, y_train=None):
    
    """
    Same format as LOO in rank learning
    """
    
    cv = LeaveOneOut()
    
    rankloo = pd.DataFrame()
    # cv main
    for cv_train_index, cv_test_index in cv.split(x_val):
        cv_x_train, cv_x_test = x_val.iloc[cv_train_index,:], x_val.iloc[cv_test_index,:]
        cv_y_train, cv_y_test = y_val[cv_train_index], y_val[cv_test_index]
        
        train_index = list(cv_x_train.index)
        
        if isempty(x_train):
            cv_x_train = pd.concat([cv_x_train, x_train], axis=0)
            cv_y_train = pd.concat([cv_y_train, y_train], axis=0)
            
        pls = PLSRegression()
        pls.fit(cv_x_train, cv_y_train)
        cv_pred_y_train = pd.DataFrame(pls.predict(cv_x_train), index=cv_x_train.index)
        cv_pred_y_test  = pd.DataFrame(pls.predict(cv_x_test),  index=cv_y_test.index)
        
        cv_train_score = cv_pred_y_train.query('index == @train_index')
        cv_test_score  = cv_pred_y_test
        
        rank_ts = give_rank_loo(cv_train_score, cv_test_score)
        rankloo = pd.concat([rankloo, rank_ts], axis=0)
        
    true_label = y_val.rank(ascending=True, method='dense')
    pred_label = rankloo
    
    correlation, pval = spearmanr(true_label, pred_label)
    rcv = correlation
    
    print('||| Cross Validation is finished! |||')
    
    return rcv


def give_rank_loo(cv_train_score, cv_test_score):
    
    ts_idx = list(cv_test_score.index)
    
    tr_ts_score = pd.concat([cv_train_score, cv_test_score], axis=0)
    ranks = tr_ts_score.rank(ascending=True, method='dense')
    
    rank_test = ranks.query('index == @ts_idx')
    
    return rank_test


def isempty(obj):
    if (obj is not None) and (obj.any().any()):
        return True
    else:
        return False
    