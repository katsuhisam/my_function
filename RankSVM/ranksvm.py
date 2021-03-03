#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:52:28 2020

@author: matsumoto
"""

import subprocess
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.datasets import dump_svmlight_file
import pandas as pd
from scipy.stats import spearmanr, kendalltau


class SVM_Rank():
    
    def __init__(self, path=None):
        self.path = path
        self.file = '/Users/matsumoto/Research/Research/Rank-SVM'
        
        self.rlist = []


    def run_svmrank(self, inp_name='train', model_name='model_train', c=0.01):
        """
        inp : train or val
        """
        
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(model_name)), 'w+')
        
        inp_name = os.path.join(self.path, '{}.txt'.format(inp_name))
        model_name = os.path.join(self.path, '{}.txt'.format(model_name))
        
        input_line = './svm_rank_learn -c {cost} {inp_name} {model_name}'.format(cost=c, inp_name=inp_name, model_name=model_name)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
        
        
    def predict_svmrank(self, inpfile='train', modelfile='model_train', resultfile='predictions.train'):
        """
        inp   : train or test
        model : train or val
        """
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(resultfile)), 'w+')
        
        modelfile   = os.path.join(self.path, '{}.txt'.format(modelfile))
        inpfile     = os.path.join(self.path, '{}.txt'.format(inpfile))
        resultfile  = os.path.join(self.path, '{}.txt'.format(resultfile))

        input_line = './svm_rank_classify {inpfile} {modelfile} {resultfile}'.format(inpfile=inpfile, modelfile=modelfile, resultfile=resultfile)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
        
    
        

class SVM_Rank_CV():
    
    def __init__(self, path=None, ascending=True, method='dense', nf=2, random_state=0, shuffle=True, kernel='linear', eve_idx='kendalltau'):
        
        self.path = path
        self.file = '/Users/matsumoto/Research/Research/Rank-SVM'
        
        self.rlist      = []
        self.ascending  = ascending
        self.method     = method
        self.nf         = nf
        self.seed       = random_state
        self.shuffle    = shuffle
        self.kernel     = kernel
        self.eve_idx    = eve_idx
    
    
    # def fit_predict(self, inp_name=None, model_name=None, resultfile=None):
        
        
    #     self.run(inp_name=inp_name, model_name=model_name) # Running Rank_SVM
        
    #     self.predict(inpfile=inp_name, modelfile=model_name, resulfile=resultfile) # Prediction
    
    
    def put_svmlight(self, x, y, qid, fname):
        
        if fname:
            dump_svmlight_file(X=x, y=y, f=os.path.join(self.path, fname), query_id=qid, zero_based=False, multilabel=False)
        else:
            print('Error : File name is None!')
    
            
    def put_svmlight2(self, x, y, fname):
        
        if fname:
            dump_svmlight_file(X=x, y=y, f=os.path.join(self.path, fname), zero_based=False, multilabel=False)
        else:
            print('Error : File name is None!')
    
    
    def run(self, inp_name='train.txt', model_name='model_train.txt', c=0.01):
        """
        inp : train or val
        """
        
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(model_name.split('.')[0])), 'w+')
        
        inp_name = os.path.join(self.path, inp_name)
        model_name = os.path.join(self.path, model_name)
        
        if self.kernel == 'linear':
            input_line = './svm_rank_learn -c {cost} {inp_name} {model_name}'.format(cost=c, inp_name=inp_name, model_name=model_name)
        elif self.kernel == 'tanimoto':
            input_line = './svm_rank_learn -t 4 -c {cost} {inp_name} {model_name}'.format(cost=c, inp_name=inp_name, model_name=model_name)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
        
        
    def predict(self, inpfile='train.txt', modelfile='model_train.txt', resultfile='predictions_train.txt'):
        """
        inp   : train or test
        model : train or val
        """
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(resultfile.split('.')[0])), 'w+')
        
        modelfile   = os.path.join(self.path, modelfile)
        inpfile     = os.path.join(self.path, inpfile)
        resultfile  = os.path.join(self.path, resultfile)

        input_line = './svm_rank_classify {inpfile} {modelfile} {resultfile}'.format(inpfile=inpfile, modelfile=modelfile, resultfile=resultfile)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
    
    
    
    def CV_Rank_Loo(self, x_val, y_val, qid_val, x_train=None, y_train=None, qid_train=None):
    
        kf = LeaveOneOut()
        
        rankloo = pd.DataFrame()
        for cv_train_index, cv_test_index in kf.split(x_val):
            cv_x_train, cv_x_test       = x_val.iloc[cv_train_index,:], x_val.iloc[cv_test_index,:]
            cv_qid_train, cv_qid_test   = qid_val[cv_train_index], qid_val[cv_test_index]
            rank_cv_y_train = y_val[cv_train_index].rank(ascending=self.ascending, method=self.method)
            rank_cv_y_test = y_val[cv_test_index].rank(ascending=self.ascending, method=self.method)
            
            train_index = list(cv_x_train.index) # preserve cv_train index
            
            if isempty(x_train):
                rank_y_train = self.rank_labeling(y_train, qid_train)
            
                cv_qid_train    = pd.concat([cv_qid_train, qid_train], axis=0) 
                cv_x_train      = pd.concat([cv_x_train, x_train], axis=0)
                rank_cv_y_train = pd.concat([rank_cv_y_train, rank_y_train], axis=0)
            
                cv_qid_train    = cv_qid_train.sort_values()
                cv_x_train      = cv_x_train.reindex(index=cv_qid_train.index)
                rank_cv_y_train = rank_cv_y_train.reindex(cv_qid_train.index)
            
            self.put_svmlight(cv_x_train, rank_cv_y_train, qid=cv_qid_train, fname='train.txt')
            self.put_svmlight(cv_x_test,  rank_cv_y_test,  qid=cv_qid_test,  fname='test.txt')
            
            self.run(inp_name='train.txt', model_name='model_train.txt')
            self.predict(inpfile='train.txt', modelfile='model_train.txt', resultfile='predictions_train.txt')
            self.predict(inpfile='test.txt',  modelfile='model_train.txt', resultfile='predictions_test.txt')
            
            res_train = pd.read_csv(os.path.join(self.path, 'predictions_train.txt'), sep=' ', header=None)
            res_test  = pd.read_csv(os.path.join(self.path, 'predictions_test.txt'), sep=' ', header=None)
            
            res_train.index, res_test.index = cv_qid_train.index, cv_qid_test.index
            
            cv_train_score = res_train.query('index == @train_index')
            cv_test_score  = res_test
            
            rank_ts = self.give_rank_loo(cv_train_score, cv_test_score)
            rankloo = pd.concat([rankloo, rank_ts], axis=0)
        
        true_label = y_val.rank(ascending=self.ascending, method=self.method)
        pred_label = rankloo
        
        if self.eve_idx == 'spearman':
            correlation, pval = spearmanr(true_label, pred_label)
        elif self.eve_idx == 'kendalltau':
            correlation, pval = kendalltau(true_label, pred_label)
            
        self.rcv          = correlation
        
        
    # def CV_Rank_Nfold(self, x_val, y_val, qid_val, x_train=None, y_train=None, qid_train=None):
    
    #     kf = KFold(n_splits=self.nf, random_state=self.seed, shuffle=self.shuffle)
        
    #     qcv = []
    #     for cv_train_index, cv_test_index in kf.split(x_val):
    #         cv_x_train, cv_x_test       = x_val.iloc[cv_train_index,:], x_val.iloc[cv_test_index,:]
    #         cv_qid_train, cv_qid_test   = qid_val[cv_train_index], qid_val[cv_test_index]
    #         rank_cv_y_train = y_val[cv_train_index].rank(ascending=self.ascending, method=self.method)
    #         rank_cv_y_test = y_val[cv_test_index].rank(ascending=self.ascending, method=self.method)
            
    #         train_index = list(cv_x_train.index) # preserve cv_train index
            
    #         if isempty(x_train):
    #             rank_y_train = self.rank_labeling(y_train, qid_train)
            
    #             cv_qid_train    = pd.concat([cv_qid_train, qid_train], axis=0) 
    #             cv_x_train      = pd.concat([cv_x_train, x_train], axis=0)
    #             rank_cv_y_train = pd.concat([rank_cv_y_train, rank_y_train], axis=0)
            
    #             cv_qid_train    = cv_qid_train.sort_values()
    #             cv_x_train      = cv_x_train.reindex(index=cv_qid_train.index)
    #             rank_cv_y_train = rank_cv_y_train.reindex(cv_qid_train.index)
            
    #         self.put_svmlight(cv_x_train, rank_cv_y_train, qid=cv_qid_train, fname='train.txt')
    #         self.put_svmlight(cv_x_test,  rank_cv_y_test,  qid=cv_qid_test,  fname='test.txt')
            
    #         self.run(inp_name='train.txt', model_name='model_train.txt')
    #         self.predict(inpfile='train.txt', modelfile='model_train.txt', resultfile='predictions_train.txt')
    #         self.predict(inpfile='test.txt',  modelfile='model_train.txt', resultfile='predictions_test.txt')
            
    #         res_train = pd.read_csv(os.path.join(self.path, 'predictions_train.txt'), sep=' ', header=None)
    #         res_test  = pd.read_csv(os.path.join(self.path, 'predictions_test.txt'), sep=' ', header=None)
            
    #         res_train.index, res_test.index = cv_qid_train.index, cv_qid_test.index
            
    #         cv_train_score = res_train.query('index == @train_index')
    #         cv_test_score  = res_test
            
    #         true_label = rank_cv_y_test
    #         pred_label = cv_test_score.rank(ascending=self.ascending, method=self.method)
        
    #         correlation, pval = spearmanr(true_label, pred_label)
    #         qcv.append(correlation)
            
    #     self.rcv          = sum(qcv) / len(qcv) #average

    
    def rank_labeling(self, y_train, qid_train):
        
        ys_ranks = pd.DataFrame()
        
        kinds = set(qid_train)
        qid_train = pd.DataFrame(qid_train)
        y_train   = pd.DataFrame(y_train)
        
        for kind in kinds:
            d_kind = qid_train.query('Assay == @kind')
            d_kind_index = list(d_kind.index)
            
            ys = y_train.query('index == @d_kind_index')
            ys_rank = ys.rank(ascending=self.ascending, method=self.method)
            
            ys_ranks = pd.concat([ys_ranks, ys_rank], axis=0)
        ys_ranks = ys_ranks.iloc[:,0]
        
        return ys_ranks
    
    
    def give_rank_loo(self, cv_train_score, cv_test_score):
        
        ts_idx = list(cv_test_score.index)
        
        tr_ts_score = pd.concat([cv_train_score, cv_test_score], axis=0)
        ranks = tr_ts_score.rank(ascending=self.ascending, method=self.method)
        
        rank_test = ranks.query('index == @ts_idx')
        
        return rank_test


def result_writing(filep, res):
    """
    preserving the output in command line
    """
    for line in res.stdout.splitlines():
        filep.write(line.decode('utf-8'))
        filep.write('\n')
        
        
def isempty(obj):
    if (obj is not None) and (obj.any().any()):
        return True
    else:
        return False
    
    
    # def CV_Rank(self, x_val, y_val, qid_val, x_train=None, y_train=None, qid_train=None, nf=5, seed=0, shuffle=True):
    
    #     kf = KFold(n_splits=nf, random_state=seed, shuffle=shuffle)
        
    #     for cv_train_index, cv_test_index in kf.split(x_val):
    #         cv_x_train, cv_x_test       = x_val.iloc[cv_train_index,:], x_val.iloc[cv_test_index,:]
    #         cv_qid_train, cv_qid_test   = qid_val[cv_train_index], qid_val[cv_test_index]
            
    #         rank_cv_y_train = y_val[cv_train_index].rank(ascending=self.acending, method=self.method)
    #         rank_cv_y_test  = y_val[cv_test_index].rank(ascending=self.acending, method=self.method)
            
            
    #         if x_train is not None:    
    #             rank_y_train = self.rank_labeling(y_train, qid_train)
            
    #             cv_x_train = pd.concat([cv_x_train, x_train], axis=0)
    #             rank_cv_y_train = pd.concat([rank_cv_y_train, rank_y_train], axis=0)
    #             cv_qid_train = pd.concat([cv_qid_train, qid_train], axis=0)    
            
    #         self.put_svmlight(cv_x_train, rank_cv_y_train, qid=cv_qid_train, fname='train.txt')
    #         self.put_svmlight(cv_x_test,  rank_cv_y_test,  qid=cv_qid_test,  fname='test.txt')
            
    #         self.run(inp_name='train.txt', model_name='model_train.txt')
            
    #         self.predict(inpfile='test.txt', modelfile='model_train.txt', resultfile='predictions_test.txt')
            
    #         res = pd.read_csv(os.path.join(self.path, 'predictions_test.txt'), sep=' ', header=None)
    #         pred_rank_cv_y_test = res.rank(ascending=True, method='dense')
            
    #         correlation, pval = spearmanr(rank_cv_y_test, pred_rank_cv_y_test)
    #         self.rlist.append(correlation)
            
    #     self.rcv = sum(self.rlist) / len(self.rlist)
    
