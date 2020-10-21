# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:43:58 2019

@author: aoshima
"""
import math
import numpy as np
import pandas as pd
import ast
from preprocess.variable_selection import ManageScaler
from evaluation.criteria import r2_rmse_mae
from sklearn import metrics
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn import preprocessing as pp
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV as gcv
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from kernels.mykernel import funcTanimotoKernel, funcTanimotoSklearn
from datatype.returntype import  cv_returnData
from util.utility import applyfunc_with_batch_mt, applyfunc_with_batch, ProductDict


class NuSVR_validate():
    """
    cross validation version. wrapper class of Nu SVR
    """
    
    def __init__(self, rseed=0, verbose=False, kernelf='rbf', is_scaling=False):
        self.verbose    = verbose
        self.rng        = np.random.RandomState(rseed)
        self.kernelf    = kernelf 
        self.is_scaling = self._set_conditions(kernelf, is_scaling)

        if self.is_scaling: # if Tanimoto kernel is used this part is skipped 
            self.xscaler = pp.MinMaxScaler(feature_range=(-1, 1))
            self.yscaler = pp.MinMaxScaler(feature_range=(-1, 1))
        
        self.model = self._set_model(self.kernelf)    
        

    def _set_conditions(self, kernelf, is_scaling):
        """
        Setting calculation conditions 
        """
        kernelf = kernelf.lower()
        if kernelf not in ['tanimoto', 'rbf']:
            ValueError('Kernel must be either RBF and Tanimoto')
            exit(1)

        if kernelf == 'tanimoto':
            print('skip the scaling due to binary kernel')
            is_scaling = False

        return is_scaling


    def _set_model(self, kernelf):
        """
        Setting the models with parameters
        """

        if kernelf == 'tanimoto':
            return NuSVR(kernel=funcTanimotoKernel)
                        
        elif kernelf == 'rbf':
            return NuSVR()

    def fit(self, x, y, weights=None):
        """
        Fit the cv model with the x and y
        """
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1,1)

        if self.is_scaling:
            xs = self.xscaler.fit_transform(x)
            ys = self.yscaler.fit_transform(y)
        else:
            xs = x
            ys = y

        if weights is None:
            self.model.fit(xs, ys.ravel())
        else:
            self.model.fit(xs, ys.ravel(), sample_weight=weights)


    def predict(self, x):
        """
        Predict y values for x

        Note: there is no parallellization here (apply_with_batch function)
        """
        x = np.array(x)
        
        if self.is_scaling:
            sx = self.xscaler.transform(x)
        else:
            sx = x
        
        if len(x) > 10000:
            #spy = applyfunc_with_batch_mt(self.pmodel.predict, sx, nworkers=2)
            spy = applyfunc_with_batch(self.model.predict, sx, batchsize=70000)
            #spy = self.pmodel.predict(sx)
        else:
            spy = self.model.predict(sx)

        if self.is_scaling:
            py = self.yscaler.inverse_transform(spy.reshape(-1,1))
        else:
            py = spy.reshape(-1,1)
        return py

    def predict_vals(self, xtrain, xtest):
        """
        predict multiple y values for mulple xs 
        """
        py1 = self.predict(xtrain)
        py2 = self.predict(xtest)
        return py1, py2

    
class NuSVRBase():
    def __init__(self, verbose, selectedScore, kernelf, paramset, is_scaling):
        self.verbose    = verbose
        self.metric, self.kernelf, self.paramset, self.is_scaling = \
                            self._set_conditions(selectedScore, kernelf, paramset, is_scaling)

        if self.is_scaling: # if Tanimoto kernel is used this part is skipped 
            self.xscaler = pp.MinMaxScaler(feature_range=(-1, 1))
            self.yscaler = pp.MinMaxScaler(feature_range=(-1, 1))
        

    def _set_conditions(self, selectedScore, kernelf, paramset, is_scaling):
        """
        Setting calculation conditions 
        """
        if selectedScore is None:
            metric = 'neg_mean_absolute_error'
        else:
            metric = selectedScore
        
        kernelf = kernelf.lower()
        if kernelf not in ['tanimoto', 'rbf']:
            ValueError('Kernel must be either RBF and Tanimoto')
            exit(1)

        if paramset is None:
            if kernelf == 'rbf':
                paramset = dict(gamma=np.logspace(-3, 2, num=5), 
                                nu=np.logspace(-3, 0, num=5, endpoint=False),
                                C=np.logspace(-3, 2, num=5, base=10))
            else:
                paramset = dict(nu=np.logspace(-4, 0, num=10, endpoint=False, base=10),
                                C=np.logspace(-3, 2, num=10, base=10))
        
        if kernelf == 'tanimoto':
            if self.verbose:
                print('skip the scaling due to binary kernel')
            is_scaling = False

        return metric, kernelf, paramset, is_scaling
    
    def get_params(self):
        return self.params


    def predict(self, x):
        """
        Predict y values for x

        Note: there is no parallellization here (apply_with_batch function)
        """
        x = np.array(x)
        
        if self.is_scaling:
            sx = self.xscaler.transform(x)
        else:
            sx = x
        
        if len(x) > 10000:
            #spy = applyfunc_with_batch_mt(self.pmodel.predict, sx, nworkers=2)
            spy = applyfunc_with_batch(self.pmodel.predict, sx, batchsize=70000)
            #spy = self.pmodel.predict(sx)
        else:
            spy = self.pmodel.predict(sx)

        if self.is_scaling:
            py = self.yscaler.inverse_transform(spy.reshape(-1,1))
        else:
            py = spy.reshape(-1,1)
        return py

    def predict_vals(self, xtrain, xtest):
        """
        predict multiple y values for mulple xs 
        """
        py1 = self.predict(xtrain)
        py2 = self.predict(xtest)
        return py1, py2


class NuSVR_DCV(NuSVRBase):
    """
    Double (Nested) cross validation version. wrapper class of Nu SVR
    """
    
    def __init__(self, rseed=0, nfin=5, nfout=5, paramset=None, verbose=False, kernelf='rbf', is_scaling=False, selectedScore=None):
        super(NuSVR_DCV, self).__init__(verbose, selectedScore, kernelf, paramset, is_scaling)
        self.nfin       = nfin
        self.nfout      = nfout
        self.rng        = np.random.RandomState(rseed)
        self.model      = self._set_model(kernelf, self.paramset, self.rng, self.nfin, selectedScore) # not determined yet...
    

    def _set_model(self, kernelf, paramset, rng, nf, selectedScore):
        """
        Setting the models with parameters
        """
        if nf == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(nf, shuffle=True, random_state=rng)

        if selectedScore is None:
            selectedScore = 'neg_mean_absolute_error'

        if kernelf == 'tanimoto':
            return gcv(NuSVR(kernel=funcTanimotoKernel), param_grid=paramset,
                        cv=cv, scoring=selectedScore, n_jobs=-1)
        elif kernelf == 'rbf':
            return gcv(NuSVR(), param_grid=paramset, 
                        cv=cv, scoring=selectedScore, n_jobs=1)
        
    
    def fit(self, x, y, weights=None):
        """
        Fit the cv model with the x and y
        """
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1,1)

        if self.is_scaling:
            xs = self.xscaler.fit_transform(x)
            ys = self.yscaler.fit_transform(y)
        else:
            xs = x
            ys = y

        if weights is None:
            self.model.fit(xs, ys.ravel())
        else:
            self.model.fit(xs, ys.ravel(), sample_weight=weights)

        # optimized parameters
        if self.kernelf == 'rbf':
            self.params = dict(kernelf=self.kernelf, gamma=self.model.best_estimator_.gamma, nu=self.model.best_estimator_.nu, C=self.model.best_estimator_.C)
            self.pmodel = NuSVR(nu=self.params['nu'], C=self.params['C'], gamma=self.params['gamma'])
        elif self.kernelf == 'tanimoto':
            self.params = dict(kernelf=self.kernelf, C=self.model.best_estimator_.C, nu=self.model.best_estimator_.nu)
            #self.pmodel = NuSVR(kernel=funcTanimotoKernel, nu=self.params['nu'], C=self.params['C'])
            self.pmodel = NuSVR(kernel=funcTanimotoSklearn, nu=self.params['nu'], C=self.params['C'])
        # prediction by the best model
        self.pmodel.fit(xs, ys.ravel())

        YptrS = self.pmodel.predict(xs)
        if self.is_scaling:
            self.Yptr = self.yscaler.inverse_transform(YptrS.reshape(-1, 1))
        else:
            self.Yptr = YptrS

        self.evaluate_r2, self.evaluate_rmse, self.evaluate_mae = r2_rmse_mae(yp=self.Yptr, yobs=y, verbose=self.verbose)
        
        # nested cross validation score
        scores = cross_val_score(self.model, X=x, y=y, cv=KFold(self.nfout, shuffle=True, random_state=self.rng))
        self.dcv_score = scores.mean()
        print(self.dcv_score)

    

if __name__ == '__main__':
    
    
    svr = NuSVR_validate(kernelf='tanimoto')
    
    svr.fit(x, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    















