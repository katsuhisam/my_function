import numpy as np
import pandas as pd
from sklearn import metrics
from evaluation.criteria import r2_rmse_mae
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn import preprocessing as pp
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV as gcv
from sklearn.pipeline import Pipeline
from kernels.mykernel import funcTanimotoKernel, funcTanimotoSklearn
from util.utility import applyfunc_with_batch_mt, applyfunc_with_batch

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:13:01 2019

@author: matsumoto
"""


class KernelRidge_CV():
    """
    cross validation version. wrapper class of KernelRidge
    """
    
    def __init__(self, rseed=0, nf=5, paramset=None, verbose=False, kernelf='linear', is_scaling=False, selectedScore=None):
        self.nf         = nf
        self.verbose    = verbose
        self.rng        = np.random.RandomState(rseed)
        
        self.metric, self.kernelf, self.paramset, self.is_scaling = \
                            self._set_conditions(selectedScore, kernelf, paramset, is_scaling)

        if self.is_scaling: # if Tanimoto kernel is used this part is skipped 
            self.xscaler = pp.MinMaxScaler(feature_range=(-1, 1))
            self.yscaler = pp.MinMaxScaler(feature_range=(-1, 1))
        
        self.model = self._set_model(self.kernelf, self.paramset, self.rng, self.nf, self.metric)        
        

    def _set_conditions(self, selectedScore, kernelf, paramset, is_scaling):
        """
        Setting calculation conditions 
        """
        if selectedScore is None:
            metric = 'neg_mean_absolute_error'
        else:
            metric = selectedScore
        
        kernelf = kernelf.lower()
        if kernelf not in ['tanimoto', 'linear']:
            ValueError('Kernel must be either Linear and Tanimoto')
            exit(1)

        if paramset is None:
            paramset = dict(alpha=np.logspace(-3, 2, num=10, base=10))
        
        if kernelf == 'tanimoto':
            print('skip the scaling due to binary kernel')
            is_scaling = False

        return metric, kernelf, paramset, is_scaling


    def _set_model(self, kernelf, paramset, rng, nf, selectedScore):
        """
        Setting the models with parameters
        """
        if nf == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(nf, shuffle=True, random_state=rng)

        if kernelf == 'tanimoto':
            return gcv(KernelRidge(kernel=funcTanimotoSklearn), param_grid=paramset,
                        cv=cv, scoring=selectedScore, n_jobs=-1)
        elif kernelf == 'linear':
            return gcv(KernelRidge(), param_grid=paramset, 
                        cv=cv, scoring=selectedScore, n_jobs=-1)
            

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
        if self.kernelf == 'tanimoto':
            self.params = dict(kernelf=self.kernelf, alpha=self.model.best_estimator_.alpha)
            #self.pmodel = NuSVR(kernel=funcTanimotoKernel, nu=self.params['nu'], C=self.params['C'])
            self.pmodel = KernelRidge(kernel=funcTanimotoSklearn, alpha=self.params['alpha'])
        # prediction by the best model
        self.pmodel.fit(xs, ys.ravel())

        YptrS = self.pmodel.predict(xs)
        if self.is_scaling:
            Yptr = self.yscaler.inverse_transform(YptrS.reshape(-1, 1))
        else:
            Yptr = YptrS

        self.evaluate_r2, self.evaluate_rmse, self.evaluate_mae = r2_rmse_mae(yp=Yptr, yobs=y, verbose=self.verbose)


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

    
    
    

if __name__ == '__main__':
    
    
    svr = KernelRidge(kernelf='tanimoto')
    
    svr.fit(x, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
















    