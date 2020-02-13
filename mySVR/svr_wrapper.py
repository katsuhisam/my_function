# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:43:58 2019

@author: aoshima
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing as pp
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV as gcv
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from kernels.mykernel import funcTanimotoKernel, funcTanimotoSklearn
from util.utility import applyfunc_with_batch_mt, applyfunc_with_batch


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

    
    
    

if __name__ == '__main__':
    
    
    svr = NuSVR_validate(kernelf='tanimoto')
    
    svr.fit(x, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    















