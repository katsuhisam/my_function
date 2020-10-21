import pandas as pd
from sklearn.decomposition import PCA

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:43:05 2020

@author: matsumoto
"""

# for mol2vec DataFrame
class PCA_run():
    
    def __init__(self, df, n_component=100, not_x_col=-4):
        
        self._pca_run(df, n_component, not_x_col)
        
    def _pca_run(self, df, n_component, not_x_col):
        x = df.iloc[:,:not_x_col]
        y = df.iloc[:,not_x_col:]
        
        self.pca = PCA(n_component)
        xp = self.pca.fit_transform(x)
        xp = pd.DataFrame(xp, index=x.index)
        
        self.dfp = pd.concat([xp, y], axis=1)
        
        
#class pca_m2v():
#    
#    def __init__(self, n_component=5, not_x_col=-1):
#        self.n_component = n_component
#        self.not_x_col = not_x_col
#        
#    def fit_trasform(self, x):
#        self.pca_model = PCA(self.n_component)
#        xs = self.pca_model.fit_transform(x.iloc[:,:self.not_x_col])
#        xs = pd.DataFrame(xs, index=x.index)
#        
#        xs = pd.concat([xs, x.iloc[:,self.not_x_col]], axis=1)
#        return xs
#    
#    def trasform(self, x):
#        xs = self.pca_model.transform(x)
#        return xs