# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:20 2019

@author: aoshima
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:00:06 2019

@author: matsumoto
"""
import pandas as pd
from sklearn import preprocessing as pp




class EachAssayScaler:
    """
    Scaler for integrating different assays.
    """
    def __init__(self, kind='standard'):
        self.scalers = {}
        self.kind   = kind
        
        self._set_scaler()
        
        
    def _set_scaler(self):
        if self.kind=='standard':
            return pp.StandardScaler()
        elif self.kind=='range':
            return pp.MinMaxScaler()
        else:
            ValueError('%s is invalid scaler'%self.kind)
        
        
    def fit_scaler(self, dict_data, yname):
        """
        calculate mean and variance of each assay data
        --- using for training data ---
        """
        data_sc = {}
        
        for key, df in dict_data.items():
            # define scaler
            scaler = self._set_scaler()
            df_copy = df.copy()
            # make y columns name
            ycol = '{}-{}'.format(yname, key)
            # scaling
            ytrs = scaler.fit_transform(df_copy[ycol].values.reshape(-1,1))
            df_copy[ycol] = ytrs
            # save scaled data into dictionary
            data_sc[key] = df_copy
            self.scalers[key] = scaler
            
        return data_sc
        
        
    def transform(self, dict_data, yname):
        """
        transform test data based on the mean and variance of training data.
        --- using for test data ---
        """
        data_sc = {}
        
        for key, df in dict_data.items():
            # define scaler
            scaler = self.scalers[key]
            df_copy = df.copy()
            # make y columns name
            ycol = '{}-{}'.format(yname, key)
            # scaling test data based on training data
            ytrs = scaler.transform(df_copy[ycol].values.reshape(-1,1))
            df_copy[ycol] = ytrs
            # save scaled data into dictionary
            data_sc[key] = df_copy
            
        return data_sc
    
    
    def inverse_transform(self, dict_data, yname):
        """
        inverse transform based on the mean and variance of training data.
        --- all data ---
        """
        data_sc = {}
        
        for key, df in dict_data.items():
            # define scaler
            scaler = self.scalers[key]
            df_copy = df.copy()
            # make y columns name
            ycol = '{}-{}'.format(yname, key)
            # restore data to original scale
            ytr = scaler.inverse_transform(df_copy[ycol].values.reshape(-1,1))
            df_copy[ycol] = ytr
            # save scaled data into dictionary
            data_sc[key] = df_copy
            
        return data_sc
    