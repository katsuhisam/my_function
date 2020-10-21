import os
import pandas as pd
import numpy as np
import pickle
import copy
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:37:26 2019

@author: matsumoto
"""

class CreateDataset:
    
    def __init__(self, tids, basefolder='/Users/matsumoto/Research/Research/Data/ChEMBL24/chemblx24-IC50/compounds'):
            self.tids = tids
            self.basefolder = basefolder
            self.dfs = {}
            self.target = {}
            self.findfs ={}
            self.all = {}
            self.gpassay = {}

            self._read_data()
            
            
    def _read_data(self):
    
        for tid in self.tids:
            
                df = pd.read_csv('{}/{}'.format(self.basefolder, tid),sep='\t', index_col=1)
                gp = df.groupby('assay_id')
                d = {}
                for asid, val in gp:
            #        if len(val) > 30:
                    d[asid] = val
                self.dfs[tid] = d
                
    
    def restrict_data_number(self, ndata):
        
        """
        Put a limit on the number of data in the assay.
        """
        
        new_dfs = copy.deepcopy(self.dfs)
        
        for key, dic in list(new_dfs.items()): # make other object
            for keyin, df in list(dic.items()):
                if df.shape[0] < ndata:
                    del dic[keyin]
            
            if len(dic) == 0:
                del new_dfs[key]
                print('{} is deleted'.format(key))
            else:
                print(key)
        
        return new_dfs
    
    
    def check_dupli_assays(self, nassays):
            
        for key, dic in self.dfs.items():
            cps = pd.DataFrame()
            for _, df in dic.items():
                cps = pd.concat([cps, df.iloc[:, :1].T], axis=0, ignore_index=True)
            
            cps.index=dic.keys()    
                
            reference = cps.loc[:, np.sum(cps.isna(), axis=0) < len(dic)-nassays]
            if reference.shape[1]>0:
                print(key)
                self.target[key] = reference
            else:
                print('-- %s is invalid --'%key)
                
                
    def mutual_comp_assays(self):
        
        for key, val in self.target.items():
            
            mut_comp = {}
            for col in val.columns:
               assays  = val[col].dropna().index.to_list()
               dict_assays = self.get_assays(key, assays)
               mut_comp[col] = dict_assays
            
            self.findfs[key] = mut_comp
            
            
    def get_assays(self, key, assays):
        
        new_dict = {}
        for assay in assays:
            new_dict[assay] = self.dfs[key][assay]
        
        return new_dict


    def gets_all(self):
        
        al = {}
        for key in self.findfs.keys():
            al[key] = pd.read_csv('{}/{}'.format(self.basefolder, key), sep='\t', index_col=0)
            
        self.all = al
    
    
    def group_assays(self):
        
        for key in self.findfs.keys():
            df = dfs[key]
            self.gpassay[key] = df
        
            
if __name__ == '__main__':
    tids = os.listdir('/Users/matsumoto/Research/Research/Data/ChEMBL24/chemblx24-IC50/compounds')
    tids = [f for f in tids if not '._' in f]
    
    cd = CreateDataset(tids)
    dfs = cd.dfs
    
    # dfss = cd.restrict_data_number(50)
    
    cd.check_dupli_assays(10) #There is an assay with at least () referece compound. 
    target = cd.target
    cd.mutual_comp_assays()
    findfs = cd.findfs
        
    cd.gets_all()
    al = cd.all
    
    cd.group_assays()
    gp = cd.gpassay
    
    # f = open('/Users/matsumoto/Research/Research/object/gp_assay_leastone.pickle','wb')
    # pickle.dump(gp,f)
    # f.close
        
    