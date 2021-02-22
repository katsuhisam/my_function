import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from svr.svr_wrapper import NuSVR_CV
from evaluation.criteria import r2_rmse_mae
from chemical.FingerPrint import Smiles2Hash, Hash2FingerPrint
from chemical.FingerPrintChild import Hash2FingerPrintPlusAsyInfo
from Dataset.DataPreparation import MakeTrain, tr_ts_split_ks
from chem.mol2vec import Mol2Vec

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:46:52 2020

@author: matsumoto
"""

class AssayPrep:
    """
    Prepare test assay and train assay data
    Test is the assay which has the most compounds
    """
    
    def __init__(self, dfs, target, dupcomp, ndata, radius=1, 
                 radius_path='/Users/matsumoto/Research/Research/Mol2Vec/mol2vec'):
        """
        target : protein target (ex. tid-9-actives.txt)
        dupcomp : duplicate compound between assay
        ndata : used assay that has more than n compounds
        radius, radius_path : for making mol2vec descriptor
        """
        self.dfs = dfs
        self.target = target
        self.dupcomp = dupcomp
        self.tes_key = ''
        self.all_dict = {}
        self.all_dict_sc = {}
        self.tr_dict = {}
        self.tr_dict_sc = {}
        self.ratios = {}
        self.radius = radius
        self.radius_path = radius_path
        self.sc_dicts = {}
    
        self._ndata_assays(ndata)
        self._hash_from_smiles()
        # self._mol2vec_from_smiles()
        self._test_assign()
        self._assaycol_assign()
        self._sc_ratio()
        self._sc_ratio_indiv()
        
    
    def _ndata_assays(self, ndata):
    
        tag = self.dfs[self.target]
        comps = tag[self.dupcomp]
        
        for comp, val in comps.items():
            if comps[comp].shape[0] > ndata:
                self.all_dict[comp] = val
                
                
    def _hash_from_smiles(self):
        
        for nval in self.all_dict.values():
            nval['hash'] = Smiles2Hash(nval['non_stereo_aromatic_smieles'])
    
    
    def _mol2vec_from_smiles(self):
        """
        convert smiles into mol2vec
        """
        radius = self.radius
        if radius == 1:
            mol2vec_model = self.radius_path + '/mol2vec_hash_radius1_chembl_minCount1.bin'
        else:
            mol2vec_model = self.radius_path + '/mol2vec_hash_radius2_chembl_minCount1.bin'
            
        for key, nval in self.all_dict.items():
            vals = Mol2Vec.GetRepresentation(nval['non_stereo_aromatic_smieles'], mol2vec_model, radius=self.radius)
            nval = pd.concat([nval, vals], axis=1)
            self.all_dict[key] = nval
            
    
    def _test_assign(self):
        """
        adopt the assay that has the most compounds
        """
        tes_shape = 0
        
        for key, val in self.all_dict.items():
            if val.shape[0] > tes_shape:
                tes_shape = val.shape[0]
                self.tes_key = key
                
    def _assaycol_assign(self):
        """
        assay number column is attached
        """
        for key in self.all_dict.keys():
            self.all_dict[key]['Assay'] = key

    
    def test_split(self, scale=False, idx_tr=None, train_size=None, train_num=None, random_state=None):
        """
        take test data from all data.
        make dictionary that includes only train data 
        
        idx_tr : compound that have 'index(idx_tr)' is included in train data
        train_size : specify train size (%)
        train_num : specify train number
        
        trofts : train data of test assay
        tsofts : test data of test assay
        
        """
        if scale == False:
            all_dict = self.all_dict
        elif scale == True:
            all_dict = self.all_dict_sc
        
        temp_train, temp_test = MakeTrain(all_dict, self.tes_key)
        
        if train_size is not None:
            sample = round(len(temp_test) * train_size) - len([idx_tr])
            temp = temp_test.drop(index = idx_tr)
            tr_temp = temp.sample(n = sample, random_state = random_state)
            
            trofts = tr_temp.append(temp_test.loc[idx_tr])
            tsofts = temp_test.drop(index = trofts.index)
            
        elif train_num is not None:
            temp = temp_test.drop(index = idx_tr)
            tr_temp = temp.sample(n=train_num-len([idx_tr]), random_state = random_state)
            
            trofts = tr_temp.append(temp_test.loc[idx_tr])
            tsofts = temp_test.drop(index = trofts.index)
        
            
        #make train data dictionary
        tr_dict = all_dict.copy()
        for k in tr_dict.keys():
            tr_dict[k] = tr_dict[k].copy()
        
        tr_dict[self.tes_key] = trofts.copy()
        
        if scale == False:
            self.tr_dict = tr_dict
        elif scale == True:
            self.tr_dict_sc = tr_dict
        
        return trofts, tsofts
    
    
    def test_split_ks(self, scale=False, idx_tr=None, train_num=None, random_state=None, fpcol='hash'):
        """
        take test data from all data.
        make dictionary that includes only train data 
        
        idx_tr : compound that have 'index(idx_tr)' is included in train data
        train_size : specify train size (%)
        train_num : specify train number
        
        trofts : train data of test assay
        tsofts : test data of test assay
        
        """
        if scale == False:
            all_dict = self.all_dict
        elif scale == True:
            all_dict = self.all_dict_sc
        
        temp_train, temp_test = MakeTrain(all_dict, self.tes_key)
            
        trofts, tsofts = tr_ts_split_ks(temp_test, idx_tr=idx_tr, train_num=train_num, random_state=random_state, fpcol=fpcol)
        
        #make train data dictionary
        tr_dict = all_dict.copy()
        for k in tr_dict.keys():
            tr_dict[k] = tr_dict[k].copy()
        
        tr_dict[self.tes_key] = trofts.copy()
        
        if scale == False:
            self.tr_dict = tr_dict
        elif scale == True:
            self.tr_dict_sc = tr_dict
        
        return trofts, tsofts
    
    
    def all_asy_train(self, scale=False):
        """
        making train data including training of test assay
        """
        if scale == False:
            tr_dict = self.tr_dict
        elif scale == True:
            tr_dict = self.tr_dict_sc
            
        train = pd.DataFrame()
        for key in tr_dict.keys():
            if key == self.tes_key:
                continue
            train = pd.concat([train, tr_dict[key]])
        train = pd.concat([train, tr_dict[self.tes_key]])
        
        grouped = train.groupby(level=0)
        train = grouped.last()
        
        return train
    
    
    def xyprep(self, train, test, xmode='ecfp'):
        """
        prepare training and test data
        """
        if xmode == 'ecfp':
            xtr = Hash2FingerPrint(train['hash']).values
            xts = Hash2FingerPrint(test['hash']).values
        elif xmode == 'm2v':
            xtr = train.loc[:, train.columns.str.contains('raidus')]
            xts = test.loc[:, test.columns.str.contains('raidus')]
        elif xmode == 'ecfp_plus':
            xtr = Hash2FingerPrintPlusAsyInfo(train).values
            xts = Hash2FingerPrintPlusAsyInfo(test).values
            
        ytr = train['pot.(log,IC50)'].values.reshape(-1,1)
        yts = test['pot.(log,IC50)'].values.reshape(-1,1)
        
        return xtr, ytr, xts, yts
    
    
    #for scaling method
    def _sc_ratio(self):
        numer = self.all_dict[self.tes_key].loc[self.dupcomp]['pot.(nMol,IC50)']
        
        for key in self.all_dict.keys():
            ratio = numer / self.all_dict[key].loc[self.dupcomp]['pot.(nMol,IC50)'] 
                
            df = self.all_dict[key].copy()
            
            if key != self.tes_key:
                df['pot.(nMol,IC50)'] = df['pot.(nMol,IC50)'] * ratio
                df['pot.(log,IC50)'] = -np.log10(df['pot.(nMol,IC50)'] *10**-9)
            elif key == self.tes_key:
                pass
            
            self.all_dict_sc[key] = df
            
    
    def _sc_ratio_indiv(self):
        
        
        for tes_key in self.all_dict.keys():
            numer = self.all_dict[tes_key].loc[self.dupcomp]['pot.(nMol,IC50)']
            
            all_dict_sc = {}
            for key in self.all_dict.keys():
                ratio = numer / self.all_dict[key].loc[self.dupcomp]['pot.(nMol,IC50)']
                
                df = self.all_dict[key].copy()
            
                if key != tes_key:
                    df['pot.(nMol,IC50)'] = df['pot.(nMol,IC50)'] * ratio
                    df['pot.(log,IC50)'] = -np.log10(df['pot.(nMol,IC50)'] *10**-9)
                elif key == tes_key:
                    pass
                
                all_dict_sc[key] = df
                
            self.sc_dicts[tes_key] = all_dict_sc
        
        
        



if __name__ == '__main__':
    
    dfs = pickle.load(open('Z:/Research/object/dup_assays.pickle','rb'))
    
    prep = AssayPrep(dfs, 'tid-9-actives.txt', 'CHEMBL939', 20) #choose target & duplicate compound
    all_dict = prep.all_dict
    all_dict_sc = prep.all_dict_sc