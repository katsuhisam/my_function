import pandas as pd
import numpy as np
import random
from functools import partial
from sklearn.metrics import r2_score
from chemical.FingerPrint import Hash2FingerPrint
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from Kernel.KernelRidge import KernelRidge_CV
from evaluation.criteria import r2_rmse_mae

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:26:35 2019

@author: matsumoto
"""

class GAKernelRidge_yscale():
    
    def __init__(self, population=100, generations=100, use_scaling=False, seed=0, debug=False, 
                maxcmp=None, use_roulette=True, use_penalty_GA=False, mutation_rate=0.2, nfold=None, 
                searchrange=(-5,5), xname=None, yname=None):
        self.pop  = population
        self.gens = generations
        self.scale = use_scaling
        self.xscaler = None
        self.yscaler = None
        self.model = None
        self.best_mask = None
        self.params = None
        self.verbose = debug
        self.maxcmp = maxcmp
        self.use_roulette = use_roulette
        self.use_penalty_GA = use_penalty_GA
        self.r_mutate = mutation_rate
        self.nfold = nfold
        self.searchrange = searchrange
        self.y_convert = None
        self.xname = xname
        self.yname = yname

        random.seed(seed) # hopefuly this work...
        
    def fit(self, x, y, x_val, y_val):
        """
        Variable selection with GA 
        input:
        -------
        x, y: training data set
        nfold: nfold-cross validation for hyperparameter identification 
        
        output:
        --------
        optimized set of variables, and performance of GA... (log)
        
        """
        trindex = x.index
        
#        x = np.array(x)
#        y = np.array(y).reshape(-1,1)
#        x_val = np.array(x_val)
#        y_val = np.array(y_val).reshape(-1,1)
        
#        if self.scale:
#            self.xscaler = StandardScaler()
#            self.yscaler = StandardScaler()
#            sx = self.xscaler.fit_transform(x)
#            sy = self.yscaler.fit_transform(y)
#        else:
        sx = x
        sy = y
        
        n_variable = count_asy(x)
        
        #preparation for GA
        creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
        creator.create('Individual', list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        #create gene expression
        toolbox.register('attr_gene', random.uniform, self.searchrange[0], self.searchrange[1])
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_gene, n_variable)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        #optimization strategy
        ga_kr_opt = partial(WrapperGA, x=sx, y=sy, x_val=x_val, y_val=y_val, trindex=trindex, nfold=self.nfold, xname=self.xname, yname=self.yname)
        
        toolbox.register('evaluate', ga_kr_opt)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        toolbox.register('select', tools.selRoulette) # or Roullet 
        
        # statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('max', np.max)
        stats.register('min', np.min)

        pop = toolbox.population(n=self.pop)
        self.final_combi, self.galogs = algorithms.eaMuPlusLambda(pop, toolbox, mu=self.pop, lambda_=self.pop, cxpb=0.5,
                                    mutpb=0.2, ngen=self.gens, stats=stats)
        
        # set the best model
        evals = [ga_kr_opt(val)[0] for val in self.final_combi]
        coef = self.final_combi[np.argmax(evals)]
        self.best_coef = np.array(coef)

        # model recostruction
        self.evaluate_r2, self.model, self.xa, self.ya, self.convy = WrapperGA(individual=self.best_coef, 
                                                                               x=sx, y=sy, x_val=x_val, y_val=y_val,
                                                                               trindex=trindex,
                                                                               nfold=self.nfold,
                                                                               xname=self.xname,
                                                                               yname=self.yname,
                                                                               return_model=True)
        # optimized r2,rmse,mae
        syptr = self.model.predict(self.xa)
#
#        if self.scale:
#            yptr = self.yscaler.inverse_transform(syptr)
#        else:
        yptr = syptr
#
        self.evaluate_r2, self.evaluate_rmse, self.evaluate_mae = r2_rmse_mae(yp=yptr, yobs=self.ya, verbose=self.verbose)
        
        
    def predict(self, x):
        """
        Wrapper function of predcit with the optimal
        """
        x = np.array(x)
        
        if self.scale:
            sx = self.xscaler.transform(x)
        else:
            sx = x

        spy = self.model.predict(sx)
        
#        if self.scale:
#        py = self.yscaler.inverse_transform(spy)
#        else:
        py = spy
        return py
    
    
    def predict_vals(self, xtr, xts):
        py1 = self.predict(xtr)
        py2 = self.predict(xts)
        return py1, py2
    
        
def WrapperGA(individual, x, y, x_val, y_val, trindex, xname, yname, nfold, return_model=False):
    """
    Wrapper function for conducting GAKernelRidge
    """
    
    g = individual
    asys = set(x['Assay'])
    
    for i, asn in enumerate(asys):
        y[yname][y['Assay'] == asn] = y[yname][y['Assay'] == asn] + g[i]
    convy = y
#    g = individual
#    for i,j in enumerate(g):
#        if i == 0:
#            yg = y + j
#        else:
#            y_bf = y + j            
#            yg = np.concatenate([yg, y_bf],1)
#    
#    mask = make_mask(trindex)
#    one = np.ones((mask.shape[1], 1))
#    yg_n = yg * mask
#    y_c = np.dot((yg_n), one)
    
    max_r2, model, xa, ya = kr_eval(x, y, x_val, y_val, xname, yname, nfold)
    print(max_r2)
    
    if return_model:
        return max_r2, model, xa, ya, convy
    else:
        return (max_r2,) #must be a tuple
    
        
def make_mask(tridx):
    if not isinstance(tridx, pd.DataFrame):
        TypeError('have to be pd.DataFrame...')
    
    assays = [detect_asy(idx) for idx in tridx]
    asynums = np.unique(assays)
    mask = pd.DataFrame(np.zeros((len(tridx), len(asynums))), columns=asynums)
    
    for i, j in enumerate(assays):
        mask.at[i, j] = 1
        
    return mask       
    
    
def detect_asy(idx):
    asyNo = int(idx.split('_')[1])
    return asyNo
    
    
        
def kr_eval(x, y, x_val, y_val, xname, yname, nfold=None):
    """
    For evaluating GA KernelRidge bsased on valdation set fitting
    """
    
    xa = pd.concat([x, x_val])
    ya = pd.concat([y, y_val])
    
    xa = Hash2FingerPrint(xa[xname]).values
    ya = ya[yname].values.reshape(-1,1)
    
#    yscaler = StandardScaler()
#    sy = yscaler.fit_transform(y)
    
    if isinstance(nfold, int):
        kr = KernelRidge_CV(kernelf='tanimoto', nf=nfold)
#    elif nfold == None:
#        svr = NuSVR_validate(kernelf='tanimoto')
        
    kr.fit(xa, ya)
    py = kr.predict(xa)
    
#    py = yscaler.inverse_transform(py)
    
    #R2 of validation set
    r2, _, _ = r2_rmse_mae(py, ya)
    model = kr.model

    return r2, model, xa, ya


def count_asy(data):
    count = len(set(data['Assay']))
    
    return count


if __name__ == 'main':
    pass