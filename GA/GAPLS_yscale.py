import random 
import array
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from functools import partial
from deap import base, creator, tools, algorithms

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from PLS.plswrappers import crossValPLS, PLS_CV
from evaluation.criteria import r2_rmse_mae

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:59:15 2020

@author: matsumoto
"""

class GAPLS_yscale():

    def __init__(self, population=100, generations=100, use_scaling=False, seed=0, debug=False, 
                maxcmp=None, use_roulette=True, use_penalty_GA=False, mutation_rate=0.2, nfold=None,
                searchrange=(-5,5), yname=None, asycol='Assay'):
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
        self.yname = yname
        self.asycol = asycol

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
        
#        x = np.array(x)
#        y = np.array(y).reshape(-1,1)
#        
#        if self.scale:
#            self.xscaler = StandardScaler()
#            self.yscaler = StandardScaler()
#            sx = self.xscaler.fit_transform(x)
#            sy = self.yscaler.fit_transform(y)
#        else:
#        self.xnames = [col for col in x.columns if col != self.asycol]
        
        sx = x
        sy = y
        
        n_variable = count_asy(sy, self.asycol)

        
        # preparation for binary GA
        creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
        creator.create('Individual', list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        #create gene expression
        toolbox.register('attr_gene', random.uniform, self.searchrange[0], self.searchrange[1]) 
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_gene, n_variable)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        # optimization strategy 
        ga_pls_cv = partial(WrapperGA, x=sx, y=sy, x_val=x_val, y_val=y_val, nfold=self.nfold,
                                yname=self.yname, method='pls', return_model=False, maxcmp=self.maxcmp)
        
        toolbox.register('evaluate', ga_pls_cv)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        if self.use_roulette:
            toolbox.register('select', tools.selRoulette) # or Roullet 
            #toolbox.register('select', tools.selTournament, tournsize=5) # or Roullet 
        else:
            toolbox.register('select', tools.selBest) # or Roullet 
        
        
        # statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('max', np.max)
        stats.register('min', np.min)
        
        pop = toolbox.population(n=self.pop)
        self.final_combi, self.galogs = algorithms.eaMuPlusLambda(pop, toolbox, mu=self.pop, lambda_=self.pop, cxpb=0.5,
                                    mutpb=self.r_mutate, ngen=self.gens, stats=stats)
        
        # for paper iterative screening
        # self.final_combi, self.galogs = algorithms.eaMuPlusLambda(pop, toolbox, mu=self.pop, lambda_=50, cxpb=0.5,
        #                             mutpb=0.2, ngen=self.gens, stats=stats)
        
        # set the best model
        evals = [ga_pls_cv(val)[0] for val in self.final_combi]
        coef = self.final_combi[np.argmax(evals)]
        self.best_coef = np.array(coef)

        # model recostruction
        self.evaluate_r2, self.model, self.xa, self.ya, self.xscaler, self.yscaler, self.convy =\
                                                                                WrapperGA(individual=self.best_coef,
                                                                                x=sx, y=sy, x_val=x_val, y_val=y_val,
                                                                                nfold=self.nfold,
                                                                                yname=self.yname,
                                                                                return_model=True,
                                                                                method='pls')
        self.yptr = self.predict(self.xa)
        
#        yptr_nonscale = self.model.predict(self.xscaler.transform(self.xa))
#        self.yptr = self.yscaler.inverse_transform(yptr_nonscale)

        self.evaluate_r2, self.evaluate_rmse, self.evaluate_mae = r2_rmse_mae(yp=self.yptr, yobs=self.ya, verbose=self.verbose)


    def predict(self, x):
        """
        Wrapper function of predcit with the optimal
        """
        x = np.array(x)
        
#        if self.scale:
#            sx = self.xscaler.transform(x)
#        else:
#            sx = x
        
        sx = self.xscaler.transform(x)

        spy = self.model.predict(sx)
        
#        if self.scale:
#            py = self.yscaler.inverse_transform(spy)
#        else:
#            py = spy
            
        py = self.yscaler.inverse_transform(spy)
        
        return py


    def predict_vals(self, xtr, xts):
        py1 = self.predict(xtr)
        py2 = self.predict(xts)
        return py1, py2

    def get_params(self):
        return self.params

def WrapperGA(individual, x, y, x_val, y_val, yname, nfold, return_model=False, asycol='Assay', 
              method='pls', maxcmp=None):
    """
    Wrapper function for conducting GAPLS
    """
    
    g = individual
    asys = set(y[asycol])
    ycop = y.copy()
    
    for i, asn in enumerate(asys):
        ycop[yname][y[asycol] == asn] = y[yname][y[asycol] == asn] + g[i]
    convy = ycop
    
    if method =='pls':
        max_r2, model, xa, ya, xscaler, yscaler = pls_eval_cross_val(x, convy, x_val, y_val, asycol, nfold, maxcmp=maxcmp)
        print(max_r2)
    
    if return_model:
        return max_r2, model, xa, ya, xscaler, yscaler, convy
    else:
        return (max_r2,)  #must be a tuple


def pls_eval_cross_val(x, y, x_val, y_val, asycol, nfold, maxcmp):
    """
    For evaluating GA PLS bsased on Cross validation
    """
    xa = pd.concat([x, x_val])
    ya = pd.concat([y, y_val])
    
    ya = ya[ya.columns[ya.columns != asycol]]
    
    xscaler = StandardScaler()
    yscaler = StandardScaler()
    
    sxa = xscaler.fit_transform(xa)
    sya = yscaler.fit_transform(ya)
    
    pls = PLS_CV(nf=nfold, maxCmp=maxcmp)
    pls.fit(sxa, sya)
    spy = pls.predict(sxa)
    
    py = yscaler.inverse_transform(spy)
    
    r2, _, _ = r2_rmse_mae(py, ya)
    model = pls.model
    return  r2, model, xa, ya ,xscaler, yscaler
    

def count_asy(data, asycol):
    count = len(set(data[asycol]))
    
    return count
