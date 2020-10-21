import pandas as pd
import numpy as np
import random
from functools import partial
from sklearn.metrics import r2_score
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from mySVR.svr_wrapper import NuSVR_validate, NuSVR_DCV
from svr.svr_wrapper import NuSVR_CV
from Kernel.KernelRidge import KernelRidge_CV
from evaluation.criteria import r2_rmse_mae
from chemical.FingerPrint import Hash2FingerPrint
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:13:42 2019

@author: matsumoto
"""

class GASVR_yscale():
    
    def __init__(self, population=100, generations=100, use_scaling=False, seed=0, debug=False, 
                maxcmp=None, use_roulette=True, use_penalty_GA=False, mutation_rate=0.2, nfold=None, 
                searchrange=(-5,5), xname=None, yname=None, trace=None, method='svr'):
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
        self.method = method
        self.trace = trace

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
#        x_val = np.array(x_val)
#        y_val = np.array(y_val).reshape(-1,1)
#        
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
        if self.trace == None:
            toolbox.register('attr_gene', random.uniform, self.searchrange[0], self.searchrange[1])
            toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_gene, n_variable)
        elif self.trace != None:
            toolbox.register('attr_gene', randompost, self.trace)
            toolbox.register('individual', initRepeating, creator.Individual, toolbox.attr_gene, n_variable)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        #optimization strategy
        ga_svr_opt = partial(WrapperGA, x=sx, y=sy, x_val=x_val, y_val=y_val, nfold=self.nfold, 
                             xname=self.xname, yname=self.yname, return_model=False, method=self.method)
        
        toolbox.register('evaluate', ga_svr_opt)
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
        evals = [ga_svr_opt(val)[0] for val in self.final_combi]
        coef = self.final_combi[np.argmax(evals)]
        self.best_coef = np.array(coef)

        # model recostruction
        self.evaluate_r2, self.evaluate_q2, self.model, self.xa, self.ya, self.xscaler, self.yscaler, self.convy = WrapperGA(individual=self.best_coef, 
                                                                                                                               x=sx, y=sy, 
                                                                                                                               x_val=x_val, y_val=y_val,
                                                                                                                               nfold=self.nfold,
                                                                                                                               xname=self.xname,
                                                                                                                               yname=self.yname,
                                                                                                                               return_model=True,
                                                                                                                               method=self.method)
        # optimized r2,rmse,mae
        syptr = self.predict(self.xa)
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
        
        if self.scale:
            py = self.yscaler.inverse_transform(spy)
        else:
            py = spy
        return py
    
    
    def predict_vals(self, xtr, xts):
        py1 = self.predict(xtr)
        py2 = self.predict(xts)
        return py1, py2
    
        
def WrapperGA(individual, x, y, x_val, y_val, nfold, method, xname=None, yname=None, return_model=False):
    """
    Wrapper function for conducting GASVR
    """
    
    g = individual
    asys = set(x['Assay'])
    ycop = y.copy()
    
    for i, asn in enumerate(asys):
        ycop[yname][y['Assay'] == asn] = ycop[yname][y['Assay'] == asn] + g[i]
    convy = ycop
    
    max_r2, max_q2, model, xa, ya, scaler = svr_eval(x, ycop, x_val, y_val, xname, yname, nfold, return_model, method)
    print(max_r2)
    
    if return_model:
        return max_r2, max_q2, model, xa, ya, scaler[0], scaler[1], convy
    else:
        return (max_r2,) #must be a tuple
    
        
#def make_mask(tridx):
#    if not isinstance(tridx, pd.DataFrame):
#        TypeError('have to be pd.DataFrame...')
#    
#    assays = [detect_asy(idx) for idx in tridx]
#    asynums = np.unique(assays)
#    mask = pd.DataFrame(np.zeros((len(tridx), len(asynums))), columns=asynums)
#    
#    for i, j in enumerate(assays):
#        mask.at[i, j] = 1
#        
#    return mask       
#    
#    
#def detect_asy(idx):
#    asyNo = int(idx.split('_')[1])
#    return asyNo
    
    
        
def svr_eval(x, y, x_val, y_val, xname=None, yname=None, nfold=None, return_model=False, method='svr'):
    """
    For evaluating GA SVR bsased on valdation set fitting
    """
    xa = pd.concat([x, x_val])
    ya = pd.concat([y, y_val])
    
    xa = xa[~xa.index.duplicated(keep='last')]
    ya = ya[~ya.index.duplicated(keep='last')]
    
    if method == 'svr':
        xa = Hash2FingerPrint(xa[xname]).values
    elif method == 'svr_m2v':
        xa = xa.iloc[:,:-1]
    ya = ya[yname].values.reshape(-1,1)
    
    # scalilng process
    if method == 'svr':
        xa_sc = xa
        ya_sc = ya
    elif method == 'svr_m2v':
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        xa_sc = x_scaler.fit_transform(xa)
        ya_sc = y_scaler.fit_transform(ya)
    
    # model definition
    if method == 'svr':
        if isinstance(nfold, int):
            mdl = NuSVR_CV(kernelf='tanimoto', nf=nfold)
        elif nfold == None:
            mdl = NuSVR_validate(kernelf='tanimoto')
    elif method == 'svr_m2v':
        mdl = NuSVR_DCV(kernelf='rbf', nfin=nfold, nfout=nfold)
   
    mdl.fit(xa_sc, ya_sc)
    py = mdl.predict(xa)
    
    # Scaling inverse process
    if method == 'svr_m2v':
        py = y_scaler.inverse_transform(py)    
        scaler = [x_scaler, y_scaler]
    else:
        scaler =[None, None]
    
    #R2 of validation set
    if method == 'svr':
        r2, _, _ = r2_rmse_mae(py, ya)
    elif method == 'svr_m2v':
        if return_model:
            r2, _, _ = r2_rmse_mae(py, ya)
            q2       = mdl.dcv_score
        else:
            r2 = mdl.dcv_score
            q2 = None
        
    model = mdl.model

    return r2, q2, model, xa, ya, scaler


def count_asy(data):
    count = len(set(data['Assay']))
    
    return count


# Bayesian inference
def randompost(trace, number=0):
    """
    number : assay number
    """
    rand = trace['alpha'][500:, number]
    rand = rand.reshape(-1)
    
    rn = random.choice(rand)
    
    return rn

def initRepeating(container, func, n_var):
    return container(func(number = i) for i in range(n_var))

        
if __name__ == 'main':
    pass