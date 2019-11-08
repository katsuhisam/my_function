import numpy as np
import pandas as pd
from evaluation.criteria import r2_rmse_mae
from datatype.returntype import cv_returnData
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.model_selection import GridSearchCV as gcv
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier




class RFRegression_CV():
    """
    cross validation version. wrapper class of RandomForest regressor
    """
    
    def __init__(self, rseed=0, nf=5, paramset=None, verbose=False, selectedScore=None, debug=False):
        self.nf         = nf
        self.verbose    = verbose
        self.rng        = np.random.RandomState(rseed)
        
        self.metric, self.cv_paramset = self._set_conditions(selectedScore, paramset)
        
        if debug:
            self.nworkers = 1
        else:
            self.nworkers = -1

        self.model = self._set_model()
        

    def _set_conditions(self, selectedScore, paramset):
        """
        Setting calculation conditions 
        """
        if selectedScore is None:
            metric = 'neg_mean_absolute_error'
        else:
            metric = selectedScore
        
        if paramset is None:
            paramset = dict(n_estimators=[50, 100, 300], max_depth=[None, 10], max_features=['sqrt', None])
                        
        return metric, paramset


    def _set_model(self):
        """
        Setting the models with parameters
        """
        if self.nf == 'loo':
            cv=LeaveOneOut()
        else:
            cv=KFold(n_splits=self.nf, shuffle=True, random_state=self.rng)
        return gcv(RandomForestRegressor(), param_grid=self.cv_paramset,cv=cv,
                scoring=self.metric, n_jobs=self.nworkers, verbose=self.verbose)
        

    def fit(self, x, y, weights=None):
        """
        Fit the cv model with the x and y
        """
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1,1)

        self.model.fit(x, y.ravel())
        
        # optimized parameters
        self.params = self.model.best_params_
        self.pmodel = RandomForestRegressor(n_estimators=self.model.best_params_['n_estimators'], max_features=self.model.best_params_['max_features'], 
                                            max_depth=self.model.best_params_['max_depth'])
        
        # prediction by the best model
        self.pmodel.fit(x, y.ravel())

        Yptr = self.pmodel.predict(x)
        self.evaluate_r2, self.evaluate_rmse, self.evaluate_mae = r2_rmse_mae(yp=Yptr, yobs=y, verbose=self.verbose)


    def get_params(self):
        return self.params


    def predict(self, x):
        """
        Predict y values for x

        Note: there is no parallellization here (apply_with_batch function)
        """
        x = np.array(x)
        
        py = self.pmodel.predict(x)
        py = py.reshape(-1,1)

        return py

    def predict_vals(self, xtrain, xtest):
        """
        predict multiple y values for mulple xs 
        """
        py1 = self.predict(xtrain)
        py2 = self.predict(xtest)
        
        return py1, py2


def crossValidationRFclassify(xtr, ytr, rng, nf=5, xts=None, yts=None, paramset=None,
                            verbose=True, selectedScore=None):
    """
	Wrapper function for cross validation for RF with training data.
    Score (metric) for CV is AUC-ROC in this case. 
	
    Input:
    ------
    rng: random number generator
    xts: x test data (None)
	yts: y test data (1d array) (None)
    
	nf: n_fold cross validation
	paramset: parameter set for cv (None)

    Output:
    -------
    cross-validated model.
	"""

    if paramset is None:
        paramset = dict(n_estimators=[50, 100, 300],  max_features=[None, 'sqrt', 'log2'])
        if verbose:
            print('Default parameter sets: ', paramset)
        
    if selectedScore is None:
        selectedScore = 'roc_auc'

    rf = RandomForestClassifier(class_weight='balanced', random_state=0) # for avoiding complexity
    
    model = gcv(rf, param_grid=paramset, cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=rng),
                n_jobs=4, scoring=selectedScore, verbose=verbose)
    model.fit(xtr, ytr)

    # optimized parameters
    if verbose:
        print("opt params\n %s" % model.best_estimator_)
    
    # prediction by the best model
    Yptr = model.predict(xtr)

    # accuray of the models to the training
    print("For trainig data")
    print((confusion_matrix(ytr, Yptr)))
    
    # apply the optimal model to the test dataset
    if (xts is not None) and (yts is not None):
            
        Ypts = model.predict(xts)
        print("For test data")
        print((confusion_matrix(yts, Ypts)))
        return cv_returnData(model, Yptr, Ypts)
    else:
        return cv_returnData(model, Yptr)

