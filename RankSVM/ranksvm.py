#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:52:28 2020

@author: matsumoto
"""

import subprocess
import os


class SVM_Rank():
    
    def __init__(self, path=None):
        self.path = path
        self.file = '/Users/matsumoto/Research/Research/Rank-SVM'


    def run_svmrank(self, inp_name='train', model_name='model_train', c=0.01):
        """
        inp : train or val
        """
        
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(model_name)), 'w+')
        
        inp_name = os.path.join(self.path, '{}.txt'.format(inp_name))
        model_name = os.path.join(self.path, '{}.txt'.format(model_name))
        
        input_line = './svm_rank_learn -c {cost} {inp_name} {model_name}'.format(cost=c, inp_name=inp_name, model_name=model_name)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
        
        
    def predict_svmrank(self, inpfile='train', modelfile='model_train', resultfile='predictions.train'):
        """
        inp   : train or test
        model : train or val
        """
        os.chdir(self.file) # move to folder for SVM-Rank
        
        fp = open(os.path.join(self.path, 'rsvm_{}_log.log'.format(resultfile)), 'w+')
        
        modelfile   = os.path.join(self.path, '{}.txt'.format(modelfile))
        inpfile     = os.path.join(self.path, '{}.txt'.format(inpfile))
        resultfile  = os.path.join(self.path, '{}.txt'.format(resultfile))

        input_line = './svm_rank_classify {inpfile} {modelfile} {resultfile}'.format(inpfile=inpfile, modelfile=modelfile, resultfile=resultfile)
        res = subprocess.run([input_line], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        result_writing(fp, res)
        fp.close()
        
        
def result_writing(filep, res):
    """
    preserving the output in command line
    """
    for line in res.stdout.splitlines():
        filep.write(line.decode('utf-8'))
        filep.write('\n')
        


if __name__ == "__main__":
    pass
