
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from helpers import *

def CV(model, train_para,train_input,train_target,train_classes,k_folds):
    # Perform k folds cross validation on training set
    # Params: model: the model to be tested
    #             k_folds: number of folds
    
    acc = torch.zeros(k_folds)
    loss = torch.zeros(k_folds)
    
    # Randomly generate k_folds groups of indices 
    cv_ind = torch.randperm(1000).reshape(k_folds, -1)
    
    # Perform cross validation on k_folds folds
    for k in range(k_folds):
        print('CV step', k+1)
        
        # Split out one group of indices for training and the rest for testing
        train_ind = torch.cat([cv_ind[0:k], cv_ind[k+1:k_folds]]).flatten()
        test_ind = cv_ind[k]
        
        # Train on the training set
        train(model, train_para, train_input,train_target,train_classes, train_ind=train_ind, print_log=False)
        
        # Test on the test set
        acc[k], loss[k] = test(model, train_input[test_ind], train_target[test_ind], train_classes[test_ind], train_para, aux=False)
        
        # Print out statistics
        print('cv acc = %.3f, cv loss = %.3f'% (acc[k], loss[k]))
    print('average cv acc = %.3f, average cv loss = %.3f, cv acc std = %.3f, cv loss std = %.3f'% (acc.mean(), loss.mean(), acc.std(), loss.std()))

