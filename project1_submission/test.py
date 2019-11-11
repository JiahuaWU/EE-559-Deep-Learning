
# coding: utf-8

# In[1]:


import torch
from torch import nn
from nets import * # Import nets described in the report
from helpers import *
import torch.nn.functional as F
import dlc_practical_prologue as prologue
import copy
from Cross_Validation import *


# In[2]:


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_data()


# In[3]:


net = final()
net.apply(weights_init)
train_para=dict()
train_para['aux_loss'] = True
train_para['adjust_lr'] = False
train_para['n_epochs'] = 25
train_para['loss_ratio'] = 0.5
train_para['batch_size'] = 5
train_para['criterion'] = nn.BCELoss()
train_para['aux_criterion'] = nn.CrossEntropyLoss()
train_para['flatten_input'] = False
net_total_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('total trainable parameters = ', net_total_para)


# In[4]:


performance_estimate(net,train_para,n_rounds = 2)

