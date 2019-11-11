
# coding: utf-8

# In[0]:


import torch
from torch import nn
from helpers import *
import copy

# In[0]

# Basic fc blocks

# Fully connected layer class
class fc_layer(nn.Module):
    def __init__(self, in_size, out_size, 
                act_fn=nn.ReLU(), batch_norm=False, drop_out=0):
        super(fc_layer, self).__init__()
        if batch_norm:
            self.fc = nn.Sequential(nn.Linear(in_size, out_size), act_fn,
                                    nn.BatchNorm1d(out_size), nn.Dropout(drop_out))
        else:
            self.fc = nn.Sequential(nn.Linear(in_size, out_size), act_fn,
                                    nn.Dropout(drop_out))
    def forward(self, x):
        return self.fc(x)

# Simple fully connected net class
class fc_net(nn.Module):
    def __init__(self, c, h, w, num_classes, 
                act_fn, batch_norm, drop_out, aux=False):
        super(fc_net, self).__init__()
        n = c*h*w
        self.fc = nn.Sequential(nn.Linear(n, n//2), act_fn,
                                fc_layer(n//2, n//2, act_fn=act_fn, batch_norm=batch_norm, drop_out=drop_out),
                                nn.Linear(n//2, num_classes))
        if not aux:
            self.fc.add_module('binary output', nn.Sigmoid())
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1) 
        x = self.fc(x) 
        return x

# In[1]

# Our models: 
# 1. all fully connected layer model
# 2. all convolutional layer model
# 3. baseline cnn model with both fc layers and convolutional layer
# 4. pseudo Siamese net (to show the effect of spliting images)
# 5. weight sharing net
# 6. pseudo Siamese net + auxiliary loss
# 7. weight sharing + auxiliary loss (final model)

# Only fully connected network
n=2*14*14
fc_1 = nn.Sequential(nn.Linear(n, n//2), nn.Sigmoid(), 
                             nn.Linear(n//2, n//4), nn.Sigmoid(),
                             nn.Linear(n//4, 1), nn.Sigmoid())

# Pure convolutional network
conv_1 = nn.Sequential(nn.Conv2d(2,10,5), nn.ReLU(), nn.BatchNorm2d(10), nn.MaxPool2d(2),
                             nn.Conv2d(10,20,3), nn.ReLU(), nn.BatchNorm2d(20), 
                             nn.Conv2d(20,1,3), nn.Sigmoid())

# No weight sharing net (originial input format)
class cnn_base(nn.Module):
    def __init__(self):
        super(cnn_base, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                  nn.MaxPool2d(2))
        self.fc = fc_net(128, 1, 1, num_classes=1, act_fn=nn.ReLU(), batch_norm=False, drop_out=0)
                     
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Pseudo weight sharing net (pseudo Siamese, input is splited 
# and passed seperately through two different 
# convolutional networks of the same structure).
class no_w_net(nn.Module):
    def __init__(self):
        super(no_w_net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                  nn.MaxPool2d(2))
        self.conv2 = copy.deepcopy(self.conv1)
        self.fc = fc_net(256, 1, 1, num_classes=1, act_fn=nn.ReLU(), batch_norm=False, drop_out=0)
                     
    def forward(self, x):
        #split two images: x.size = [batch_size, 2, 14, 14]
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # pass into different networks
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        #x1.size() = x2.size() = [batch_size, 128, 1, 1]
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.fc(x3)
        return x3

# auxiliary loss + base_cnn (without weight sharing)
class no_w_a_net(nn.Module):
    def __init__(self):
        super(no_w_a_net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                  nn.MaxPool2d(2))
        self.conv2 = copy.deepcopy(self.conv1)                               
        self.fc = fc_net(256, 1, 1, num_classes=1, act_fn=nn.ReLU(), batch_norm=False, drop_out=0)
        self.aux = fc_net(128, 1, 1, num_classes=10, act_fn=nn.ReLU(), batch_norm=False, drop_out=0, aux=True)
                     
    def forward(self, x):
        #split two images: x.size = [batch_size, 2, 14, 14]
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # different weights
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        # output 
        x3 = torch.cat([x1, x2], dim=1)
        main_out = self.fc(x3)
        aux_out1 = self.aux(x1)
        aux_out2 = self.aux(x2)
        return main_out, aux_out1, aux_out2
    
# Only weight sharing netv(Inputs are splited and passed 
# through the same convolutional network)
class w_net(nn.Module):
    def __init__(self):
        super(w_net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                  nn.MaxPool2d(2))                                
        self.fc = fc_net(256, 1, 1, num_classes=1, act_fn=nn.ReLU(), batch_norm=False, drop_out=0)
                     
    def forward(self, x):
        # split two images: x.size = [batch_size, 2, 14, 14]
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # weight sharing
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        
        # merge two inputs before fc layers
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.fc(x3)
        return x3

# final model with both weight sharing and auxiliary losses
class final(nn.Module):
    def __init__(self):
        super(final, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                  nn.MaxPool2d(2),
                                  nn.Dropout2d(0.4))                                
        self.fc = fc_net(256, 1, 1, num_classes=1, act_fn=nn.ReLU(), batch_norm=False, drop_out=0)
        self.aux = fc_net(128, 1, 1, num_classes=10, act_fn=nn.ReLU(), batch_norm=False, drop_out=0, aux=True)
                     
    def forward(self, x):
        # split two images: x.size = [batch_size, 2, 14, 14]
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # weight sharing
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        
        # output 
        x3 = torch.cat([x1, x2], dim=1)
        main_out = self.fc(x3)
        aux_out1 = self.aux(x1)
        aux_out2 = self.aux(x2)
        return main_out, aux_out1, aux_out2

