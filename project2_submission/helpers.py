"""
This file contains all the necessary functions
and modules.
"""
# In[0]:

# Necessary modules

import math
import torch


# In[2]:

# Framework classes

class Module(object):
    """Super class for other modules"""
    
    def forward(self , *input) :
        """ 
        Forward process the input data to generate predictions
        """ 
        raise NotImplementedError
        
    def backward(self , *dl_dx):
        """
        Pass gradients backwards and store gradients for each paramter
        
        """
        raise NotImplementedError
        
    def params(self):
        """ 
        Consist of a list of pairs, each composed of a parameter tensor
        and a gradient tensor of same size; None for parameterless modules.
        
        """
        return []


class Linear(Module):
    """
    Fully connected layer
    
    Args: in_size: size of each input sample
          out_size: size of each output sample
    
    """
    
    def __init__(self, in_size, out_size, add_bias=True):
        super(Linear, self).__init__()
        self.w = torch.empty(out_size, in_size) 
        self.dl_dw = torch.zeros_like(self.w)  
        if add_bias:
            self.b = torch.empty(out_size, 1)   
            self.dl_db = torch.zeros_like(self.b) 
            
        # initialize the variable for model input 
        self.x = 0.      

        # parameter initialization
        self.reset_parameters()    

    def reset_parameters(self):
        # reset model parameters by uniform distribution

        # std = 1 / in_size
        std = 1. / math.sqrt(self.w.size(1))
        
        # initialize weights with uniform(-std, std)
        self.w.uniform_(-std, std)
        if self.b is not None:
            self.b.uniform_(-std, std)
         
    def forward(self, input):
        # Make sure the input size consistent with the model. 
        self.x = input.clone().view(input.size(0), -1, 1)
        
        # s = W * x + b
        if self.b is None:
            return self.w.matmul(self.x)
        else:
            return self.w.matmul(self.x).add(self.b)
    
    def backward(self, dl_ds):
        # Gradients w.r.t. parameters: 
        # dl_dW = dl_ds * x^t 
        # dl_db = dl_ds
        self.dl_dw.add_(dl_ds.matmul(self.x.transpose(1, 2)).sum(dim=0))
        if self.b is not None:
            self.dl_db.add_(dl_ds.sum(dim=0)) 
            
        # Pass backwards dl_dx = W^t * dl_ds
        return self.w.t().matmul(dl_ds)
            
    def params(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
        

class Tanh(Module):
    """Tanh activation function"""
    
    def __init__(self):
        super(Tanh, self).__init__()
        self.s = 0.
        
    def forward(self, input):
        # store the input for backwards
        self.s = input.clone() 
        
        # x = tanh(s)
        return self.s.tanh()
    
    def backward(self, dl_dx):
        # dx/ds = 4/(exp(s) + exp(-s))^2
        dx_ds = 4 * ((self.s.exp() + self.s.mul(-1).exp()).pow(-2))
        
        # dl/ds = dl/dx * dx/ds (element-wise)
        return  dl_dx.mul(dx_ds)
    
    def params (self):
        return []

        
class ReLU(Module):
    """ReLU activation function """
    
    def __init__(self):
        super(ReLU, self).__init__()
        self.s = 0.
        
    def forward(self, input):
        # store input for backwards
        self.s = input.clone()
        
        # x = ReLU(s)
        return self.s.relu()
    
    def backward(self, dl_dx):
        # dx/ds = 1_{s>0}
        dx_ds = self.s.gt(0).float()
        
        # dl_ds = dl/dx * dx/ds
        return dl_dx.mul(dx_ds)   

    def params(self):
        return [] 
              
class SGD(object):
    """SGD optimizer"""
    
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        # Do a single step for each parameter
        # with SGD update: w += -lr * dl_dw
        for module_param in self.params:
            if len(module_param) != 0: 
                for param_group in module_param:
                    (param, param_grad) = param_group
                    if param is not None:
                        param.add_(-self.lr * param_grad)
    
    def zero_grad(self):
        # clean the gradient stored in the model
        for module_param in self.params:
            if len(module_param) != 0: 
                for param_group in module_param:
                    (param, param_grad) = param_group
                    if param is not None:
                        param_grad.zero_()
                
                
                
class Sequential(Module):
    """
    Build a model in a sequential order 
    
    Args: *args: a list of modules
    """
    
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.module_list = list(args)
    
    def forward(self, input):
        # forward pass through all moduels
        output = input        
        for module in self.module_list:
            output = module.forward(output)
            
        return output
    
    def backward(self, dl_dy):
        # backward through all modules
        dl_dx = dl_dy
        for module in self.module_list[ : : -1]:
            dl_dx = module.backward(dl_dx)

    def add_module(self, *modules):
        # append a list of new modules
        for module in modules:
            self.module_list.append(module)
    
    def params(self):
        # return a list of lists of module parameters
        return [module.params() for module in self.module_list]


class MSELoss(object):
    """
    Mean square loss 
    
    Args: pred: a list of predictions made by the model 
          target: a list of labels 
    """
    
    def __init__(self, pred, target):
        super(MSELoss, self).__init__()
        self.y = pred
        self.t = target.view(target.size(0), -1, 1).float()

    def forward(self):
        # MSE = (y - t)^2, average w.r.t. each sample
        return (self.y - self.t).pow(2).sum(dim=1).mean()
    
    def backward(self):
        # dl/dy = 2 * (y - t) / batch_size 
        # as we average loss across a batch during backward
        return 2 * (self.y - self.t) / self.y.size(0)


# In[3]:

# Utility functions

def generate_data(n_samples):
    """
    Generates input data with shape 
    and corresponding targets with shape (n_sample, 2).
    For a sample inside a circle with radius 
    1/sqrt(2*pi), the target is 1 (the first element) 
    and 0 (the second element) otherwise. We use one-hot 
    coding for the target.
    
    Args: n_samples: the number of samples
    
    Return: inputs: (n_sample, 2) dimension FloatTensor
            targets: (n_sample, 2) dimension LongTensor 
    """
    
    # sample inputs from [0, 1]^2
    inputs = torch.rand(n_samples, 2)
    
    # label inputs depending on their norm
    tmp = (torch.norm(inputs.sub(0.5), dim=1).lt(1/math.sqrt(2*math.pi))).float().unsqueeze(-1)
    targets = torch.cat([(tmp+1)%2, tmp], dim=1).long()
    
    return inputs, targets


def evaluate(model, inputs, targets, outputs=None):
    """
    Evaluates the model with the accuracy metric
    
    Args: model: trained model
          inputs,
          targets,
          outputs: during training, outputs are available

    Returns: error_num: the number of errors
    """
    
    # get predictions
    if outputs is None:
        outputs = model.forward(inputs)
    prediction = torch.argmax(outputs, dim=1).squeeze()
    
    # compare predictions with targets
    error_num = prediction.ne(torch.argmax(targets, dim=1)).sum().item() 
    
    return error_num
    
