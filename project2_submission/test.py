# In[0]:

# Necessary modules

from helpers import *  
import math
import torch
torch.set_grad_enabled(False)

# In[1]:

# Constants for defaulted inputs

NUMBER_SAMPLES = 1000
NUMBER_EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 10
ACT_FUNC = 'relu'


# In[2]:

# Train and evaluate 

def run(n_epochs=NUMBER_EPOCHS, lr=LEARNING_RATE, 
        batch_size=BATCH_SIZE, n_samples=NUMBER_SAMPLES, 
        act_fn=ACT_FUNC):
    """
    Generates train and test datasets then trains and tests model.
    
    Args: n_epochs: number of training epochs
            lr: learning rate of the optimizer
            batch_size: mini-batch size
            n_samples: number of samples in both train and test sets
            act_fn: hidden layer activation function
    """
    
    # generate train/test data
    train_inp, train_target = generate_data(n_samples=n_samples)
    test_inp, test_target = generate_data(n_samples=n_samples)
    
    # normalize input with training set mean and std
    mu, std = train_inp.mean(dim=0), train_inp.std(dim=0)
    train_inp.sub_(mu).div_(std)
    test_inp.sub_(mu).div_(std)
    
    # choose either ReLU or Tanh as hidden layer activation functions
    if act_fn=='relu':
        model = Sequential(Linear(2, 25), ReLU(), 
                           Linear(25,25), ReLU(), 
                           Linear(25,25), ReLU(), 
                           Linear(25, 2), Tanh())
    if act_fn=='tanh':
        model = Sequential(Linear(2, 25), Tanh(), 
                           Linear(25,25), Tanh(), 
                           Linear(25,25), Tanh(),
                           Linear(25, 2), Tanh())

    # define optimizer and loss
    optimizer = SGD(model.params(), lr=lr)
    criterion = MSELoss
    
    # print info
    info_start = """
    Training Began
    
    Default model: Linear(2, 25), ReLU, Linear(25,25), ReLU, Linear(25,25), ReLU, Linear(25, 2), Tanh. 
    
    Epoch number: {0}, batch size: {1}, number of samples: {2}, learning rate: {3}
    """
    print(info_start.format(n_epochs, batch_size, n_samples, lr))
    
    for e in range(n_epochs):  
        
        # permutate indices
        idx = torch.randperm(n_samples)

        err_epoch = 0.
        
        # batch training
        for i in idx[: : batch_size]:
            
            # forward 
            output = model.forward(train_inp[i: i+batch_size])
            loss = criterion(output, train_target[i: i+batch_size])
            
            # clear gradient
            optimizer.zero_grad()
            
            # backward and update
            model.backward(loss.backward())
            optimizer.step()
            
            # record errors
            err_epoch  += evaluate(
                                   model=model, 
                                   inputs=train_inp[i: i+batch_size], 
                                   targets=train_target[i: i+batch_size],
                                   outputs=output) / n_samples

        # print logs
        print("Epoch {}: Training Loss: {:4.2f}, Training Error Rate: {:6.2%}\n".format(e, loss.forward(), err_epoch))

    # evaluate model with test data
    test_error = evaluate(model=model, inputs=test_inp, targets=test_target) / n_samples
    
    # print test info
    info_end = """
    Training Ended
    
    Test on {} test data: Test error rate = {:6.2%}
    """
    print(info_end.format(n_samples, test_error))


# In[3]:

if __name__ == '__main__':

    # Execute main
    run()