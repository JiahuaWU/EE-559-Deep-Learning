import torch
from torch import nn
import dlc_practical_prologue as prologue

# Weights initialization object
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def generate_data(normalization=True):
    # Generate data using prologue
    # param: normalization: To normalize the data or not
    # return: data needed for training and testing
    
    # Generate data using prologue
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    
    # Normalize data if requested
    if normalization:
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)
        train_target = train_target.float()
        test_target = test_target.float()
    return train_input, train_target, train_classes, test_input, test_target, test_classes 

def performance_estimate(net, train_para, n_rounds = 2):
    # Access the performance of a network structure through n_round tests 
    # where both data and weight initialization are randomized
    #
    # param: net: The network structure to be tested 
    #            train_para: Parameter setting for training      
    #            n_rounds: Number of rounds. The default is set to be 2 to avoid std to be nan

    final_acc = torch.zeros(n_rounds)
    final_loss = torch.zeros(n_rounds)
    print('Total training rounds: ',n_rounds)
    for i in range(n_rounds):
        
        # Reload randomized data
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_data()
        
        # Train the model through data (Note that the random initialization 
        # of model weights is included in the function "train")
        print('--------------------------------------------------------')
        print ('Start training for the ',i+1, ' round(s)')
        train(net, train_para, train_input, train_target, train_classes, print_log=True)
        
        # Print out statistics 
        final_acc[i], final_loss[i] = test(net, test_input, test_target, train_para)
        print('Test accuracy = %.3f Test loss = %.3f'%(final_acc[i],final_loss[i]))
    print('------------------------------------------------------------------------------')
    print('Test mean acc = %.3f, Test acc std = %.3f, Test loss mean = %.3f, Test loss std = %.3f'%(final_acc.mean(), final_acc.std(), final_loss.mean(), final_loss.std()))

def train(model, train_para, train_input,train_target,train_classes,train_ind=torch.arange(1000), print_log=True):
    # Train the model
    # params: model: the model to be trained
    #             train_ind: 1D tensor of sample indices 
    #             print_log: print out the training log or not
    #.            plot_log: plot the training curve or not
    
    # Initialize model parameters
    model.apply(weights_init)
    
    # Switch the model to training mode
    model.train()
    
    # Read in training parameters
    adjust_lr = train_para['adjust_lr']
    n_epochs = train_para['n_epochs']
    loss_ratio = train_para['loss_ratio']
    n_samples = train_ind.size(0)
    batch_size = train_para['batch_size']
    criterion = train_para['criterion']
    aux_criterion = train_para['aux_criterion']
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_batches = n_samples/batch_size

    # store loss/accuracy 
    major_loss_per_epoch = torch.zeros(n_epochs)
    aux_loss_per_epoch = torch.zeros(n_epochs)
    acc_per_epoch = torch.zeros(n_epochs)
    for e in range(n_epochs):      
        # Generate indices for each batch
        for ind in torch.utils.data.BatchSampler(torch.utils.data.SubsetRandomSampler(train_ind),
                                                       batch_size=batch_size, drop_last=False):  
            # Access data from training set
            input = train_input[[ind]]
            target = train_target[[ind]]
            classes = train_classes[[ind]]
            if train_para['flatten_input']: 
                input = torch.flatten(input, start_dim=1)

            # Forward
            output = model(input)
            
            # Calculate training statistics
            if train_para['aux_loss']:
                major_loss = criterion(output[0].squeeze(), target)
                aux_loss = aux_criterion(output[1], classes[:,0]) + aux_criterion(output[2], classes[:,1])
                loss = loss_ratio*major_loss + aux_loss
                major_loss_per_epoch[e] += major_loss.item()
                aux_loss_per_epoch[e] += aux_loss.item()
                pred = output[0].reshape(target.size()).round()
                     
            else:
                loss = criterion(output.flatten(), target)
                major_loss_per_epoch[e] += loss.item()
                pred = output.reshape(target.size()).round()
                
             # accummulate right predctions   
            acc_per_epoch[e] += pred.eq(target).sum().item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        aux_loss_per_epoch[e] /= n_batches
        major_loss_per_epoch[e] /= n_batches
        acc_per_epoch[e] /= n_samples
        
        # Print log
        if print_log:
            print('epoch [%d]: training main loss = %.3f, auxiliary loss = %.3f, training accuracy = %.3f'%(
                    e+1, major_loss_per_epoch[e], aux_loss_per_epoch[e], acc_per_epoch[e]))
            
def test(model, input, target, train_para):
    # Test the model,acc_per_peopoch[]e
    # params: model: the model to be tested
    #             input: test input
    #             target: test target
    # Return:  test accuracy and loss of the main task 

    # Read in parameters
    criterion = train_para['criterion']
    n = input.size(0)
    
    # Flatten the input into 1D tensor or not 
    if train_para['flatten_input']:
        input = input.flatten(start_dim=1)
        
    # Switch the model to evaluation mode 
    model.eval()
    with torch.no_grad():
        # Predict       
        output = model(input)
        if train_para['aux_loss']:
            pred_target = output[0]
        else:
            pred_target = output
        pred_target = pred_target.reshape(target.size())
        
        # Obatain loss
        major_loss = criterion(pred_target, target).item()
 
        # Obtain number of correct predictions
        pred_target = pred_target.round()
        
        acc = (pred_target.eq(target).sum().item())
                
    # Return test accuracy and loss of the model 
    return acc/n, major_loss


