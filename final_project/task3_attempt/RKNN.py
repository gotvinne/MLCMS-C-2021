"""Model implementing ANN approximating ODE via RKNN integrator template.
"""
import numpy as np
import torch.nn as nn
from sw.IntegratorNN import IntegratorNN
from torch import reshape
from sw.numerical_integrator import *

class RKNN(nn.Module):
    def __init__(self,net1, net2):
        super(RKNN, self).__init__()
        self.model1 = net1
        #IntegratorNN.__init__(self,input_size, depth, output_size)
        self.model2 = net2
        #IntegratorNN.__init__(self,input_size, depth, output_size)
        print(type(self.model1))

    def forward(self, x, dt):
        x1 = self.model1(x)
        x2 = self.model2(x1 + 1/2*mul(dt,x1))
        return x1, x2

def training_step(model, train_loader, optimizer, criteria, batches):
    """Performs the training of one batch of the RKNN network. 
    Args:
        model (nn.Module): RKNN model 
        train_loader (nn.Dataloader): Dataloader for training set 
        optimizer (nn.optim): optimizer function 
        criteria (nn): loss function 
        batches (int): no. batches
    Returns:
        float: avg loss  
    """
    running_loss = 0.
    last_loss = 0.

    for i,data in enumerate(train_loader):
            xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()
            
            dt = time[:,1] - time[:,0]
            dt = reshape(dt, (dt.size(dim=0),1))
            k0, k1 = model(xk, dt)
            loss = criteria(xk1, runge_kutta(xk,k0,k1,dt))

            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            if i%batches == batches -1:
                last_loss = running_loss / batches # loss per batch
                running_loss = 0.
    return last_loss
    

def RKNN_training(model, epochs,train_loader, optimizer, criteria, val_loader, batches):
    """Perfoms cross-validation of the RKNN network
    Args:
        model (nn.Module): RKNN model 
        epochs (int): number of epochs
        train_loader (Dataloader): Dataloader of the training set 
        optimizer (nn.optim)): Optimizer function 
        criteria (nn): loss function
        val_loader (Dataloader): Dataloader of the validation set 
        batches (int): number of batches 
    Returns:
        np.array: array of validation loss and training loss per epoch 
    """
    
    loss_data = np.zeros([epochs,2],dtype=np.float64)

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_loss = training_step(model, train_loader, optimizer, criteria, batches)
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(val_loader):
            xk, xk1, time = vdata[:, :, 0].float(), vdata[:, :, 1].float(), vdata[:, :, 2].float()
            dt = time[:,1] - time[:,0]
            dt = reshape(dt, (dt.size(dim=0),1))
            vk0, vk1 = model(xk, dt)
            vxk1 = runge_kutta(xk,vk0,vk1,dt)
            vloss = criteria(xk1, vxk1)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        loss_data[epoch,0] = avg_loss
        loss_data[epoch,1] = avg_vloss
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    return loss_data