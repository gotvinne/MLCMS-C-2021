"""Model implementing ANN approximating ODE via Euler integrator template.
"""
import numpy as np
import torch.utils.data
from torch import reshape
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from sw.numerical_integrator import eulers_method


class IntegratorNN(nn.Module):
    """
        A class used to represent a simple integrator template
        ...

        Attributes
        ----------
        lx : nn.Linear
            x linear layers defining the neural network

        Methods
        -------
        forward(x: torch.Tensor)
            Defines the forwarding of the data through the network
    """

    def __init__(self, input_size: int, depth: int, output_size: int):
        """
        Constructing the integrator template
        :param input_size: Defining the input dimension to the NN
        :param depth: Number of nodes per layer
        :param output_size: Defining the output dimension to the NN
        """
        super().__init__()
        self.l1 = nn.Linear(input_size, depth)
        self.l2 = nn.Linear(depth, depth)
        self.l3 = nn.Linear(depth, depth)
        self.l4 = nn.Linear(depth, depth)
        self.l5 = nn.Linear(depth, depth)
        self.l6 = nn.Linear(depth, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forwarding of the data through the network
        :param x: State
        :return: The output of the network
        """
        x = func.leaky_relu(self.l1(x))
        x = func.leaky_relu(self.l2(x))
        x = func.leaky_relu(self.l3(x))
        x = func.leaky_relu(self.l4(x))
        x = func.leaky_relu(self.l5(x))
        x = self.l6(x)
        return x


def integrator_training(model: nn.Module, lr: float, epochs: int, train_set: torch.utils.data.DataLoader) -> np.array:
    """
    Defining the training algorithm for the IntegratorNN class
    :param model: Model trained to approximate ODE
    :param lr: learning rate
    :param epochs: number of epochs used in training
    :param train_set: training data
    :return: The training loss for every batch
    """
    optimizer = optim.Adam(model.parameters(), lr)
    loss_data = np.zeros([epochs * len(train_set)], dtype=np.float64)

    batch_nr = 0
    for _ in range(epochs):
        for data in train_set:
            xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()

            optimizer.zero_grad()
            dt = time[:, 1] - time[:, 0]
            dt = reshape(dt, (dt.size(dim=0), 1))
            output = model(xk)

            loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
            loss_data[batch_nr] = loss
            batch_nr += 1
            loss.backward()
            optimizer.step()
    return loss_data

def calculate_val_loss(model: nn.Module, val_set: torch.utils.data.DataLoader) -> float:
    """
    Calculation of the validation loss
    :param model: Model trained to approximate ODE
    :param val_set: validation set
    :return: average loss
    """
    size = len(val_set)
    val_loss = 0

    for data in val_set:
        xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()
        dt = time[:, 1] - time[:, 0]
        dt = reshape(dt, (dt.size(dim=0), 1))
        output = model(xk)
        loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
        val_loss += loss
    return val_loss / size


def integrator_validating(model: nn.Module, lr: float, train_set: torch.utils.data.DataLoader,
               val_set: torch.utils.data.DataLoader) -> object:
    """
    Validating the model by training until validation loss is minimized
    :param model: Model trained to approximate ODE
    :param model: NN trained to approximate ODE
    :param lr: learning rate
    :param train_set: training set
    :param val_set: validation set
    :return: average loss and number of additional training steps
    """
    val_loss = float('inf')
    train_step = 0
    while True:
        current_loss = calculate_val_loss(model, val_set)
        if (current_loss < val_loss):
            val_loss = current_loss
            train_step += 1
            integrator_training(model, lr, 1, train_set)
        else:
            break
    return val_loss, train_step

def integrator_bifurcation_training(model: nn.Module, lr: float, epochs: int, train_set: torch.utils.data.DataLoader,alpha) -> np.array:
    """
    Defining the training algorithm for the IntegratorNN class
    :param model: Model trained to approximate ODE
    :param lr: learning rate
    :param epochs: number of epochs used in training
    :param train_set: training data
    :return: The training loss for every batch
    """
    optimizer = optim.Adam(model.parameters(), lr)
    loss_data = np.zeros([epochs * len(train_set)], dtype=np.float64)
    

    batch_nr = 0
    for _ in range(epochs):
        for data in train_set:
            xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()
            
            n = np.zeros(shape=(xk.shape[0], 1), dtype=np.float32) 
            for i in range (xk.shape[0]):
                n[i] = alpha
            alpha_tensor= torch.tensor(n)
            tensor_input = torch.cat((xk,alpha_tensor), 1)
            
            optimizer.zero_grad()
            dt = time[:, 1] - time[:, 0]
            dt = reshape(dt, (dt.size(dim=0), 1))
            output = model(tensor_input)

            loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
            loss_data[batch_nr] = loss
            batch_nr += 1
            loss.backward()
            optimizer.step()
    return loss_data

def integrator_bifurcation_validating(model: nn.Module, lr: float, train_set: torch.utils.data.DataLoader,
               val_set: torch.utils.data.DataLoader) -> object:
    """
    Validating the model by training until validation loss is minimized
    :param model: Model trained to approximate ODE
    :param model: NN trained to approximate ODE
    :param lr: learning rate
    :param train_set: training set
    :param val_set: validation set
    :return: average loss and number of additional training steps
    """
    val_loss = float('inf')
    train_step = 0
    while True:
        current_loss = calculate_val_loss(model, val_set)
        if (current_loss < val_loss):
            val_loss = current_loss
            train_step += 1
            integrator_training(model, lr, 1, train_set)
        else:
            break
    return val_loss, train_step

def integrator_bifurcation_validating(model: nn.Module, lr: float, train_set: torch.utils.data.DataLoader,
               val_set: torch.utils.data.DataLoader, alpha) -> object:
    """
    Validating the model by training until validation loss is minimized
    :param model: Model trained to approximate ODE
    :param model: NN trained to approximate ODE
    :param lr: learning rate
    :param train_set: training set
    :param val_set: validation set
    :return: average loss and number of additional training steps
    """
    val_loss = float('inf')
    train_step = 0
    while True:
        current_loss = calculate_bi_val_loss(model, val_set, alpha)
        if (current_loss < val_loss):
            val_loss = current_loss
            train_step += 1
            integrator_bifurcation_training(model, lr, 1, train_set,alpha)
        else:
            break
    return val_loss, train_step

def calculate_bi_val_loss(model: nn.Module, val_set: torch.utils.data.DataLoader, alpha) -> float:
    """
    Calculation of the validation loss
    :param model: Model trained to approximate ODE
    :param val_set: validation set
    :return: average loss
    """
    size = len(val_set)
    val_loss = 0

    for data in val_set:
        xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()
        
        n = np.zeros(shape=(xk.shape[0], 1), dtype=np.float32) 
        for i in range (xk.shape[0]):
            n[i] = alpha
        alpha_tensor= torch.tensor(n)
        tensor_input = torch.cat((xk,alpha_tensor), 1)
        
        dt = time[:, 1] - time[:, 0]
        dt = reshape(dt, (dt.size(dim=0), 1))
        output = model(tensor_input)
        loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
        val_loss += loss
        
    return val_loss / size
