"""Model implementing ANN approximating ODE via Explicit Runge-Kutta integrator template.
"""
from torch import mul, reshape
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
import numpy as np

from sw.IntegratorNN import *
from sw.numerical_integrator import erk


class ERKNN():
    """
            A class used to represent a general explicit Runge-Kutta method
            ...

            Attributes
            ----------
            integrator_dict : dictionary defining the integrator method
            order : Number of neural networks needed to implement the method
            networks : list containing all the neural networks

            Methods
            -------
            forward(x: torch.Tensor, dt: torch.Tensor)
                Defines the forwarding of the data through the network
            validate_method()
                Checks if integrator method is valid
        """

    def __init__(self, input_size: int, depth: int, output_size: int, integrator_dict: dict):
        """
        Initializing the object along with the neural networks
        :param input_size: Defining the input dimension to the NN
        :param depth: Number of nodes per layer
        :param output_size: Defining the output dimension to the NN
        :param integrator_dict: Defining the explicit integrator method
        """
        self.integrator_dict = integrator_dict

        valid, self.order = self.validate_method()
        if not valid:
            raise ValueError("Invalid integration method! \n")

        self.networks = []
        for _ in range(self.order):
            self.networks.append(IntegratorNN(input_size, depth, output_size))

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Defines the forwarding of the data through the network
        :param x: State
        :param dt: Time step
        :return: The output of the network of networks
        """
        output = []
        for i, net in enumerate(self.networks):
            if i == 0:
                output.append(net(x))
            else:
                output.append(net(x + self.integrator_dict["a"][i] * mul(dt, output[i - 1])))
        return output

    def validate_method(self):
        a_list = self.integrator_dict["a"]
        b_list = self.integrator_dict["b"]

        return len(a_list) == len(b_list), len(a_list)


def erk_training(model: ERKNN, lr: float, epochs: int, train_set: torch.utils.data.DataLoader) -> np.array:
    """
    Defining the training algorithm of the network
    :rtype: The loss from each batch
    """
    parameters = list()
    for net in model.networks:
        parameters += net.parameters()
    optimizer = optim.Adam(parameters, lr)

    loss_data = np.zeros([epochs * len(train_set)], dtype=np.float64)
    batch_nr = 0
    for _ in range(epochs):
        for data in train_set:
            xk, xk1, time = data[:, :, 0].float(), data[:, :, 1].float(), data[:, :, 2].float()

            dt = time[:, 1] - time[:, 0]
            dt = reshape(dt, (dt.size(dim=0), 1))
            output = model.forward(xk, dt)
            optimizer.zero_grad()

            loss = func.mse_loss(xk1, erk(xk, dt, output, model.integrator_dict["b"]))
            loss_data[batch_nr] = loss
            batch_nr += 1
            loss.backward()
            optimizer.step()
    return loss_data


def erk_val_loss(model: ERKNN, val_set: torch.utils.data.DataLoader) -> float:
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
        output = model.forward(xk, dt)
        loss = func.mse_loss(xk1, erk(xk, dt, output, model.integrator_dict["b"]))
        val_loss += loss
    return val_loss / size


def erk_validating(model: ERKNN, lr: float, train_set: torch.utils.data.DataLoader,
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
        current_loss = erk_val_loss(model, val_set)
        if (current_loss < val_loss):
            val_loss = current_loss
            train_step += 1
            erk_training(model, lr, 1, train_set)
        else:
            break
    return val_loss, train_step
