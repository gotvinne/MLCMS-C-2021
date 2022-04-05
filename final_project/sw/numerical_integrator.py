"""
Module describing numerical integrators
- Eulers method
- Runge Kutta, second order approximation
"""
import torch
from torch import mul


def eulers_method(x: torch.Tensor, x_dot: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Implementing ERK1 numerical approximation
    :param x: State
    :param x_dot: State derivative
    :param dt: Time step
    :return: Approximated next state
    """
    
    return x + mul(dt, x_dot)


def erk(x: torch.Tensor, dt: torch.Tensor, k: list, b: list) -> torch.Tensor:
    """
    Defining an general explicit Runge-Kutta method
    :rtype: The approximated next state
    """
    func_evaluations = 0
    for i in range(len(k)):
        func_evaluations += mul(b[i], k[i])
    return x + mul(dt, func_evaluations)


def runge_kutta(x: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Implementing a version of ERK2 numerical approximation
    :param x: State
    :param k0: ODE evaluation
    :param k1: ODE evaluation
    :param dt: Time step
    :return: Approximated next state
    """
    return x + 1 / 6 * mul(dt, (k0 + 2 * k1))
