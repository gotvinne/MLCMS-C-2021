from scipy.integrate import odeint
import numpy as np 


def lorenz(state, t):
    """Calculates the lorentz step
    Args:
        x (float): Current x value
        y (float): Current y value
        z (float): Current z value
        sigma (float): Parameter defining Lorenz system
        rho (float): Parameter defining Lorenz system
        beta (float): Parameter defining Lorenz system

    Returns:
        [float, float, float]: Returns the step for all states
    """
    x,y,z = state
    sigma, rho, beta = 10, 28,8/3
    x_dot = sigma*(y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z
    return x_dot, y_dot, z_dot


def solver(SIM0, t):
    """Solves the lorenz system using odeint 

    Args:
        SIM0 (array): initial state 
        t (array): time steps 
    Returns:
        [array]: full time series 
    """
    return odeint(lorenz, SIM0, t)


def time_delay_lorenz(states, dt):
    """
    This function computes the time delayed embedding of the lorenz attractor
    :param x: data
    :param dn: time-delay
    """
    x_t = states
    x_t_delta_t = np.roll(states, dt)
    x_t_delta_2t = np.roll(states, 2*dt)
    return x_t, x_t_delta_t, x_t_delta_2t

