
"""Module loading, implementing and visualizing a function approximation of a data set using least square method and radial basis functions
"""
import numpy as np
from scipy.linalg import lstsq

def load_txt_data(filepath):
    """Loads to dimensional data set containing states and function values into numpy arrays
        (x,f(x))

    Args:
        filepath (string): filepath to txt file

    Returns:
        [np.array]: states
        [np.array]: function values
    """

    data = np.loadtxt(filepath)
    x = data[:,0]
    F = data[:,1]

    return x, F

def least_square(x,F,dim):
    """Finds linear system matrix A using least square minimalization, to fit function values
        Ax = F 

    Args:
        x (np.array): states
        F (np.array): function values
        dim (int): row dimension of the A matrix

    Returns:
        [m]: array representing factors Y = MX
    """
    A = np.vstack([x, np.zeros(dim)]).T # Do not add any offset
    return lstsq(A,F)[0]

def rbf(x, center, epsilon):
    """Define one radial function:
        rbf(x) = exp(-||xl-x||^2/ε^2)

    Args:
        x (np.array): states defined by data set
        center (float): defines center of the radial function
        epsilon (float): bandwidth of the radial function

    Returns:
        [np.array]: Array containing data points of the radial function given parameters
    """
    xl = np.ones(len(x))*center #Change back to len
    radial_func = np.exp(-(xl-x)**2/(epsilon**2))
    return radial_func

def rbf_matrix(x,L,epsilon,ss):
    """Creating the rbf matrix phi, for optimalization

    Args:
        x (np.array): states 
        L (int): Number of rbf
        epsilon (float): rbf parameter
        ss (lst): interval for placing rbf

    Returns:
        [np.ndarray]: The rbf matrix
    """
    phi = np.empty([L,len(x)])
    # Calculating range defined by state
    for (index, center) in np.ndenumerate(np.linspace(np.min(ss[0]),np.max(ss[1]),num=L)):
        radial_func = rbf(x,center,epsilon)
        phi[index[0],:] = radial_func
    return phi

def rbf_lincomb(x,c,L,epsilon,ss):
    """Calculate the linear combination between the radial basis functions and the coefficient C

    Args:
        x (np.array): states 
        c (np.array): Coefficients derived by optimization
        L (int): Number of rbf
        epsilon (float): rbf parameter
        ss (lst): interval for placing rbf

    Returns:
        [np.array]: The approximated states
        [np.ndarray]: The scaled radial basis functions
    """
    approx = np.zeros(len(x))
    radial = np.ndarray([L,len(x)])
    for (index, center) in np.ndenumerate(np.linspace(np.min(ss[0]),np.max(ss[1]),num=L)):
        radial_func = rbf(x,center,epsilon)*c[index]
        radial[index,:] = radial_func
        approx = approx + radial_func

    return approx, radial

def rbf_approx(x,F,phi,epsilon,ss):
    """Finding optimal linear combination of rbf functions to fit data F

    Args:
        x (np.array): States
        F (np.array): Function values
        phi (np.ndarray): rbf matrix
        epsilon (float): rbf parameter
        ss (lst): interval for placing rbf

    Returns:
        [np.array]: Approximated values
    """
    c = least_square(phi,F,phi.shape[1])
    approx, radial = rbf_lincomb(x,c,len(phi),epsilon,ss)
    return approx,radial