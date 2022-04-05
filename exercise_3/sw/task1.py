
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sw.utils.plot_utils import generate_field_vectors, configuration, add_orbit

def linear_system(t,vec,A):
    """Represents a system of linear ordinary differential equations

    Args:
        t (int): Variable used for scipy's ivp solver
        vec (np.array): Input vector
        A (np.array): linear system

    Returns:
        np.array: np.array with the updated state. 
    """
    return np.matmul(A,vec)

def plot_phase_portrait(A,e_val,alpha):
    """Plot phase portraits
    Args:
        A (np.array): Linear system matrix
        e_val (np.complex): complex value
        plt_str (string): Defines visualisation of orbit
        flow (Bunch object, scipy.ivp_integrate()): Object holding the solution of an ODE
        alpha (float): Bifurcation variable
    """
    x1 = np.linspace(-1.0, 1.0, 15)
    x2 = np.linspace(-1.0, 1.0, 15)

    X1, X2 = np.meshgrid(x1, x2)
    u, v = np.zeros(X1.shape), np.zeros(X2.shape)
    NI, NJ = X1.shape

    generate_field_vectors(NI,NJ,X1,X2,u,v,A,linear_system)

    titlestring = "α=" + str(np.round(alpha, 3)) + ", λ1=" + str(np.round(e_val[0], 3)) + ", λ2=" + str(
        np.round(e_val[1], 3))

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X1, X2, u, v, density=[0.5, 1])
    ax0.set_title(titlestring)

            
def pole_plot_str(e_val):
    plt_str = "b-"
    if (np.iscomplex(e_val[0])):
        plt_str = "r-"
    return plt_str