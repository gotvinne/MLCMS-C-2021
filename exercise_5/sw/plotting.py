"""Module simplifying the plotting funtionality found in matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
from sw.vector_approx import *

def plot_approximation(x,F,approx,title,ls="-",save_filepath=None):
    """Plotting the data set and the approximation

    Args:
        x (np.array): data set states
        F (np.array): data set function values 
        approx (np.array): Array holding the approximated values
        title (string): Plot title
        ls (string): Line style for plotting
        save_filepath (string, optional): filepath to be saved. Defaults to None.
    """
    fig = plt.figure()
    plt.plot(x,F,"r*",label="Data set")
    plt.plot(x,approx,ls,label="Approximation")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title(title)

    if save_filepath:
        fig.savefig(save_filepath)

def plot_rbf(x,phi,title,save_filepath=None):
    """Plotting rbf

    Args:
        x (np.array): states
        phi (np.ndarray): rbf matrix
        title (string): Title of plot 
        save_filepath (string, optional): filepath to be saved. Defaults to None.
    """
    fig = plt.figure()
    for i in range(len(phi)):
        plt.plot(x,phi[i,:],"*")
    
    plt.xlabel("x")
    plt.ylabel("φ_l")
    plt.title(title)
    
    if save_filepath:
        fig.savefig(save_filepath)

def phase_portrait(x0, T, dt, A, ss=10, save_filepath=None):
    """Creates a phase portrait for the approximation along with the estimated trajectory from 
        one point x0

    Args:
        x0 (array): initial point
        T (int): end time
        dt (float): time step
        A (array): matrix for linear transformation
    """
    # Define statespace
    x1 = np.linspace(-ss, ss, 50)
    x2 = np.linspace(-ss, ss, 50)

    X1, X2 = np.meshgrid(x1,x2)

    # phase portrait and trajectory 
    u = A[0][0] * X1 + A[1][0] * X2
    v = A[0][1] * X1 + A[1][1] * X2

    traj = np.zeros(shape=(int(T/dt),2))
    for i in range(int(T/dt)):
        traj[i]=(x0)
        x0 = dt * (x0 @ A) + x0
    traj=traj.T

    #Plot the phase portrait
    plt.figure(figsize=(6,6))
    plt.streamplot(X1, X2, u, v, linewidth=0.8)
    if (save_filepath):
        plt.savefig(save_filepath)
    plt.plot(traj[0], traj[1])

def plot_scatter(x,y,c=None,lab="",xlim=None,ylim=None):
    """Plott two sets of data in a scatter plot

    Args:
        x (np.array): x-data set
        y (np.array): y-data set
        c (str, optional): Color of the scatter points. Defaults to None.
        lab (str, optional): Label of the data set. Defaults to "".
        xlim (float, optional): Sets the upper and lower x-value of the plot. Defaults to None.
        ylim (float, optional): Sets the upper and lower y-value of the plot. Defaults to None.
    """
    plt.scatter(x,y,color=c,label=lab)

    if (xlim and ylim):
        plt.xlim = (xlim)
        plt.ylim = (ylim)

    if (lab):
        plt.legend()

def plot_spesifics(fig,x_lab="",y_lab="",title="",save_filepath=None):
    """Define the characteristics of a plot

    Args:
        fig (matplotlib.figure): The figure object
        x_lab (str, optional): Label of x-axis. Defaults to "".
        y_lab (str, optional): Label of y-axis. Defaults to "".
        title (str, optional): Title of the figure. Defaults to "".
        save_filepath (str, optional): The filepath to save the figure. Defaults to None.
    """
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.title(title)

    if (save_filepath):
        fig.savefig(save_filepath)