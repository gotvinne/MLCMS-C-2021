import numpy as np
from sympy import solve, Eq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sw.utils.plot_utils import generate_field_vectors, configuration, add_orbit

def AndronovHopf(t,vec,alpha):
    """Defines Andronov-Hopf dynamical system
    Args:
        t (int): Variable used for scipy's ivp solver
        vec (np.array): Input vector
        alpha (int): bifurcation value

    Returns:
        list: Andronov-Hopf update step
    """
    dx1 = alpha*vec[0]-vec[1]-vec[0]*(vec[0]**2+vec[1]**2)
    dx2 = vec[0]+alpha*vec[1]-vec[1]*(vec[0]**2+vec[1]**2)
    return [dx1, dx2]

def plot_phase_portrait(alpha):
    """Plot phase portraits

    Args:
        alpha (float): Bifurcation variable
        flow (Bunch object, scipy.ivp_integrate()): Object holding the solution of an ODE
        streamplot (bool): Plot stream plot
    """
    
    x1 = np.linspace(-2.0, 2.0, 15)
    x2 = np.linspace(-2.0, 2.0, 15)

    X1, X2 = np.meshgrid(x1, x2)
    u, v = np.zeros(X1.shape), np.zeros(X2.shape)
    NI, NJ = X1.shape

    generate_field_vectors(NI,NJ,X1,X2,u,v,alpha,func=AndronovHopf)
    
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X1, X2, u, v)
    ax0.set_title("α=" + str(round(alpha, 2)));

        
def plot_bifurcation_surface(x_axis,a1_axis,a2_axis,rotation1=None,rotation2=None,xlabel="α_2",ylabel="α_1",zlabel="x"):
    """
    Args:
        x_axis ([type]): [description]
        a1_axis ([type]): [description]
        a2_axis ([type]): [description]
        rotation1 ([type], optional): [description]. Defaults to None.
        rotation2 ([type], optional): [description]. Defaults to None.
        xlabel (str, optional): [description]. Defaults to "α_2".
        ylabel (str, optional): [description]. Defaults to "α_1".
        zlabel (str, optional): [description]. Defaults to "x".
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a1_axis, a2_axis, x_axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if (rotation1 or rotation2):
        ax.view_init(rotation1, rotation2)
    plt.show()
    
def cusp(x, alpha):
    """Cusp dynamics
    Args:
        x (float): State value
        alpha (list): defining the alphas 

    Returns:
        float: The Cusp update step
    """
    return alpha[0]+alpha[1]*x-x**3

def calculate_bifurcation_surface(a1_axis,a2_axis,x_axis,a1):
    """
    Args:
        a1_axis (list): list of alpha_1 values
        a2_axis (list): list of alpha_2 values
        x_axis (list): list of x values
        a1 (Sympy variable): Values used for sympy's solve
    """

    for a2 in np.arange(-1, 1, 0.05):
        for x in np.arange(-1, 1, 0.05):
            # Solve equation:
            eq = Eq(cusp(x, [a1, a2]), 0)
            solution = solve(eq, a1)

            # Store solutions
            a1_axis.append(solution[0])
            a2_axis.append(a2)
            x_axis.append(x)
