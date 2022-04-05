import matplotlib.pyplot as plt
import numpy as np

def bifurcation_plot(alpha,start_alpha,pos_equli, neg_equli):
    """Plots the bifurcation plot for x_dot = alpha - a*x + b
    """
    plt.plot(alpha, pos_equli, '-', label='stable')
    plt.plot(alpha, neg_equli, '-', label='unstable')
    plt.plot(start_alpha, 0, 'x', label='bifurcation')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('x')
    plt.xlim(-2, 8)
    plt.legend()
    plt.show()

def nonlinear_equilibria(alpha, a, b):
    """Generalized equation solver for alpha - ax^2-b = 0

    Returns:
        tuple: +/- x and b(starting point)
    """
    #negate all alpha values lower than b
    alpha_values = np.where(alpha < b, np.nan, alpha)
    x = np.sqrt((alpha_values - b)/a)

    return b, x, -x