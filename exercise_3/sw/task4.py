import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    """Logistic map dynamics

    Args:
        x (float): x state value
        r (float): bifurcation variable

    Returns:
        float: Logistic step
    """
    return r * x * (1 - x)

def plot_configuration(r, x0):
    """Configurates a orbit plot

    Args:
        r (float): [description]
        x0 (float): Initial value
    """
    plt.figure()
    plt.title("r = " + str(r) + ", x0 = " + str(x0))
    plt.xlabel('$t$')
    plt.ylabel('$x$')
     
def logistic_bifurcations(r, rend, x0, tend):
    """Plots logistic bifurcations from an interval of the bifurcation variable

    Args:
        r (float): [description]
        rend (float): [description]
        x0 (float): [description]
        tend (float): 
    """
    t = np.arange(0, tend)
    for r in np.arange(r, rend + 0.25, 0.25):
        k = x0
        x = np.zeros(tend)
        for time_step in t:
            x[time_step] = k
            k = logistic(k, r)

        plot_configuration(r,x0)
        plt.plot(t, x, "-b") 

def lorenz(x,y,z,sigma,rho,beta):
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
    x_dot = sigma*(y - x)
    y_dot = rho*x - y - x*z
    z_dot = x*y - beta*z
    return x_dot, y_dot, z_dot

def random_guess():
    """Returns random float in range [0.0,1.0)
    """
    return np.random.random()

def logistic_bifurcation_diagram(rbegin, rend, x0, tend, rincrement):
    """Plots the bifurcation diagram of the logistic map, by randomly simulate the system dynamics for a random initial value

    Args:
        rbegin (float): Bifurcation start value
        rend (float): Bifurcation end value
        x0 (float): initial state value
        tend (float): Simulation end time 
        rincrement (int): The increment of the bifurcation variable
    """
    r = np.arange(rbegin, rend + rincrement, rincrement)
    x = np.zeros(r.shape[0])
    t = np.arange(0, tend)

    for index, rvalue in np.ndenumerate(r):
        k = random_guess()
        for _ in t:
            k = logistic(k, rvalue)
        x[index] = k
    logistic_plot(r, x, x0)

def logistic_plot(r, x, x0):
    """Plot for the logistic bifurcation diagram

    Args:
        r ([type]): [description]
        x ([type]): [description]
        x0 ([type]): [description]
    """
    plt.figure(figsize=(5, 3), dpi=200)
    plt.title("Bifurcation diagram, x0 = " + str(x0))
    plt.xlabel('$r$')
    plt.ylabel('$x$')
    plt.plot(r, x, ",", color="k") 
    plt.show()

def lorenz_equilibrium(rho, beta):
    """Calculates the equilibrium

    Args:
        rho (np.array): Empty list
        beta (float): System parameter

    Returns:
        list: Equilibrium of the lorenz system
    """
    rho = np.where(rho < 1, np.nan, rho)
    x = np.sqrt(beta*(rho-1))
    z = rho-1
    return x,-x,z

def plot_equilibria(rho, x_pos,x_neg,z):
    """Plot equilibria of state x, y and z.
    Args:
        rho (list): bifurcation values
        x_pos (list): x values
        x_neg (list): y values
        z (list): z values
    """
    plt.plot(rho, x_pos, '-', label='x=y')
    plt.plot(rho, x_neg, '-', label='x=y')
    plt.plot(rho,z, '-', label='z')
    plt.xlabel('$rho$')
    plt.ylabel('x')
    plt.xlim(0, 50)
    plt.ylim(-20,20)
    plt.legend()
    plt.show()

def plot_difference(dt,num_steps,diff,xlabel,ylabel):
    """[summary]

    Args:
        dt (float): Time increment
        num_steps (int): Number of steps
        diff (list): Difference values
        xlabel (string): x label
        ylabel (string): label
    """
    plt.figure()
    t = np.linspace(0, dt * num_steps, num_steps + 1)
    plt.plot(t, diff)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
