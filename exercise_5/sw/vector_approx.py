""" Module implementing software in order to approximate vector fields. 
"""
import numpy as np 
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from sw.function_approx import rbf_lincomb, rbf_matrix, rbf
from scipy.integrate import solve_ivp

def least_squares_vector(x0,x1,dt):
    """uses least squares to approximate a linear transformation between x0 and x1

    Args:
        x0 (array): initial vector field
        x1 (array): vector field at t_1
        dt (float): time step 

    Returns:
        array: approximated matrix 
    """
    v_hat = (x1-x0)/dt
    return np.linalg.lstsq(x0,v_hat, rcond=10000)[0].T

def approx_vector(x0,x1, dt, A): # Better function naming?
    """Approximates x1 using A*x0 and computes MSE 

    Args:
        x0 (array): initial vector field
        x1 (array): vectorfield at t_1
        dt (float): time step
        A (array): matrix for linear transformation

    Returns:
        array: (x1_hat, MSE) approximated x1 and MSE between x1 and x1_hat
    """
    x1_hat = (dt*(A@x0.T) + x0.T).T 
    MSE = np.mean((x1_hat-x1)**2)

    plt.scatter(x0[:,0], x1[:,0], c='blue', s=1, alpha=1, label='Data')
    plt.scatter(x0[:,0], x1_hat[:,0], c='orange', s=1, alpha=1, label='Approximated function')
    plt.xlabel('x1')
    plt.ylabel('f(x1)')
    plt.legend()
    plt.savefig("figures/task_2/approx_fig_x.png")
    plt.show()

    plt.scatter(x0[:,1], x1[:,1], c='blue', s=1, alpha=1, label='Data')
    plt.scatter(x0[:,1], x1_hat[:,1], c='orange', s=1, alpha=1, label='Approximated function')
    plt.xlabel('x2')
    plt.ylabel('f(x2)')
    plt.legend()
    plt.savefig("figures/task_2/approx_fig_y.png")
    plt.show()
    return x1_hat, MSE 

def linear_system(t,x,A):
    """Linear system representation, the arguments are required from solve_ivp()

    Args:
        t (float): time
        x (np.array): states
        A (np.ndarray): linear system matrix

    Returns:
        [np.ndarray]: The matrix multiplication of A and x, the dimensions need to agree.
    """
    return np.dot(A,x)
    
def forecast_linear_system(A,x0,dt):
    """Solving a linear system for a time dt, returning the position of the states

    Args:
        A (np.ndarray): Linear system matrix
        x0 (np.array): states
        dt (float): simulation time

    Returns:
        [np.array]: forcasted states
    """
    x1 = np.zeros(x0.shape)
    for i in range(len(x0)):
        # Solving ODE for each data point
        sol = solve_ivp(linear_system,[0,dt],np.array([x0[i,0],x0[i,1]]),args=[A])
        x1[i,:] = sol.y[:,-1]
    return x1

def optimal_dt(x0,x1,dt_array):
    """Finding the optimal time step, minimizing error

    Args:
        x0 (np.array): initial states
        x1 (np.array): forecasted states
        dt_array (np.array): time values to be evaluated

    Returns:
        [np.array]: mse of the evaluated time points.
    """
    mse = np.zeros(len(dt_array))
    for (index,dt) in np.ndenumerate(dt_array):
        A = least_squares_vector(x0,x1,dt)
        X1_hat = forecast_linear_system(A,x0,dt)
        mse[index] = np.mean(np.square(X1_hat-x1))
    return mse

def rbf_approx_vector(x,F,L,epsilon,ss):
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
    phi = rbf_matrix(x,L,epsilon,ss).T
    c = lstsq(phi,F)[0].T
    approx, radial = rbf_lincomb(x,c,L,epsilon,ss)
    return c, approx, radial

def rbf_system(t,x_rbf,C,epsilon,ss):
    """Function forecasting a rbf system used in solve_ivp()

    Args:
        t (float): Parameter needed for solve_ivp()
        x_rbf (np.array): states
        C (np.ndarray): Matrix containing rbf optimized coefficients
        epsilon (float): rbf parameter
        ss (lst): interval for placing rbf

    Returns:
        [np.array]: Returning forcasted state
    """
    x_dot = np.zeros(x_rbf.shape)
    for (index, center) in np.ndenumerate(np.linspace(np.min(ss[0]),np.max(ss[1]),num=C.shape[1])):
        xl = np.ones(len(x_rbf))*center 
        radial_func = np.exp(-(xl-x_rbf)**2/(epsilon**2))
        x_dot[0] += radial_func[0]*C[0,index]
        x_dot[1] += radial_func[1]*C[1,index]
    return x_dot
