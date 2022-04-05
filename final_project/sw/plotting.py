""" Plotting lib for final project
"""
import matplotlib.pyplot as plt
import numpy as np
from sw.simulation import simulatingNN
import torch.nn as nn
from sw.ERKNN import ERKNN

""" Constants defining the scenario layout
"""
WIDTH = 30
HEIGHT = 10
GOAL = 27
START = 10

OBSTACLE_WIDTH = 1
OBSTACLE_HEIGHT = 4.2
OBSTACLE_POS = 20


def plot_scatter(traj: np.ndarray, c: str = None, lab: str = "", title: str = None, save_filepath: str = None):
    """
    Plotting a scatter plot of data in traj
    :param traj: Trajectory
    :param c: color
    :param lab: label of the trajectory
    :param title: title of the plot
    :param save_filepath: Filepath to save the figure
    """
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(traj[0, :], traj[1, :], color=c, label=lab)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plot_scenario()
    if save_filepath:
        fig.savefig(save_filepath)


def vectorize_scenario(model: nn.Module):
    """
    Calculating the derivatives and position from the scenario space, in order to make phase portrait
    :param model: NN trained to approximate ODE
    :return: positions corresponding derivatives of the given scenario
    """
    X, Y = scenario_meshgrid()
    U, V = scenerio_vectorfield(X, Y)

    index = 0
    for j in range(len(Y)):
        for i in range(len(X)):
            pos = np.array([X[j, i], Y[j, i]])
            dp = simulatingNN(None, pos, model)
            U[index], V[index] = dp[0], dp[1]
            index += 1
    return X, Y, U, V


def phase_portraitNN(X: np.meshgrid, Y: np.meshgrid, U: np.ndarray, V: np.ndarray, save_filepath: str = None):
    """"
    Plotting a phase portrait of a NN
    :param X: numpy grid over x values of the scenario.
    :param Y: numpy grid over y values of the scenario.
    :param U: numpy vector contain all vector values for the scenario.
    :param V: numpy vector contain all vector values for the scenario.
    :param save_filepath: Filepath to save the figure
    """
    fig = plt.figure(figsize=(12, 8))
    plt.title("Phase portrait")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.quiver(X, Y, U, V, color='b', alpha=1.0)

    plot_scenario()

    if save_filepath:
        fig.savefig(save_filepath)


def phase_portraitERK(model: ERKNN):
    """
    Plotting the phase portrait of an ERK-model
    :param model: ERKNN
    """
    X, Y = scenario_meshgrid()
    U, V = scenerio_vectorfield(X, Y)
    for net in model.networks:
        x, y, u, v = vectorize_scenario(net)
        U = U + u
        V = V + v

    U = U / model.order
    V = V / model.order
    phase_portraitNN(X, Y, U, V)


def plot_scenario(w: float = WIDTH, h: float = HEIGHT, goal: float = GOAL, start: float = START,
                  ow: float = OBSTACLE_WIDTH, oh: float = OBSTACLE_HEIGHT,
                  obstacle: float = OBSTACLE_POS):
    """
    Plotting the scenario in a matplotlib.pyplot figure
    :param w: width of scenario
    :param h: height of
    :param goal: minimum x value defining the target
    :param start: maximum x value defining the source
    :param ow: width of obstacle
    :param oh: height of obstacle
    :param obstacle: minimum x value of obstacle
    """
    # Visualizing scenario
    plt.xlim([0, w])
    plt.ylim([0, h])
    plt.vlines(start, 0, h, colors="green", linestyles='dashed')
    plt.vlines(goal, 0, h, colors="orange", linestyles='dashed')
    # Obstacles
    plt.vlines(obstacle, 0, oh, colors="gray")
    plt.vlines(obstacle + ow, 0, oh, colors="gray")
    plt.hlines(oh, obstacle, obstacle + ow, colors="gray")

    plt.vlines(obstacle, h - oh, h, colors="gray")
    plt.vlines(obstacle + ow, h - oh, h, colors="gray")
    plt.hlines(h - oh, obstacle, obstacle + ow, colors="gray")


def plot_traj(num: int, traj: np.ndarray, save_filepath: str = None):
    """
    Plotting trajectories solved using solve_ivp() in a visualization of the scenario
    :param num: Number of pedestrian trajectories
    :param traj: trajectory data
    :param save_filepath: Filepath to save the figure
    """
    fig = plt.figure(figsize=(12, 8))

    for i in range(num):
        plt.plot(traj[2 * i, :], traj[2 * i + 1, :], label="Trajectory" + str(i))

    plt.title("Estimating position of gravity - Euler template")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plot_scenario()

    if save_filepath:
        fig.savefig(save_filepath)


def plot_batch_mse(loss_data: np.array, save_filepath: str = None):
    """
    Plotting the mse loss for every training batch
    :param loss_data: training loss
    :param save_filepath: Filepath to save the figure
    """
    x_axis = np.arange(0, len(loss_data))
    fig = plt.figure(figsize=(12, 12))
    plt.plot(x_axis, loss_data, label="Training loss")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("Batches")
    if save_filepath:
        fig.savefig(save_filepath)


def plot_rknn_mse(loss_data: np.array, save_filepath: str = None):
    """
    Plotting the cross-validated loss of the RKNN
    :param loss_data: loss data
    :param save_filepath: Filepath to be saved
    """
    x_axis = np.arange(0, len(loss_data))
    fig = plt.figure(figsize=(12, 12))
    plt.plot(x_axis, loss_data[:, 0], label="Training loss")
    plt.plot(x_axis, loss_data[:, 1], label="Validation loss")
    plt.legend()
    plt.ylabel("MSE Runge-Kutta")
    plt.xlabel("Epochs")
    if save_filepath:
        fig.savefig(save_filepath)


def scenario_meshgrid(w: float = WIDTH, h: float = HEIGHT):
    """
    :param w: float, predefined to be the width of the scenario from Vadere
    :param h: float, predefined to be the higth of the scenario from Vadere
    :return: X and Y which is a np.meshgrid corresponding to the size of the scenario.
    """
    x = np.linspace(0, w)
    y = np.linspace(0, h)
    X, Y = np.meshgrid(x, y)
    return X, Y


def scenerio_vectorfield(X: np.meshgrid, Y: np.meshgrid):
    """
    :param X: np.meshgrid, a meshgrid used to instantiate the U vector filled with 0 with size X*X
    :param Y: np.meshgrid, a meshgrid used to instantiate the V vector filled with 0 with size Y*Y
    :return: U, V: two vectors of size (X*X) and (Y*Y)
    """
    U = np.zeros((len(X) * len(X)))
    V = np.zeros((len(Y) * len(Y)))
    return U, V
