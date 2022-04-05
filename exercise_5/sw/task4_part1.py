"""Module for plotting data set against an arbitrary time unit, and also plotting a time series over a delayed version of same times series according to Takens theorem.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_coordinate_values_over_time(x,save_filepath=None):
    """
    this method plots a series of values against a arbitrary time unit.
    the time value will be the same size as the input array size.
    :param x (array)
    :param save_filepath (string)
    """
    fig = plt.figure()
    y = np.linspace(0, 999, len(x))
    plt.plot(y, x)
    plt.xlabel("line number")
    plt.ylabel("coordinate x")
    plt.title("takens_1: time series over first coordinate")

    if save_filepath:
        fig.savefig(save_filepath)

def plot_reconstruction_of_shadow_manifold(x,j=80,save_filepath=None):
    """
    This method plots a time series against a delayed version of the same values.
    This is done in order to reconstruct a shadow manifold based on the input array
    Normally 2d+1 delayed times series is needed but in some cases one dimension is enough.
    :param x (array)
    :param j (int) amount time series should be delayed.
    :param save_filepath (string)
    """
    fig = plt.figure()
    i = 0
    for coor in x:
        plt.plot(x[i - j], coor, "b.")
        i = i + 1

    plt.xlabel("coordinate x - ∆n")
    plt.ylabel("coordinate x")
    plt.title("Time-delayed embedding: ∆n = 80")
    if save_filepath:
        fig.savefig(save_filepath)
