""" Module creating data from csv
"""
import pandas as pd
import numpy as np
from numpy.random import shuffle


def df_index(columns: np.array, string: str) -> int:
    """
    Indexing of a pandas.Dataframe
    :param columns: describing the data of a pandas.Dataframe
    :param string: Data string
    :return: row index of the data described by string
    """
    return columns.get_indexer_for([string])[0]


def fetch_training_data(data_filepath: str):
    """
    Fetching data from csv filepath
    :param data_filepath: Filepath to the csv-file
    :return: array describing the columns and the data set transposed
    """
    df = pd.read_csv(data_filepath, delimiter=" ")
    data = df.to_numpy()
    shuffle(data)
    return df.columns, data.T


def fill_ndarray(num_data: int, columns: np.array, data: np.ndarray, start: int = 0, end: int = None) -> np.ndarray:
    """
    Reshaping data
    :param num_data: number of data points
    :param columns: array describing the data columns
    :param data: matrix holding the data to be formated
    :param start: start index
    :param end: end index
    :return: The formated data
    """
    data = np.concatenate((data[df_index(columns, "startX-PID1"), start:end],
                           data[df_index(columns, "startY-PID1"), start:end],
                           data[df_index(columns, "endX-PID1"), start:end],
                           data[df_index(columns, "endY-PID1"), start:end],
                           data[df_index(columns, "simTime"), start:end],
                           data[df_index(columns, "endTime-PID1"), start:end]))
    return data.reshape((num_data, 2, 3), order="F")


def get_train_and_val_data(filepath: str, train_percent: int):
    """
    Extracting train and validation data from data set
    :param filepath: Filepath to the data set
    :param train_percent: percentage describing the radio train-val
    :return: training data and validation data
    """
    columns, data = fetch_training_data(filepath)
    num_train = (data.shape[1] * train_percent) // 100

    train_data = fill_ndarray(num_train, columns, data, end=num_train)
    val_data = fill_ndarray(data.shape[1] - num_train, columns, data, start=num_train)
    return train_data, val_data


def get_sim_data(filepath: str, ped_number: int):
    """
    Extracing the trajectory from a given pedestrian in the data set
    :param filepath: Filepath to the data set
    :param ped_number: PedestrianId
    :return: Trajectory of the spesified pedestrian and number of simulation steps
    """
    df = pd.read_csv(filepath, delimiter=" ")
    org_df = df.groupby("pedestrianId")
    sim_steps = org_df.size()

    traj = np.zeros([2, sim_steps[ped_number]])
    traj[0, :] = org_df.get_group(ped_number)["startX-PID1"].to_numpy()
    traj[1, :] = org_df.get_group(ped_number)["startY-PID1"].to_numpy()
    return traj.T, sim_steps


def get_bif_train_and_val_data(filepath: str, train_percent: int, alpha: float):
    """
    Extracting train and validation data from data set, and adding bifurcation parameter
    :param filepath: Filepath to the data set
    :param train_percent: percentage describing the radio train-val
    :param alpha: bifurcation parameter 
    :return: training data and validation data
    """
    columns, data = fetch_training_data(filepath)
    num_train = (data.shape[1] * train_percent) // 100

    train_data = fill_ndarray(num_train, columns, data, end=num_train)
    val_data = fill_ndarray(data.shape[1] - num_train, columns, data, start=num_train)

    train_alpha = np.full((num_train,1,3),alpha)
    val_alpha = np.full((data.shape[1]-num_train,1,3),alpha)

    train_data = np.concatenate((train_data, train_alpha), axis=1)

    val_data = np.concatenate((val_data, val_alpha), axis=1)

    return train_data, val_data
