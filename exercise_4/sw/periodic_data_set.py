"""Script generating periodic data set, and storing it in a csv file, returning the numpy array of the data
"""
from math import pi, cos, sin
import csv 
import pandas as pd

DATA_FILE_PATH = "datasets/periodic_data_set.csv"

def periodic_function(k,N):
    """Describing the data points
    y_k = [cos(t_k),sin(t_k)], t_k = 2πk/(N+1)

    Args:
        k (int): Current iteration
        N (int): Input data dimensionality
    """
    t_k = 2*pi*k/(N+1)
    return [cos(t_k),sin(t_k)]

def generate_csv(N): 
    """Generates a csv-file representing the data set and stores it in location given by DATA_FILE_PATH

    Args:
        N (int): Input data dimensionality
    """
    with open(DATA_FILE_PATH,'w') as data_file: 
        writer = csv.writer(data_file)
        writer.writerow(["x","y"])
        for i in range(1,N+1):
            data_point = periodic_function(i,N)
            data_point = map(str,data_point)
            writer.writerow(data_point)

def read_csv():
    """Read csv given by DATA_FILE_PATH into a np.array

    Returns:
        [np.ndarray]: np.ndarray representing all data points
    """
    periodic_df = pd.read_csv(DATA_FILE_PATH,delimiter=",")
    data_points = periodic_df.to_numpy()
    return data_points
    




