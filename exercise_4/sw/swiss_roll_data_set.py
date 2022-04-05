"""Script generating the swiss roll data set from pythons scipy library.
"""

from sklearn import datasets as sklearn
import csv 
import pandas as pd
import numpy as np

DATA_FILE_PATH = "datasets/swiss_roll_data_set.csv"

def generate_csv(N):
    """Generates "swiss-roll" manifold data-set and writes it to csv-file in DATA_FILE_PATH 

    Args:
        N (int): size of data set
    """
    X,t = sklearn.make_swiss_roll(N)
    with open(DATA_FILE_PATH,'w') as data_file: 
        writer = csv.writer(data_file)
        writer.writerow(["x","y","z","unvariate_pos"])
        for i in range(N):
           writer.writerow([X[i][0],X[i][1],X[i][2],t[i]])

def read_csv():
    """Reads the csv file and returns a np.ndarray representing it

    Returns:
        np.ndarray: numpy represention of the data 
        np.array: unvariate data points
    """
    data_points = np.empty(0)
    unvariate_points = np.empty(0)
    swiss_roll_df = pd.read_csv(DATA_FILE_PATH,delimiter=",")
    rows = len(swiss_roll_df)
    columns = len(swiss_roll_df.columns)

    for (_,row) in swiss_roll_df.iterrows():
        data_points = np.append(data_points,[row["x"],row["y"],row["z"]])
        unvariate_points = np.append(unvariate_points,row["unvariate_pos"])
    return data_points.reshape(rows,columns-1),unvariate_points
