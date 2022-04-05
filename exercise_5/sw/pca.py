import numpy as np
from scipy.linalg import svd
from matplotlib import pyplot as plt

def pca(data, num_pc = 0):
    """Perfoms Principal component analysis using singular value decomposition (SVD) on data, and reduces the dimension to num_pc

    Args:
        data (np.array): matrix of data Nxn
        num_pc (int): number of principal components we reduce to, if 0 we don't reduce at all

    Returns:
        [u,s,vt,sigma]: SVD decomposition along with variance matrix
    """
    #If num_pc==0 we consider the whole dimension 
    if num_pc == 0:
        num_pc = data.shape[1]

    # centering the matrix
    x_mean = data - np.mean(data,axis=0)
    # performing single value decomposition
    U, S, V_t = svd(x_mean)
    # create m x n Sigma matrix
    sigma = np.zeros((x_mean.shape[0], x_mean.shape[1]))
    # populate Sigma with m x n diagonal matrix ; x_mean.shape[1]
    sigma[:min(x_mean.shape), :min(x_mean.shape)] = np.diag(S)
    #Reducing the number of components 
    sigma[num_pc:] = 0
    return U, S, V_t, sigma