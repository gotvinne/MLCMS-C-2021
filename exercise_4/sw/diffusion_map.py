"""Implementing the diffusion map algorithm for data dimentionality reduction
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import inv

def diag_normalization(np_mat):
    """Returning the diagonal normalized matrix

    Args:
        np_mat (np.array): Input matrix

    Returns:
        np.array: Diagonal matrix consisting of the sum of every row for the input matrix
    """
    summation = sum(np_mat)
    M = np.diag(summation)
    return M

def half_inv_diag_matrix(diag_mat):
    """Calulated the half inverse diagonal matrix
    The half inverse for a diagonal matrix: hinv = diag(1/sqrt(lambda_ii))

    Args:
        diag_mat (np.array): Diagonal numpy matrix

    Returns:
        np.array: Half inverse diagonal numpy matrix
    """
    vec = np.diag(diag_mat)
    vec = 1/np.sqrt(vec)
    half_inverse = np.diag(vec)
    return half_inverse

def eigen_decomposition(l,T):
    """Calculates eigenvalues and eigenvectors, and returns the l largest eigenvalues and corresponding eigenvectors

    Args:
        l (int): Number of largest eigenvalues and corresponding eigenvectors
        T (np.array): Numpy symmetric matrix 

    Returns:
        np.array: A numpy array with the l eigenvalues?????
        np.array: A numpy matrix containing the l'th eigenvector for the l'th column
    """
    N = T.shape[0]
    eig_val,eig_vec = np.linalg.eig(T)

    # Finding indexes
    idx = np.argpartition(eig_val, -l)[-l:]
    indices = idx[np.argsort((-eig_val)[idx])]

    # LL eigenvalue array
    ll = np.array([eig_val[i] for i in indices])

    # Vl eigenvector matrix
    vl = np.empty(0)
    vl = np.append(vl, [eig_vec[:,i] for i in indices])
    vl = vl.reshape(l,N).transpose()
    return np.real(ll),np.real(vl)

def eigen_functions(Q_hinv,vl):
    """Returns a np.matrix of eigenfunctions where each column represent a eigenfunction

    Args:
        Q_hinv (np.array): Numpy half inverted matrix
        vl (np.array): Matrix containing the l eigenvectors

    Returns:
        np.array: A matrix where each l'th column represents the l'th eigenfunction 
    """
    N = Q_hinv.shape[0]
    l = vl.shape[1]
    eig_func = np.empty(0)
    eig_func = np.append(eig_func,np.array([Q_hinv@vl[:,i] for i in range(l)]))
    eig_func = eig_func.reshape(l,N).transpose()
    eig_func = np.delete(eig_func,0,axis=1)
    return eig_func

def diffusion_map(eta,l,data):
    """One variant of the diffusion map algorithm, returns the non-constant eigenvalues and eigenfunctions making the Laplace-Beltrami operator

    Args:
        eta (float): Diameter of the dataset coeffisient
        l (int): Number of eigenfunctions
        data (np.ndarray): Data Matrix

    Returns:
        np.array: ll the roots of the diffusion map
        np.ndarray: eig_func the eigenfuntions of the diffusion map
    """
    N = data.shape[0]
    # Similarity matrix
    dist = distance_matrix(data,data)
    diameter = np.amax(dist)
    epsilon = eta*diameter

    # Normalization and diagonalization
    W = np.exp(-np.square(dist)/epsilon) # Kernal matrix
    P = diag_normalization(W)
    P_inv = inv(P)
    K = P_inv@W@P_inv
    Q = diag_normalization(K)
    Q_half_inv = half_inv_diag_matrix(Q)
    T = Q_half_inv@K@Q_half_inv

    # Eigen decomposition and functions
    ll, vl = eigen_decomposition(l+1,T)
    ll = np.power(ll,1/eta)

    # The 0'th eigenvalue and eigenfunction are irrelevant:
    ll = np.delete(ll,0)
    eig_func = eigen_functions(Q_half_inv,vl)
    return ll,eig_func