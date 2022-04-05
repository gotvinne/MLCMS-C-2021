"""Library implementing the Principal component analysis as a dimentionality reduction method
"""

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

def energy_pca(data, num_pc=0):
    """Computes the energy (explained variance) on each principal component

    Args:
        data (np.array): matrix of data Nxn
        num_pc (int, optional): Number of principal components we reduce to, if 0 we don't reduce

    Returns:
        np.array: array of energy equivalent to PC-i  
    """
    # equivalent to L and defines, how much principal components are taken in consideration
    if num_pc== 0:
        num_pc = data.shape[1]
    # performing pca on data matrix
    _, S, _, sigma = pca(data, num_pc)
    # calculating trace of squared matrix s
    trace = np.sum(np.square(S))
    # calculating energy matrix
    energy = np.square(sigma) / trace
    return energy.diagonal() 

def draw_pc(data, vt, s):
    """Plots the data points along with the principal components

    Args:
        data (np.array): data matrix Nxn
        vt (np.array): eigenvectors of principal componen
        s (np.array): singular values of principal components
    """
    x_mean, f_x_mean = np.mean(data,axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(data[:,0], data[:,1], color="c")
    # plotting principal components
    for i, pc in enumerate(vt):
        ax.quiver(x_mean, f_x_mean, pc[0], pc[1], label=f"Principal Component {i+1}",
        color=f"C{i}",scale_units="xy", scale=1, )

    plt.title(f"PC-1 S={np.around(s[0], 2)} "
              f", PC-2 S=: {np.around(s[1],2)}")
    plt.legend(loc="best")
    plt.savefig("figures/task1/task1_1_pca.png", bbox_inches='tight')
    plt.show()

def draw_img_per_pc(img, pcs):
    """Plots the same images using different numbers of principal components

    Args:
        img (np.array): matrix of image we want to draw
        pcs (list): list of the number of principal components we want to test
    """
    for num_pc in pcs:
        u,s,vt,sigma = pca(img,num_pc)
        reconstruction = u @ sigma @ vt
        plt.imshow(reconstruction)
        plt.title(f'reconstructed image with number of pcs={num_pc}')
        plt.savefig(f"figures/task1/task1_2_pc={num_pc}.png", bbox_inches='tight')
        plt.show()
        energy = np.sum(energy_pca(img,num_pc))
        energy_loss = (1-energy)*100
        print("Sum of energy: ",energy) 
        print("energy loss: ", energy_loss, "%")

def find_one_percent_loss(img):
    """Finds the smallest number of prinicpal components needed to get under 1% energy loss
    and plots the image reconstruction 

    Args:
        img (np.array): image matrix 
    """
    #We know that it's less than 120, as the loss here is at 0.2%
    num_pc = 120 
    
    while num_pc>0:
        energy = np.sum(energy_pca(img,num_pc))
        if (1-energy)*100>1.0:
            num_pc+=1
            print("less than 1% energy loss at number of principal components: ", num_pc)
            break 
        num_pc-=1

    u,s,vt,sigma = pca(img,num_pc)
    reconstruction = u @ sigma @ vt
    plt.imshow(reconstruction)
    plt.title(f'reconstructed image with number of pcs={num_pc}')
    plt.savefig(f"task1_2_pc={num_pc}.png", bbox_inches='tight')
    plt.show()
    energy = np.sum(energy_pca(img,num_pc))
    energy_loss = (1-energy)*100
    print("Sum of energy: ",energy) 
    print("energy loss: ", energy_loss, "%")

def plot_energy_loss(data):
    max_number_pcs = data.shape[1]
    energy_pc = np.zeros((max_number_pcs,2))
    for i in range(0,max_number_pcs):
        energy = np.sum(energy_pca(data,i+1))
        energy_pc[i,0]= i+1
        energy_pc[i,1] = (1-energy)*100
    plt.plot(energy_pc[:,0], energy_pc[:,1])
    #plt.plot(energy_pc[:,0], 0*energy_pc[:,0]+1)
    plt.xlabel("Number of PCs")
    plt.ylabel("Energy loss in %")
    plt.savefig(f"figures/task1/task1_2_energy_loss_plot.png", bbox_inches='tight')
    plt.show()

def draw_ped_path(ped_data, pedestrians, pca=None):
    """Plots the path of a specific number of pedestrians given in a list

    Args:
        ped_data (np.array): time series of all pedestrians 
        pedestrians (list): which pedestrians to plot ex [1,2] plots pedestrian 1 and 2 
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for ped in pedestrians:
        ax.scatter(ped_data[:,2*(ped-1)],ped_data[:,2*(ped-1)+1], label=f"Pedestrian {ped}", color =f"C{ped-1}")
        ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Path of Pedestrians {*pedestrians,}")
    ax.legend()
    plt.savefig(f"figures/task1/task1_3_ped_path_pc={pca}.png", bbox_inches='tight')
    plt.show()