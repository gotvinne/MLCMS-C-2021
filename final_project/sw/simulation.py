""" Module simulating data set or the performance of a integrator template neural network
"""
from scipy.integrate import solve_ivp
from torch import from_numpy, tensor
import torch
import numpy as np
from random import randrange

from sw.dataformating import get_sim_data
from sw.numerical_integrator import erk
from sw.ERKNN import ERKNN

# Constants:
NUM_PEDESTRAINS = 100


def gravityCenterERK(model: ERKNN, sim_steps: int, dt: float, data_filepath: str, num: int = NUM_PEDESTRAINS) -> np.ndarray:
    """
    Calculating center of gravity for num pedestrians using ERK-model
    :param model: ERKNN model
    :param sim_steps: Simulation steps
    :param dt: time step
    :param data_filepath: filepath to data set
    :param num: number of pedestrains to be evaluated
    :return: the np.ndarray containing the trajectory of the estimated center of gravity
    """
    traj = np.zeros([2, sim_steps])

    for i in range(1, num + 1):
        traj = np.add(traj, simulateERK(model, i, sim_steps, dt, data_filepath))
    return (1 / num) * traj


def simulateERK(model: ERKNN, pedestrainId: int, sim_steps: int, dt: float, data_filepath: str) -> np.ndarray:
    """
    Simulate one trajectory of the ERKNN class, the simulation is done via the RK method given by model
    :param model: ERKNN model
    :param pedestrainId: The pedestrain ID
    :param sim_steps: simulation steps
    :param dt: time step
    :param data_filepath: filepath to data set
    :return: np.ndarray containing the simulated trajectory of a pedestrian using ERKNN
    """
    sim_data, _ = get_sim_data(data_filepath, pedestrainId)
    traj = torch.empty((2, sim_steps))
    traj[:, 0] = from_numpy(sim_data[0, :])
    dt = tensor([dt])

    for i in range(1, sim_steps):
        k = model.forward(traj[:, i - 1], dt)
        traj[:, i] = erk(traj[:, i - 1], dt, k, model.integrator_dict["b"])
    return traj.detach().numpy()


def simulateRandomERK(model: ERKNN, num: int, sim_steps: int, dt: float, data_filepath: str) -> np.ndarray:
    """
    Simulate num random pedestrian trajectories using the model
    :param model: ERKNN
    :param num: number of trajectories
    :param sim_steps: Number of trajectory data points
    :param time_step: The time step between each data point
    :param data_filepath: Filepath to the data set
    :return: Num trajectories
    """
    res = [randrange(1, 100) for _ in range(num)]
    traj = np.zeros([num * 2, sim_steps])

    for i, ped in enumerate(res):
        traj[2 * i:2 * i + 2, :] = simulateERK(model, ped, sim_steps, dt, data_filepath)
    return traj


def simulatingNN(t: np.array, x: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """
    Function representing the ODE, with the form demanded by scipy.integrate.solve_ivp()
    :param model: NN trained to approximate ODE
    :param t: Parameter from solve_ivp()
    :param x: State
    :return: The derivatives in two dimensions given state
    """
    ## the neural network need tensor input, solve_ivp is implemented using numpy
    x = from_numpy(x)
    return model(x.float()).detach()


def gravityCenterEuler(model: torch.nn.Module, time: float, data_filepath: str, num: int = NUM_PEDESTRAINS) -> np.ndarray:
    """
    Calculating the center of gravity of for num pedestrains using IntegratorNN model
    :param model: NN trained to approximate ODE
    :param time: time step
    :param data_filepath: Filepath to the data set
    :param num: number of pedestrains to be evaluated
    :return: the np.ndarray containing the trajectory of the estimated center of gravity
    """
    traj = np.zeros([2, time])

    for i in range(1, num + 1):
        traj = np.add(traj, simulateEuler(model,i, time,data_filepath))

    return (1 / num) * traj


def simulateRandomEuler(num: int, time: float, model: torch.nn.Module, data_filepath: str) -> np.ndarray:
    """
    Simulate num random pedestrian trajectories using the model
    :param num: Number of pedestrians trajectories to be simulated
    :param time: time horizon to simulate
    :param model: NN trained to approximate ODE
    :param data_filepath: Filepath to the data set
    :return: Num trajectories
    """
    res = [randrange(1, 100) for _ in range(num)]
    traj = np.zeros([num * 2, time])

    for i, ped in enumerate(res):
        traj[2 * i:2 * i + 2, :] = simulateEuler(model,ped,time,data_filepath)
    return traj


def simulateEuler(model: torch.nn.Module,pedestrainID: int, time: float, data_filepath: str) -> np.ndarray:
    """
    Simulate num random pedestrian trajectories using NN, the simulation is done using scipy.integrate.solve_ivp(), yielding a RK45 method.
    :param num: number of pedestrians to be simulated
    :param t: end time interval
    :param model: NN trained to approximate ODE
    :param data_filepath: Filepath to data set
    :return: num simulated trajectories
    """
    traj = np.zeros([2, time])
    sim_data, _ = get_sim_data(data_filepath, pedestrainID)
    x0 = sim_data[0, :]

    sol = solve_ivp(simulatingNN, [0, time], x0, args=[model], t_eval=np.arange(0, time))
    traj[0, :] = sol.y[0]
    traj[1, :] = sol.y[1]
    return traj


def bif_simulatingNN(t: np.array, x: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """
    Function representing the ODE, with the form demanded by scipy.integrate.solve_ivp() (with bifurcation parameter)
    :param t: Parameter from solve_ivp()
    :param x: State
    :param model: NN trained to approximate ODE
    :param alpha: Hopf constant
    :return: The derivatives in two dimensions given state, and alpha 
    """
    ## the neural network need tensor input, solve_ivp is implemented using numpy
    x = from_numpy(x)
    alpha = x[2]
    y = model(x.float()).detach()
    n = np.zeros(shape=(1), dtype=np.float32) 
    n[0] = alpha
    alpha_tensor= torch.tensor(n)
    x = torch.cat((y,alpha_tensor), 0)
    
    return x


def simulate_bif_trajectories(num: int, t: float, model: torch.nn.Module, data_filepath: str,
                              alpha: float) -> np.ndarray:
    """
    Simulate num random pedestrian trajectories using NN (with bifurcation parameter)
    :param num: number of pedestrians to be simulated
    :param t: end time interval
    :param model: NN trained to approximate ODE
    :param data_filepath: Filepath to data set
    :param alpha: bifurcation parameter
    :return: num simulated trajectories
    """
    res = [randrange(1, 100) for _ in range(num)]
    traj = np.zeros([num * 2, t])

    for i, ped in enumerate(res):
        sim_data, _ = get_sim_data(data_filepath, ped)
        x0 = np.array([sim_data[0, 0], sim_data[0,1], alpha], dtype=float)
        
        sol = solve_ivp(bif_simulatingNN, [0, t], x0, args=[model], t_eval=np.arange(0, t))
        traj[2 * i, :] = sol.y[0]
        traj[2 * i + 1, :] = sol.y[1]
    return traj
