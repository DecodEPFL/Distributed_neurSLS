import torch
import numpy as np


def calculate_collisions(x, sys, min_dist):
    deltax = x[:, 0::4].repeat(sys.n_agents, 1, 1) - x[:, 0::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    deltay = x[:, 1::4].repeat(sys.n_agents, 1, 1) - x[:, 1::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    distance_sq = (deltax ** 2 + deltay ** 2)
    n_coll = ((0.0001 < distance_sq) * (distance_sq < min_dist ** 2)).sum()
    return n_coll


def set_params(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # # # # # # # # Parameters # # # # # # # #
    min_dist = .75  # min distance for collision avoidance
    t_end = 150
    Ts = 0.05
    n_agents = 4
    x0, xbar = get_ini_cond()
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-2
    epochs = 500
    Q = torch.kron(torch.eye(n_agents,device = device), torch.diag(torch.tensor([1, 1, 1, 1.],device = device)))
    alpha_u = 0.001#0.1  # Regularization parameter for penalizing the input
    alpha_ca = 1e3
    alpha_obst = 5e3
    n_xi = np.array([30, 30, 30, 30])  # \xi dimension -- number of states of REN
    l = np.array([30, 30, 30, 30])  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 10  # number of trajectories collected at each step of the learning
    std_ini = 0.4#.4  # standard deviation of initial conditions
    return min_dist, t_end, n_agents, x0, xbar, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
    l, n_traj, std_ini, Ts


def get_ini_cond(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Corridor problem
    x0 = torch.tensor([-2, -3.5, 0, 0,
                       2, -3.5, 0, 0,
                       2, -5, 0, 0,
                       -2, -5, 0, 0,
                       ],device = device)
    xbar = torch.tensor([-2, 5, 0, 0,
                         2, 5, 0, 0,
                         2, 3.5, 0, 0,
                         -2, 3.5, 0, 0,
                         ],device = device)

    # xbar = torch.tensor([2, 3.5, 0, 0,
    #                       -2, 3.5, 0, 0,
    #                       -2, 5, 0, 0,
    #                       2, 5, 0, 0,
    #                       ], device=device)

    return x0, xbar