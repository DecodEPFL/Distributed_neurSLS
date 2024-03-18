#!/usr/bin/env python
"""
Train a system of distributed REN controller for the system of 4 interconnected robots in a corridor
Author: Danilo Saccani (danilo.saccani@epfl.ch)

"""
import numpy as np
import torch
import scipy

from models import Controller, SystemsOfSprings, Input
from plots import plot_trajectories, plot_traj_vs_time, plot_losses
from loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_side
from utils import calculate_collisions, set_params


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
np.random.seed(1234)

ny = 2  # output dimension of single ren = input dimension of single ren from other ren
nx = 4  # number of states single agent = dimension of single exogenous input
n_agents = 4
torch.manual_seed(1)
print_simulation = True

params = set_params()
min_dist, t_end, n_agents, x0, xbar, learning_rate, epochs, Q, \
    alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini, Ts = params

# Definition of interconnection matrix M
M1 = torch.hstack((torch.zeros((4, 8),device=device), torch.eye(4,device=device), torch.zeros((4, 12),device=device)))
M2 = torch.hstack((torch.zeros((2, 2),device=device), torch.eye(2,device=device), torch.zeros((2, 20),device=device)))
M3 = torch.hstack((torch.zeros((2, 6),device=device), torch.eye(2,device=device), torch.zeros((2, 16),device=device)))
M4 = torch.hstack((torch.zeros((4, 12),device=device), torch.eye(4,device=device), torch.zeros((4, 8),device=device)))
M5 = torch.hstack((torch.zeros((2, 4),device=device), torch.eye(2,device=device), torch.zeros((2, 18),device=device)))
M6 = torch.hstack((torch.eye(2,device=device), torch.zeros((2, 22),device=device)))
M7 = torch.hstack((torch.zeros((4, 16),device=device), torch.eye(4,device=device), torch.zeros((4, 4),device=device)))
M8 = torch.hstack((torch.zeros((2, 2),device=device), torch.eye(2,device=device), torch.zeros((2, 20),device=device)))
M9 = torch.hstack((torch.zeros((2, 6),device=device), torch.eye(2,device=device), torch.zeros((2, 16),device=device)))
M10 = torch.hstack((torch.zeros((4, 20),device=device), torch.eye(4,device=device)))
M11 = torch.hstack((torch.eye(2,device=device), torch.zeros((2, 22),device=device)))
M12 = torch.hstack((torch.zeros((2, 4),device=device), torch.eye(2,device=device), torch.zeros((2, 18),device=device)))

M = torch.vstack((M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12))

Muy = M[:, 0:8]
Mud = M[:, 8:24]

sys_model = 'NeurSLS-distributedREN'

sys = SystemsOfSprings(xbar,Ts)

# # # # # # # # Define models # # # # # # # #

ctl = Input(8, t_end)

ctl = Controller(sys.f, n_agents, Muy, Mud, np.array([ny + nx, ny + nx, ny + nx, ny + nx]), np.array([ny, ny, ny, ny]),
                 n_xi, l, True, t_end)
# # # # # # # # Define optimizer and parameters # # # # # # # #
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)

# # # # # # # # Training # # # # # # # #
print("------- Print open loop trajectories --------")

x_log = torch.zeros((t_end, sys.n),device=device)
u_log = torch.zeros((t_end, sys.m),device=device)
w_in = torch.zeros((t_end + 1, sys.n),device=device)
if print_simulation:
    u = torch.zeros(sys.m,device=device)
    x = x0
    for t in range(t_end):
        x = sys(t, x, u, w_in[t, :])
        x_log[t, :] = x.detach()
        u_log[t, :] = u.detach()
    plot_trajectories(x_log, xbar, sys.n_agents, text="CL - before training", T=t_end, obst=alpha_obst)

# # # # # # # # Training # # # # # # # #
print("------------ Begin training ------------")
print("Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate +
      " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f" % std_ini)
print(" -- alpha_u: %.1f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e" % alpha_obst)
print("--------- --------- ---------  ---------")



lossl = np.zeros(epochs)
lossxl = np.zeros(epochs)
lossul = np.zeros(epochs)
losscal = np.zeros(epochs)
lossobstl = np.zeros(epochs)
x = x0
xi = torch.zeros((sum(n_xi)),device=device)
for epoch in range(epochs):
    gamma = []
    optimizer.zero_grad()
    loss_x, loss_u, loss_ca, loss_obst, loss_side = 0, 0, 0, 0, 0
    #if epoch > 100:
    #    optimizer.param_groups[0]['lr'] = 1e-2
    for kk in range(n_traj):
        x.detach()
        x = x0 + std_ini * torch.randn(x0.shape, device=device)
        u = torch.zeros(sys.m,device=device)
        omega = (x, u)
        us = torch.zeros(sys.m,device=device)
        xi = torch.zeros((sum(n_xi)),device=device)
        for t in range(t_end):
            x = sys(t, x, us, w_in[t, :])
            omega = (x, us)
            u, xi, gamma, us = ctl(t, u, x, xi, omega)
            #if epoch>1:
            #    print("max input: %.4f --- " % (torch.max(u)))
            #    print("max input amplified: %.4f --- " % (torch.max(us)))
            loss_x = loss_x + f_loss_states(t, x, sys, Q)
            loss_u = loss_u + alpha_u * f_loss_u(t, us)
            loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist)
            loss_side = loss_side + alpha_obst * f_loss_side(x)
            if alpha_obst != 0:
                loss_obst = loss_obst + alpha_obst * f_loss_obst(x)
    loss = loss_x + loss_ca + loss_obst + loss_side + loss_u
    print("Epoch: %i --- Loss: %.4f ---||--- Loss x: %.2f --- " % (epoch, loss / t_end, loss_x) +
          "Loss u: %.2f --- Loss ca: %.2f --- Loss obst: %.2f" % (loss_u, loss_ca, loss_obst) +
          "Loss side: %.2f " % loss_side)
    print("Max u: %.4f --- " % (torch.norm(ctl.sp.u[:, 1])))
    print("Amplifier: %.4f --- " % (ctl.amplifier))
    loss.backward()
    optimizer.step()
    lossl[epoch] = loss.detach()
    lossxl[epoch] = loss_x.detach()
    lossul[epoch] = loss_u.detach()
    losscal[epoch] = loss.detach()
    lossobstl[epoch] = loss_obst.detach()

# # # # # # # # Save trained model # # # # # # # #
torch.save(ctl.netREN.state_dict(), "trained_models/" + sys_model + "_tmp.pt")
# # # # # # # # Print & plot results # # # # # # # #

# SIMULATION AND PLOTS

x_log = torch.zeros((t_end, sys.n),device=device)
u_log = torch.zeros((t_end, sys.m),device=device)
w_in = torch.zeros((t_end + 1, sys.n),device=device)

u = torch.zeros(sys.m,device=device)
us = torch.zeros(sys.m,device=device)
x = x0
xi = torch.zeros((sum(n_xi)),device=device)
omega = (x, u)
for t in range(t_end):
    x = sys(t, x, us, w_in[t, :])
    omega = (x, us)
    u, xi, gamma, us = ctl(t, u, x, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = us.detach()
plot_traj_vs_time(t_end, sys.n_agents, x_log, u_log)
# Number of collisions
n_coll = calculate_collisions(x_log, sys, min_dist)
print("Number of collisions after training: %d" % n_coll)
plot_losses(epochs, lossl, lossxl, lossul, losscal,lossobstl)
xbarExp = sys.xbar.cpu()
x_log = x_log.cpu()
scipy.io.savemat('dataTrainedSystem_REN.mat', dict( x_log=x_log.detach().numpy(), xbar=xbarExp.detach().numpy()))

# Extended time
u = torch.zeros(sys.m,device=device)
us = torch.zeros(sys.m,device=device)
x = x0
xi = torch.zeros((sum(n_xi)),device=device)
omega = (x, u)
t_ext = t_end * 2
x_log = torch.zeros(t_ext, sys.n,device=device)
u_log = torch.zeros(t_ext, sys.m,device=device)
w_in = torch.zeros((t_ext + 1, sys.n),device=device)
for t in range(t_ext):
    x = sys(t, x, us, w_in[t, :])
    omega = (x, us)
    u, xi, gamma, us = ctl(t, u, x, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = us.detach()
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)
#plot_GIF()
xbarExp = sys.xbar.cpu()
x_log = x_log.cpu()
scipy.io.savemat('dataTrainedSystem_REN_ext.mat', dict( x_log=x_log.detach().numpy(), xbar=xbarExp.detach().numpy(),epochs=epochs))



