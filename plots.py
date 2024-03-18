import torch
import matplotlib.pyplot as plt
import numpy as np
#from src.load_model import load

from loss_functions import f_loss_obst


def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=False, min_dist=1, f=5,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    x = x.cpu()
    xbar = xbar.cpu()

    fig = plt.figure(f)
    if obst:
        yy, xx = np.meshgrid(np.linspace(-6, 6, 120), np.linspace(-6, 6, 120))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0], device=device))
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    x = x.cpu()
    xbar = xbar.cpu()
    for i in range(n_agents):
        plt.plot(x[:T + 1, 4 * i].detach(), x[:T + 1, 4 * i + 1].detach(), color=colors[i % 12], linewidth=1)
        # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
        plt.plot(x[T:, 4 * i].detach(), x[T:, 4 * i + 1].detach(), color='k', linewidth=0.125, linestyle='dotted')
    for i in range(n_agents):
        plt.plot(x[0, 4 * i].detach(), x[0, 4 * i + 1].detach(), color=colors[i % 12], marker='o', fillstyle='none')
        plt.plot(xbar[4 * i].detach(), xbar[4 * i + 1].detach(), color=colors[i % 12], marker='*')
    ax = plt.gca()
    if dots:
        for i in range(n_agents):
            for j in range(T):
                plt.plot(x[j, 4 * i].detach(), x[j, 4 * i + 1].detach(), color=colors[i % 12], marker='o')
    if circles:
        for i in range(n_agents):
            r = min_dist / 2
            # if obst:
            #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
            # else:
            circle = plt.Circle((x[T, 4 * i].detach(), x[T, 4 * i + 1].detach()), r, color=colors[i % 12], alpha=0.5,
                                zorder=10)
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_trajectories.eps', format='eps')
    else:
        plt.show()
    return fig
def plot_traj_vs_time(t_end, n_agents, x, u=None, text="", save=False, filename=None):
    x = x.cpu()
    u = u.cpu()
    t = torch.linspace(0, t_end - 1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4 * p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:, 4 * i])
        plt.plot(t, x[:, 4 * i + 1])
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:, 4 * i + 2])
        plt.plot(t, x[:, 4 * i + 3])
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2 * i])
            plt.plot(t, u[:, 2 * i + 1])
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()


def plot_losses(epochs, lossl, lossxl, lossul, losscal, lossobstl, text="", save=False, filename=None):
    t = torch.linspace(0, epochs - 1, epochs)
    plt.figure(figsize=(4 * 2, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, lossl[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$loss$')
    plt.subplot(1, 2, 2)
    plt.plot(t, lossxl[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossx$')

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 3, 1)
    plt.plot(t, lossul[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossu$')
    plt.subplot(1, 3, 2)
    plt.plot(t, losscal[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossoa$')
    plt.suptitle(text)
    plt.subplot(1, 3, 3)
    plt.plot(t, lossobstl[:])
    plt.suptitle(text)
    plt.xlabel(r'$t$')
    plt.title(r'$lossobst$')

    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()
# def plot_GIF(gif=True, t_plot=100, std=0):
#     # t_plot = 14, 22, 100
#     out = load('distributedREN', False)
#     sys, ctl, x0, t_end, min_dist = out
#     # Extended time
#     t_ext = t_end * 4
#     x_log = torch.zeros(t_ext, sys.n)
#     u_log = torch.zeros(t_ext, sys.m)
#     w_in = torch.zeros(t_ext + 1, sys.n)
#     w_in[0, :] = (x0.detach() - sys.xbar)
#     u = torch.zeros(sys.m)
#     x = sys.xbar
#     xi = torch.zeros(ctl.psi_u.n_xi)
#     omega = (x, u)
#     for t in range(t_ext):
#         x, _ = sys(t, x, u, w_in[t, :])
#         u, xi, omega = ctl(t, x, xi, omega)
#         x_log[t, :] = x.detach()
#         u_log[t, :] = u.detach()
#     for t_plot in range(1,t_end):
#         fig = plot_trajectories(x_log, sys.xbar, sys.n_agents, save=True, T=t_plot, obst=True, circles=True,
#                                 axis=True, min_dist=min_dist, f=7)
#         plt.axis('equal')
#         ax = fig.gca()
#         plt.axis('equal')
#         fig.set_size_inches(6, 6.5)
#         ax.set(xlim=(-3, 3), ylim=(-3, 3.5))
#         plt.tight_layout()
#         plt.savefig("gif/distREN_%03i" % t_plot + ".png")
#         plt.close(fig)
#     # # 2nd trajectory
#     x_log2 = torch.zeros(t_ext, sys.n)
#     w_in = torch.zeros(t_ext + 1, sys.n)
#     w_in[0, :] = (x0.detach() - sys.xbar) + std * torch.randn(x0.shape)
#     u = torch.zeros(sys.m)
#     x = sys.xbar
#     xi = torch.zeros(ctl.psi_u.n_xi)
#     omega = (x, u)
#     for t in range(t_ext):
#         x, _ = sys(t, x, u, w_in[t, :])
#         u, xi, omega = ctl(t, x, xi, omega)
#         x_log2[t, :] = x.detach()
#     for t_plot in range(1,t_end):
#         fig = plot_trajectories(x_log, sys.xbar, sys.n_agents, save=True, T=1, obst=True, circles=False,
#                                 axis=True, min_dist=min_dist, f=7)
#         fig = plot_trajectories(x_log2, sys.xbar, sys.n_agents, save=True, T=t_plot, obst=False, circles=True,
#                                 axis=True, min_dist=min_dist, f=7)
#         plt.axis('equal')
#         ax = fig.gca()
#         plt.axis('equal')
#         fig.set_size_inches(6, 6.5)
#         ax.set(xlim=(-3, 3), ylim=(-3, 3.5))
#         plt.tight_layout()
#         plt.savefig("gif/distREN_%03i" % (t_end-1+t_plot) + ".png")
#         plt.close(fig)
#     # 3rd trajectory
#     x_log3 = torch.zeros(t_ext, sys.n)
#     w_in = torch.zeros(t_ext + 1, sys.n)
#     w_in[0, :] = (x0.detach() - sys.xbar) + std * torch.randn(x0.shape)
#     u = torch.zeros(sys.m)
#     x = sys.xbar
#     xi = torch.zeros(ctl.psi_u.n_xi)
#     omega = (x, u)
#     for t in range(t_ext):
#         x, _ = sys(t, x, u, w_in[t, :])
#         u, xi, omega = ctl(t, x, xi, omega)
#         x_log3[t, :] = x.detach()
#     for t_plot in range(1, t_end):
#         fig = plot_trajectories(x_log, sys.xbar, sys.n_agents, save=True, T=1, obst=True, circles=False,
#                                 axis=True, min_dist=min_dist, f=7)
#         fig = plot_trajectories(x_log2, sys.xbar, sys.n_agents, save=True, T=1, obst=False, circles=False,
#                                 axis=True, min_dist=min_dist, f=7)
#         fig = plot_trajectories(x_log3, sys.xbar, sys.n_agents, save=True, T=t_plot, obst=False, circles=True,
#                                 axis=True, min_dist=min_dist, f=7)
#         plt.axis('equal')
#         ax = fig.gca()
#         plt.axis('equal')
#         fig.set_size_inches(6, 6.5)
#         ax.set(xlim=(-3, 3), ylim=(-3, 3.5))
#         plt.tight_layout()
#         plt.savefig("gif/distREN_%03i" % (2*t_end-2 + t_plot) + ".png")
#         plt.close(fig)
#     #os.system("convert -delay 4 -loop 0 gif/corridor_*.png gif/corridor.gif")
#     #os.system("rm gif/corridor_*.png")