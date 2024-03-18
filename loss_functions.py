import torch
import numpy as np

def f_loss_states(t, x, sys, Q=None, Qp=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    gamma = 1
    if Q is None:
        Q = 0.1*torch.eye(sys.n,device = device)
    if Qp is None:
        qp = 100 * torch.eye(2,device = device)
        qv = torch.zeros(2, 2,device = device)
        Qp = torch.block_diag(qp, qv, qp, qv, qp, qv, qp, qv)

    if t < 30:
        xbar = torch.tensor([0.75, 2, 0, 0,
                             0.75, -2, 0, 0,
                             -0.75, -2, 0, 0,
                             -0.75, 2, 0, 0,
                             ],device = device)
        dx = x - xbar
        pQp = torch.matmul(torch.matmul(dx, Qp), dx)
        xQx = 0
    else:
        pQp = 0
        dx = x - sys.xbar
        xQx = torch.matmul(torch.matmul(dx, Q), dx)

    return xQx+pQp  # * (gamma**(100-t))


def f_loss_u(t, u):
    loss_u = (u ** 2).sum()
    return loss_u


def f_loss_ca(x, sys, min_dist=0.5,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    min_sec_dist = min_dist + 0.2
    # collision avoidance:
    deltaqx = x[0::4].repeat(sys.n_agents, 1) - x[0::4].repeat(sys.n_agents, 1).transpose(0, 1)
    deltaqy = x[1::4].repeat(sys.n_agents, 1) - x[1::4].repeat(sys.n_agents, 1).transpose(0, 1)
    distance_sq = deltaqx ** 2 + deltaqy ** 2
    mask = torch.logical_not(torch.eye(sys.n // 4,device=device))
    loss_ca = (1 / (distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * mask).sum() / 2
    return loss_ca


def normpdf(q, mu, cov,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    d = torch.tensor(2,device=device)
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2)
    out = torch.tensor(0,device = device)
    for qi in qs:
        den = (2 * torch.tensor(np.pi,device=device)) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu) ** 2 / cov).sum())
        out = out + num / den
    return out


def f_loss_obst(x, sys=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if x.size(0) == 4:
        qx = x[::4].unsqueeze(1)
        qy = x[1::4].unsqueeze(1)
        q = torch.cat((qx, qy), dim=1).view(1, -1).squeeze()
    else:
        Mx = torch.zeros(8, 16,device = device)
        Mx[0, 0] = 1
        Mx[1, 1] = 1
        Mx[2, 4] = 1
        Mx[3, 5] = 1
        Mx[4, 8] = 1
        Mx[5, 9] = 1
        Mx[6, 12] = 1
        Mx[7, 13] = 1
        q = torch.matmul(Mx, x)
    mu1 = torch.tensor([[-4, 0.0]],device=device)
    mu2 = torch.tensor([[4, 0.0]],device=device)
    cov = torch.tensor([[3, 0.2]],device=device)

    Q1  = normpdf(q, mu=mu1, cov=cov)
    Q2 = normpdf(q, mu=mu2, cov=cov)

    return (Q1 + Q2).sum()


def f_loss_side(x):
    qx = x[::4]
    qy = x[1::4]
    side = torch.relu(qx - 5) + torch.relu(-5 - qx) + torch.relu(qy - 20) + torch.relu(-20 - qy)
    return side.sum()
