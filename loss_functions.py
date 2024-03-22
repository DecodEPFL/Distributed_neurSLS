import torch
import numpy as np

def f_loss_states(t, x, sys, Q=None, Qp=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    gamma = 1
    if Q is None:
        Q = torch.eye(sys.n,device = device)
    if Qp is None:
        qp = torch.eye(2,device = device)
        qv = torch.zeros(2, 2,device = device)
        Qp = torch.block_diag(qp, qv, qp, qv, qp, qv, qp, qv)

    if t < 70:
        xbar = torch.tensor([0.75, 2, 0, 0,
                             0.75, -2, 0, 0,
                             -0.75, -2, 0, 0,
                             -0.75, 2, 0, 0,
                             ],device = device)
        dx = x - xbar
        pQp = 0#torch.matmul(torch.matmul(dx, Qp), dx)
        xQx = 100*torch.matmul(torch.matmul(dx, Q), dx)
    else:
        pQp = 0
        dx = x - sys.xbar
        xQx = torch.matmul(torch.matmul(dx, Q), dx)
        #xQx = 0

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
        #mu1 = torch.tensor([[-4, 0.0]],device=device)
        #mu2 = torch.tensor([[4, 0.0]],device=device)
        #cov = torch.tensor([[3, 0.2]],device=device)
        mu1 = torch.tensor([[-4, 0.0]], device=device)
        mu2 = torch.tensor([[4, 0.0]], device=device)
        mu3 = torch.tensor([[-4.5, 0.0]], device=device)
        mu4 = torch.tensor([[4.5, 0.0]], device=device)
        mu5 = torch.tensor([[-3.5, 0.0]], device=device)
        mu6 = torch.tensor([[3.5, 0.0]], device=device)
        mu7 = torch.tensor([[-3, 0.0]], device=device)
        mu8 = torch.tensor([[3, 0.0]], device=device)
        mu9 = torch.tensor([[-2.5, 0.0]], device=device)
        mu10 = torch.tensor([[2.5, 0.0]], device=device)
        mu11 = torch.tensor([[-5, 0.0]], device=device)
        mu12 = torch.tensor([[5, 0.0]], device=device)
        mu13 = torch.tensor([[-5.5, 0.0]], device=device)
        mu14 = torch.tensor([[5.5, 0.0]], device=device)
        mu15 = torch.tensor([[-6, 0.0]], device=device)
        mu16 = torch.tensor([[6, 0.0]], device=device)
        cov = torch.tensor([[0.25, 0.25]], device=device)


        Q1  = normpdf(q, mu=mu1, cov=cov)
        Q2 = normpdf(q, mu=mu2, cov=cov)
        Q3 = normpdf(q, mu=mu3, cov=cov)
        Q4 = normpdf(q, mu=mu4, cov=cov)
        Q5 = normpdf(q, mu=mu5, cov=cov)
        Q6 = normpdf(q, mu=mu6, cov=cov)
        Q7 = normpdf(q, mu=mu7, cov=cov)
        Q8 = normpdf(q, mu=mu8, cov=cov)
        Q9 = normpdf(q, mu=mu9, cov=cov)
        Q10 = normpdf(q, mu=mu10, cov=cov)
        Q11 = normpdf(q, mu=mu11, cov=cov)
        Q12 = normpdf(q, mu=mu12, cov=cov)
        Q13 = normpdf(q, mu=mu13, cov=cov)
        Q14 = normpdf(q, mu=mu14, cov=cov)
        Q15 = normpdf(q, mu=mu15, cov=cov)
        Q16 = normpdf(q, mu=mu16, cov=cov)

        QQ = (Q1 + Q2 + Q3 + Q4 + Q5 + Q6 + Q7 + Q8 + Q9 + Q10 + Q11 + Q12 + Q13 + Q14 + Q15 + Q16).sum()
        mask = QQ > 0.007
        return QQ*mask


def f_loss_side(x):
    qx = x[::4]
    qy = x[1::4]
    side = torch.relu(qx - 5) + torch.relu(-5 - qx) + torch.relu(qy - 20) + torch.relu(-20 - qy)
    return side.sum()


def f_loss_formation(x,sys,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    Mx = torch.zeros(sys.n_agents, sys.n, device=device)
    Mx[0, 0] = 1
    Mx[1, 4] = 1
    Mx[2, 8] = 1
    Mx[3, 12] = 1
    px = torch.matmul(Mx, x)
    My = torch.zeros(sys.n_agents, sys.n, device=device)
    My[0, 1] = 1
    My[1, 5] = 1
    My[2, 9] = 1
    My[3, 13] = 1
    py = torch.matmul(My, x)
    deltax = px.repeat(1, 1).T - px.repeat(1, 1)
    deltay = py.repeat(1, 1).T - py.repeat(1, 1)
    maxlength5 = sys.maxlength5
    maxlength6 = sys.maxlength6
    maxlength7 = sys.maxlength7
    maxlength8 = sys.maxlength8
    maxlength9 = sys.maxlength9
    maxlength10 = sys.maxlength10
    Fk5 = torch.abs(torch.sqrt(deltax[0, 1] ** 2 + deltay[0, 1] ** 2) - maxlength5)
    Fk6 = torch.abs(torch.sqrt(deltax[1, 2] ** 2 + deltay[1, 2] ** 2) - maxlength6)
    Fk7 = torch.abs(torch.sqrt(deltax[2, 3] ** 2 + deltay[2, 3] ** 2) - maxlength7)
    Fk8 = torch.abs(torch.sqrt(deltax[3, 0] ** 2 + deltay[3, 0] ** 2) - maxlength8)
    Fk9 = torch.abs(torch.sqrt(deltax[2, 0] ** 2 + deltay[2, 0] ** 2) - maxlength9)
    Fk10 = torch.abs(torch.sqrt(deltax[3, 1] ** 2 + deltay[3, 1] ** 2) - maxlength10)

    Fsprings = Fk5+Fk6+Fk7+Fk8+Fk9+Fk10
    return Fsprings