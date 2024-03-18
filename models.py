#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class RENRG(nn.Module):
    def __init__(self, n, m, n_xi, l,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n1
        self.l = l  # nel paper q
        self.m = m  # nel paper p

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l,device=device) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi,device=device) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n,device=device) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi,device=device) * std))
        self.D21 = nn.Parameter((torch.randn(m, l,device=device) * std))
        if m >= n:
            self.Z3 = nn.Parameter(torch.randn(m - n, n,device=device) * std)
            self.X3 = nn.Parameter(torch.randn(n, n,device=device) * std)
            self.Y3 = nn.Parameter(torch.randn(n, n,device=device) * std)
        else:
            self.Z3 = nn.Parameter(torch.randn(n - m, m,device=device) * std)
            self.X3 = nn.Parameter(torch.randn(m, m,device=device) * std)
            self.Y3 = nn.Parameter(torch.randn(m, m,device=device) * std)
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n,device=device) * std))
        # bias:
        self.bxi = nn.Parameter(torch.randn(n_xi,device=device))
        self.bv = nn.Parameter(torch.randn(l,device=device))
        self.bu = nn.Parameter(torch.randn(m,device=device))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi,device=device)
        self.B1 = torch.zeros(n_xi, l,device=device)
        self.E = torch.zeros(n_xi, n_xi,device=device)
        self.Lambda = torch.ones(l,device=device)
        self.C1 = torch.zeros(l, n_xi,device=device)
        self.D11 = torch.zeros(l, l,device=device)
        self.Lq = torch.zeros(m, m,device=device)
        self.Lr = torch.zeros(n, n,device=device)
        self.D22 = torch.zeros(m, n,device=device)

    def forward(self, t, w, xi, gammap,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # Parameters update-------------------------------------------------------
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        R = gammap * torch.eye(n, n,device = device)
        Q = (-1 / gammap) * torch.eye(m, m,device = device)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(min(n, m),device = device)
        if m >= n:
            N = torch.vstack((F.linear(torch.eye(min(n, m),device = device) - M,
                                       torch.inverse(torch.eye(min(n, m),device = device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(min(n, m),device = device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(min(n, m),device = device) + M),
                                       (torch.eye(min(n, m),device = device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(min(n, m),device = device) + M), self.Z3)))

        self.D22 = gammap * N
        R_capital = R - (1 / gammap) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m,device=device)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l,device=device) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Forward dynamics-------------------------------------------------------
        vec = torch.zeros(self.l,device = device)
        vec[0] = 1
        epsilon = torch.zeros(self.l,device = device)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :]) #+ self.bv[0]
        epsilon = epsilon + vec * torch.relu(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l,device = device)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :]) \
                #+ self.bv[i]
            epsilon = epsilon + vec * torch.relu(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2) #+ self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22) #+ self.bu
        return u, xi_

class Input(torch.nn.Module):
    def __init__(self, m, t_end, active=True,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.t_end = t_end
        self.m = m
        if active:
            std = 1
            self.u = torch.nn.Parameter(torch.randn(t_end, m, requires_grad=True,device=device) * std)
        else:
            self.u = torch.zeros(t_end, m,device=device)

    def forward(self, t,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if t < self.t_end:
            return self.u[t, :]
        else:
            return torch.zeros(self.m,device = device)


class NetworkedRENs(nn.Module):
    def __init__(self, N, Muy, Mud, n, p, n_xi, l,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n1
        self.l = l  # nel paper q
        self.Muy = Muy
        self.Mud = Mud
        self.N = N
        self.r = nn.ModuleList([RENRG(self.n[j], self.p[j], self.n_xi[j], self.l[j]) for j in range(N)])
        self.y = nn.Parameter(torch.randn(N,device=device))
        self.gammaw = torch.randn(1,device=device)

    def forward(self, t, ym, d, xim,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        yp = torch.abs(self.y)
        stopu = 0
        stopy = 0
        stop = 0
        stopx = 0
        u = torch.matmul(self.Muy, ym) + torch.matmul(self.Mud, d)
        y_list = []
        xi_list = []
        gamma_list = []
        for j, l in enumerate(self.r):
            wideu = l.n
            widey = l.m
            stopu = stopu + wideu
            stopy = stopy + widey
            gamma = 1 / (torch.sqrt(torch.tensor(2,device=device)) + yp[j])
            widex = l.n_xi
            startx = stopx
            stopx = stopx + widex
            start = stop
            stop = stop + wideu
            index = range(start, stop)
            indexx = range(startx, stopx)
            yt, xitemp = l(t, u[index], xim[indexx], gamma)
            y_list.append(yt)
            xi_list.append(xitemp)
            gamma_list.append(gamma)

        y = torch.cat(y_list)
        xi = torch.cat(xi_list)

        return y, xi, gamma_list


class SystemsOfSprings(torch.nn.Module):
    def __init__(self, xbar,Ts,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.xbar = xbar
        self.n_agents = 4
        self.n = 16
        self.m = 8
        self.h = Ts


    def f(self, t, x,u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        m1, m2, m3, m4 = 1,1,1,1
        kspringGround = 1
        cdampGround = 2
        kspringInter = 1
        cdampInter = 0

        k1,k2,k3,k4 = kspringGround,kspringGround,kspringGround,kspringGround
        k5,k6,k7,k8,k9,k10 = kspringInter,kspringInter,kspringInter,kspringInter,kspringInter,kspringInter
        c1,c2,c3,c4 = cdampGround,cdampGround,cdampGround,cdampGround
        c5,c6,c7,c8,c9,c10 = cdampInter,cdampInter,cdampInter,cdampInter,cdampInter,cdampInter

        maxlength5 = 4
        maxlength6 = 1.5
        maxlength7 = 4
        maxlength8 = 1.5
        maxlength9 = torch.sqrt(torch.tensor(maxlength7 ** 2 + maxlength8 ** 2, device=device))
        maxlength10 = torch.sqrt(torch.tensor(maxlength7 ** 2 + maxlength8 ** 2, device=device))

        Mx = torch.zeros(self.n_agents,self.n,device=device)
        Mx[0,0] = 1
        Mx[1, 4] = 1
        Mx[2, 8] = 1
        Mx[3, 12] = 1
        px = torch.matmul(Mx,x)
        My = torch.zeros(self.n_agents, self.n,device=device)
        My[0,1] = 1
        My[1, 5] = 1
        My[2, 9] = 1
        My[3, 13] = 1
        py = torch.matmul(My, x)
        Mvx = torch.zeros(self.n_agents, self.n,device=device)
        Mvx[0, 2] = 1
        Mvx[1, 6] = 1
        Mvx[2, 10] = 1
        Mvx[3, 14] = 1
        vx = torch.matmul(Mvx, x)
        Mvy = torch.zeros(self.n_agents, self.n,device=device)
        Mvy[0, 3] = 1
        Mvy[1, 7] = 1
        Mvy[2, 11] = 1
        Mvy[3, 15] = 1
        vy = torch.matmul(Mvy, x)

        mv1 = torch.zeros(2,16,device=device)
        mv1[0,2] = 1
        mv1[1,3] = 1
        mv2 = torch.zeros(2, 16,device=device)
        mv2[0, 6] = 1
        mv2[1, 7] = 1
        mv3 = torch.zeros(2, 16,device=device)
        mv3[0, 10] = 1
        mv3[1, 11] = 1
        mv4 = torch.zeros(2, 16,device=device)
        mv4[0, 14] = 1
        mv4[1, 15] = 1
        mp = torch.zeros(2,16,device=device)
        Mp = torch.cat((mv1,mp,mv2,mp,mv3,mp,mv4,mp),0)






        B = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1/m1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1/m1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1/m2, 0, 0, 0, 0, 0],
            [0, 0, 0, 1/m2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1/m3, 0, 0, 0],
            [0, 0, 0, 0, 0, 1/m3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1/m4, 0],
            [0, 0, 0, 0, 0, 0, 0, 1/m4],
        ],device=device)


        xt = torch.matmul(Mx,self.xbar)
        yt = torch.matmul(My,self.xbar)

        deltax = px.repeat(1,1).T-px.repeat(1,1)
        deltay = py.repeat(1,1).T-py.repeat(1,1)
        deltavx = vx.repeat(1,1).T-vx.repeat(1,1)
        deltavy = vy.repeat(1,1).T-vy.repeat(1,1)
        deltaxt = px-xt
        deltayt = py-yt


        projx = torch.cos(torch.atan2(deltay, deltax))
        projy = torch.sin(torch.atan2(deltay, deltax))
        projvx = torch.cos(torch.atan2(deltavy, deltavx))
        projvy = torch.sin(torch.atan2(deltavy, deltavx))
        projxt = torch.cos(torch.atan2(deltayt, deltaxt))
        projyt = torch.sin(torch.atan2(deltayt, deltaxt))
        projvxt = torch.cos(torch.atan2(vy, vx))
        projvyt = torch.sin(torch.atan2(vy, vx))

        Fc01 = c1 * torch.sqrt(vx[0] ** 2 + vy[0] ** 2)
        Fc02 = c2 * torch.sqrt(vx[1] ** 2 + vy[1] ** 2)
        Fc03 = c3 * torch.sqrt(vx[2] ** 2 + vy[2] ** 2)
        Fc04 = c4 * torch.sqrt(vx[3] ** 2 + vy[3] ** 2)

        Fk01 = k1 * torch.sqrt(deltaxt[0] ** 2 + deltayt[0] ** 2)
        Fk02 = k2 * torch.sqrt(deltaxt[1] ** 2 + deltayt[1] ** 2)
        Fk03 = k3 * torch.sqrt(deltaxt[2] ** 2 + deltayt[2] ** 2)
        Fk04 = k4 * torch.sqrt(deltaxt[3] ** 2 + deltayt[3] ** 2)

        Fk5 = k5 * (torch.sqrt(deltax[0, 1] ** 2 + deltay[0, 1] ** 2) - maxlength5)
        Fk6 = k6 * (torch.sqrt(deltax[1, 2] ** 2 + deltay[1, 2] ** 2) - maxlength6)
        Fk7 = k7 * (torch.sqrt(deltax[2, 3] ** 2 + deltay[2, 3] ** 2) - maxlength7)
        Fk8 = k8 * (torch.sqrt(deltax[3, 0] ** 2 + deltay[3, 0] ** 2) - maxlength8)
        Fk9 = k9 * (torch.sqrt(deltax[2, 0] ** 2 + deltay[2, 0] ** 2) - maxlength9)
        Fk10 = k10 * (torch.sqrt(deltax[3, 1] ** 2 + deltay[3, 1] ** 2) - maxlength10)

        Fc5 = c5 * torch.sqrt(deltavx[0, 1] ** 2 + deltavy[0, 1] ** 2)
        Fc6 = c6 * torch.sqrt(deltavx[1, 2] ** 2 + deltavy[1, 2] ** 2)
        Fc7 = c7 * torch.sqrt(deltavx[2, 3] ** 2 + deltavy[2, 3] ** 2)
        Fc8 = c8 * torch.sqrt(deltavx[3, 0] ** 2 + deltavy[3, 0] ** 2)
        Fc9 = c9 * torch.sqrt(deltavx[2, 0] ** 2 + deltavy[2, 0] ** 2)
        Fc10 = c10 * torch.sqrt(deltavx[3, 1] ** 2 + deltavy[3, 1] ** 2)

        Fspring1x = -(Fk5 * projx[0, 1] +  Fk8 * projx[0, 3]  +Fk9 * projx[0, 2])
        Fdamp1x = -(Fc5 * projvx[0, 1] + Fc8 * projvx[0, 3] + Fc9 * projvx[0, 2])
        Fground1x = -Fk01 * projxt[0] - Fc01 * projvxt[0]
        Fspring1y = -(Fk5 * projy[0, 1] +  Fk8 * projy[0, 3] +  Fk9 * projy[0, 2])
        Fdamp1y = -(Fc5 * projvy[0, 1] + Fc8 * projvy[0, 3] + Fc9 * projvy[0, 2])
        Fground1y = -Fk01 * projyt[0] - Fc01 * projvyt[0]

        Fspring2x = -( Fk5 * projx[1, 0] +  Fk6 * projx[1, 2] +  Fk10 * projx[1, 3])
        Fdamp2x = -(Fc5 * projvx[1, 0] + Fc6 * projvx[1, 2] + Fc10 * projvx[1, 3])
        Fground2x = -Fk02 * projxt[1] - Fc02 * projvxt[1]
        Fspring2y = -(Fk5 * projy[1, 0] + Fk6 * projy[1, 2] +  Fk10 * projy[ 1, 3])
        Fdamp2y = -(Fc5 * projvy[1, 0] + Fc6 * projvy[1, 2] + Fc10 * projvy[1, 3])
        Fground2y = -Fk02 * projyt[1] - Fc02 * projvyt[1]

        Fspring3x = -(Fk6 * projx[2, 1] + Fk7 * projx[2, 3] +  Fk9 * projx[2, 0])
        Fdamp3x = -(Fc6 * projvx[2, 1] + Fc7 * projvx[2, 3] + Fc9 * projvx[2, 0])
        Fground3x = -Fk03 * projxt[2] - Fc03 * projvxt[2]
        Fspring3y = -(Fk6 * projy[2, 1] + Fk7 * projy[2, 3] +  Fk9 * projy[2, 0])
        Fdamp3y = -(Fc6 * projvy[2, 1] + Fc7 * projvy[2, 3] + Fc9 * projvy[2, 0])
        Fground3y = -Fk03 * projyt[2] - Fc03 * projvyt[2]

        Fspring4x = -( Fk7 * projx[3, 2] +  Fk8 * projx[3, 0] + Fk10 * projx[3, 1])
        Fdamp4x = -(Fc7 * projvx[3, 2] + Fc8 * projvx[3, 0] + Fc10 * projvx[3, 1])
        Fground4x = -Fk04 * projxt[3] - Fc04 * projvxt[3]
        Fspring4y = -( Fk7 * projy[3, 2] +  Fk8 * projy[3, 0] + Fk10 * projy[3, 1])
        Fdamp4y = -(Fc7 * projvy[3, 2] + Fc8 * projvy[3, 0] + Fc10 * projvy[3, 1])
        Fground4y = -Fk04 * projyt[3] - Fc04 * projvyt[3]

        A1x = torch.tensor([
            0,
            0,
            (Fspring1x + Fdamp1x + Fground1x)/m1,
            (Fspring1y + Fdamp1y + Fground1y)/m1,
            0,
            0,
            (Fspring2x + Fdamp2x + Fground2x)/m2,
            (Fspring2y + Fdamp2y + Fground2y)/m2,
            0,
            0,
            (Fspring3x + Fdamp3x + Fground3x)/m3,
            (Fspring3y + Fdamp3y + Fground3y)/m3,
            0,
            0,
            (Fspring4x + Fdamp4x + Fground4x)/m4,
            (Fspring4y + Fdamp4y + Fground4y)/m4
            ],device=device)

        A2x = torch.matmul(Mp,x)

        Ax = A1x+A2x

        x1 = x + (Ax + torch.matmul(B, u)) * self.h

        return x1

    def forward(self, t, x, u, w):
        x1 = self.f(t, x, u)
        return x1



class Noise_reconstruction(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, omega):
        x, u = omega
        x1 = self.f(t, x, u)
        return x1


class Controller(nn.Module):
    def __init__(self, f, N, Muy, Mud, n, m, n_xi, l, use_sp=False, t_end_sp=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.use_sp = use_sp
        self.N = N
        self.n = n
        self.m = m
        self.noise_r = Noise_reconstruction(f)
        self.netREN = NetworkedRENs(N, Muy, Mud, n, m, n_xi, l)
        std = 1
        self.amplifier = torch.nn.Parameter(torch.randn(1, requires_grad=True, device=device) * std)
        if use_sp:  # setpoint that enters additively in the reconstruction of omega
            self.sp = Input(torch.tensor(16,device = device), t_end_sp, active=use_sp)


    def forward(self, t, u, xm, xi, omega):
        xr = self.noise_r(t, omega)
        w_ = xm - xr
        if self.use_sp:
            w_ = w_ + self.sp(t)
        u_, xi_, gamma = self.netREN(t, u, w_, xi)
        us = self.amplifier*u_
        return u_, xi_, gamma, us
