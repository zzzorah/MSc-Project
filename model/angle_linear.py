# Copy from https://github.com/fei-aiart/NAS-Lung?tab=readme-ov-file models/net_sphere.py
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch
from model.utils import myphi

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        # self.mlambda = [
        #     lambda x: x ** 0,
        #     lambda x: x ** 1,
        #     lambda x: 2 * x ** 2 - 1,
        #     lambda x: 4 * x ** 3 - 3 * x,
        #     lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        #     lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        # ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len  (128*512)
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        # w = 512*227
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = 8 * cos_theta ** 4 - 8 * cos_theta ** 2 + 1
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)