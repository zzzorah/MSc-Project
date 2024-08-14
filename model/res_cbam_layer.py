# Copy from https://github.com/fei-aiart/NAS-Lung?tab=readme-ov-file models/cnn_res.py
from torch import nn
import torch

class ResCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)
        self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sp_Softmax = nn.Softmax(1)
        self.sp_sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)
        # x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        a = self.ch_Linear1(x_ch_avg_pool)
        x_ch_avg_linear = self.ch_Linear2(a)

        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))
        ch_out = (self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x
        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        x_sp_avg_pool = torch.sum(ch_out, 1, keepdim=True) / self.in_planes
        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)
        sp_out = self.sp_Conv(sp_conv1)
        sp_out = self.sp_sigmoid(sp_out.view(x.size(0), -1)).view(x.size(0), 1, x.size(2), x.size(3), x.size(4))
        out = sp_out * x + x
        return out
