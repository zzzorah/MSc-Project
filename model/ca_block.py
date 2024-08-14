# Based on the 2D implementation, modify it to 3D. Source: https://github.com/houqb/CoordAttention/blob/main/coordatt.py
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CABlock(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CABlock, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()
        
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        _, _, d, h, w = x.size()
        x_d = self.pool_d(x) # (N, C, D, H=1, W=1) => N x C x D x 1 x 1
        x_h = self.pool_h(x) # (N, C, D=1, H, W=1) => N x C x 1 x H x 1
        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4) # (N, C, D=1, H, W=1) => (N, C, H, D=1, W=1) => N x C x H x 1 x 1
        x_w = self.pool_w(x) # (N, C, D=1, H=1, W) => N x C x 1 x 1 x W
        x_w = self.pool_w(x).permute(0, 1, 4, 3, 2) # (N, C, D=1, H=1, W) => (N, C, W, H=1, D=1) => N x C x W x 1 x 1

        y = torch.cat([x_d, x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 4, 3, 2)

        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h * a_d

        return out