from torch import nn
from typing import Union


class Conv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, tuple], bias=True, stride=1, dilation=1, groups=1):
        super(Conv3dLayer, self).__init__()
        self.padding = kernel_size // 2

        self.layer_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=dilation, groups=groups, bias=bias)
        self.layer_2 = nn.BatchNorm3d(out_channels)
        self.layer_3 = nn.ReLU()

    def forward(self, input):
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x