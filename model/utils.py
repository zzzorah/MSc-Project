import typing
from torch import nn
import math

def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int, padding: int, dilation=1, groups=1, bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module

def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

def conv3d_pooling(in_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias)

def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)