from torch import nn
from model.utils import make_conv3d

class ResidualBlock(nn.Module):
    """
    a simple residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out