from torch import nn
from model.utils import conv3d_same_size, conv3d_pooling
from model.res_net import ResidualBlock
from model.res_cbam_layer import ResCBAMLayer
from model.ordinal_regression_layer import OrdinalRegressionLayer

debug = True
class ConvRes(nn.Module):
    def __init__(self, config):
        super(ConvRes, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResCBAMLayer(4, 32)
        layers = []
        i = 0
        for stage in config:
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
            layers.append(ResCBAMLayer(self.last_channel, 32//(2**i)))
        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        self.fc = OrdinalRegressionLayer(in_features=self.last_channel, num_classes=5)

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_cbam(out)
        out = self.layers(out)
        if debug:
            print(out.size())
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        out = self.fc(out)
        return out