from torch import nn
from model.conv3d_layer import Conv3dLayer
from model.ca_block import CABlock
from model.drdb import DilatedResidualDenseBlock
from model.angle_linear import AngleLinear
from model.res_cbam_layer import ResCBAMLayer

debug = False
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv1 = Conv3dLayer(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = Conv3dLayer(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config # [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]]
        self.last_channel = 4
        self.cbam = ResCBAMLayer(4, 32)
        self.ca = CABlock(self.last_channel, self.last_channel)
        self.drdb = DilatedResidualDenseBlock(self.last_channel)
        self.l = 3
        self.m = 3
        self.n = 4
        self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        self.fc = AngleLinear(in_features=2048, out_features=5)

    def forward(self, inputs):
        # stage 1
        outputs = self.conv1(inputs)

        # stage 2
        outputs = self.conv2(outputs)
        outputs = self.cbam(outputs)

        # stage 3 - L
        for _ in range(self.l):
            outputs = self.drdb(outputs)
        outputs = self.ca(outputs)
        
        # stage 4 - M
        for _ in range(self.m):
            outputs = self.drdb(outputs)
        outputs = self.ca(outputs)

        # stage 5 - N
        for _ in range(self.n):
            outputs = self.drdb(outputs)
        outputs = self.ca(outputs)

        # stage 6
        outputs = self.avg_pooling(outputs)
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)

        return outputs