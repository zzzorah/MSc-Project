from torch import nn
from model.utils import conv3d_same_size, conv3d_pooling
from model.res_net import ResidualBlock
from model.res_cbam_layer import ResCBAMLayer
from model.ordinal_regression_layer import OrdinalRegressionLayer
from model.p_rank_layer import PRankLayer
from model.ca_block import CABlock
from model.drdb import DilatedResidualDenseBlock

debug = False
class ConvRes(nn.Module):
    def __init__(self, config):
        super(ConvRes, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config # [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]]
        self.last_channel = 4
        # self.first_cbam = ResCBAMLayer(4, 32)
        # layers = []
        # i = 0
        # for stage in config:
        #     i = i+1
        #     layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
        #     for channel in stage:
        #         layers.append(ResidualBlock(self.last_channel, channel))
        #         self.last_channel = channel
        #     layers.append(ResCBAMLayer(self.last_channel, 32//(2**i)))
        # self.layers = nn.Sequential(*layers)
        # self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        # # self.fc = OrdinalRegressionLayer(in_features=self.last_channel, num_classes=5)
        self.ca = CABlock(self.last_channel, self.last_channel)
        self.drdb = DilatedResidualDenseBlock(self.last_channel)
        self.l = 3
        self.m = 3
        self.n = 4
        self.avg_pooling = nn.AvgPool3d(kernel_size=8, stride=8)
        # self.prank = PRankLayer(in_features=self.last_channel, num_classes=5)
        self.prank = PRankLayer(in_features=256, num_classes=5)

    def forward(self, inputs):
        # stage 1
        # print('stage 1')
        outputs = self.conv1(inputs)

        # stage 2
        # print('stage 2-1')
        outputs = self.conv2(outputs)
        # print('stage 2-2')
        # print(f'outputs size = {outputs.size()}')
        outputs = self.ca(outputs)

        # stage 3 - L
        # print('stage 3')
        for _ in range(self.l):
            outputs = self.drdb(outputs)
        # print(f'outputs size = {outputs.size()}')
        outputs = self.ca(outputs)
        
        # stage 4 - M
        # print('stage 4')
        for _ in range(self.m):
            outputs = self.drdb(outputs)
        # print(f'outputs size = {outputs.size()}')
        outputs = self.ca(outputs)

        # stage 5 - N
        # print('stage 5')
        for _ in range(self.n):
            outputs = self.drdb(outputs)
        # print(f'outputs size = {outputs.size()}')
        outputs = self.ca(outputs)

        # stage 6
        # print('stage 6-1')
        # print(f'outputs size = {outputs.size()}')
        outputs = self.avg_pooling(outputs)
        # print('stage 6-2')
        # print(f'outputs size = {outputs.size()}')
        outputs = outputs.view(outputs.size(0), -1)
        # print('stage 6-3')
        # print(f'outputs size = {outputs.size()}')
        outputs, scores = self.prank(outputs)

        return outputs, scores
        # if debug:
        #     print(inputs.size())
        # out = self.conv1(inputs)
        # if debug:
        #     print(out.size())
        # out = self.conv2(out)
        # if debug:
        #     print(out.size())
        # out = self.first_cbam(out)
        # out = self.layers(out)
        # if debug:
        #     print(out.size())
        # out = self.avg_pooling(out)
        # out = out.view(out.size(0), -1)
        # if debug:
        #     print(out.size())
        # # out = self.fc(out)
        # out, scores = self.prank(out)
        # return out, scores