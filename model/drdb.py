import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualDenseBlock(nn.Module):
    def __init__(self, in_channels): # growth_rate=12, num_layers=3
        super(DilatedResidualDenseBlock, self).__init__()
        self.in_channels = in_channels
        # self.growth_rate = growth_rate
        # self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv3d(in_channels, 12, kernel_size=3, padding=1, dilation=1))
        self.convs.append(nn.Conv3d(in_channels + 12, 12, kernel_size=3, padding=2, dilation=2))
        self.convs.append(nn.Conv3d(in_channels + 24, 12, kernel_size=3, padding=3, dilation=3))
        
        self.resize_conv = nn.Conv3d(in_channels + 36, in_channels, kernel_size=1)

        # for i in range(num_layers):
        #     self.convs.append(nn.Conv3d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=i+1, dilation=i+1))

        # self.conv1x1 = nn.Conv3d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        inputs = x
        concat_features = [x]
        
        for conv in self.convs:
            # print(f'***')
            # for x in concat_features:
            #     print(f'x size = {x.size()}')
            x = conv(torch.cat(concat_features, dim=1))
            # x = F.relu(x)
            concat_features.append(x)
        
        # x = self.conv1x1(torch.cat(concat_features, dim=1))
        x = self.resize_conv(torch.cat(concat_features, dim=1))
        outputs = x + inputs
        return outputs