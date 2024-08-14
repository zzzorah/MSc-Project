# Implementation base on https://doi.org/10.1007/s11547-023-01730-6
import torch
import torch.nn as nn

class DilatedResidualDenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DilatedResidualDenseBlock, self).__init__()
        self.in_channels = in_channels

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv3d(in_channels, 12, kernel_size=3, padding=1, dilation=1))
        self.convs.append(nn.Conv3d(in_channels + 12, 12, kernel_size=3, padding=2, dilation=2))
        self.convs.append(nn.Conv3d(in_channels + 24, 12, kernel_size=3, padding=3, dilation=3))
        
        self.resize_conv = nn.Conv3d(in_channels + 36, in_channels, kernel_size=1)

    def forward(self, x):
        inputs = x
        concat_features = [x]
        
        for conv in self.convs:
            x = conv(torch.cat(concat_features, dim=1))
            concat_features.append(x)

        x = self.resize_conv(torch.cat(concat_features, dim=1))
        outputs = x + inputs
        return outputs