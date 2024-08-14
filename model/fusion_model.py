import torch
from torch import nn
from model.mlp import MLP
from model.cnn import CNN
from model.angle_linear import AngleLinear

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.mlp = MLP()
        self.cnn = CNN([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
        self.fc = AngleLinear(in_features=512, out_features=5)

    def forward(self, image_data, clinical_data):
        image_features = self.cnn(image_data)
        clinical_features = self.mlp(clinical_data)
        combined_features = torch.bmm(clinical_features, image_features)
        outputs = combined_features.view(combined_features.size(0), -1)
        outputs = self.fc(outputs)
        return outputs