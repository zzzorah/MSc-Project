from torch import nn

class OrdinalRegressionLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(OrdinalRegressionLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.fc(x)