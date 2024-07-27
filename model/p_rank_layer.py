import torch
import torch.nn as nn

class PRankLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(PRankLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        self.thresholds = nn.Parameter(torch.linspace(-1, 1, num_classes - 1))

    def forward(self, x):
        scores = self.fc(x)
        thresholds = torch.cat([self.thresholds, torch.tensor([float('inf')]).to(x.device)])
        outputs = torch.sum(scores > thresholds.unsqueeze(0), dim=1)
        return outputs, scores
