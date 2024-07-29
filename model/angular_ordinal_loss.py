
import torch
from torch import nn

class AngularOrdinalLoss(nn.Module):
    def __init__(self):
        super(AngularOrdinalLoss, self).__init__()

    def forward(self, cosine, phi, targets):
        N, C = cosine.size()
        targets = targets.view(-1, 1)  # [N, 1]
        one_hot = torch.zeros(N, C).scatter_(1, targets, 1)  # [N, C]
        
        weights = torch.abs(torch.arange(C).float() - targets.float()).view(N, C)
        loss = weights * one_hot * (torch.log(1 + torch.exp(-cosine)) + torch.log(1 + torch.exp(-phi)))
        
        return loss.mean()