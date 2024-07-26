import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalLoss(nn.Module):
    def __init__(self):
        super(OrdinalLoss, self).__init__()

    def forward(self, outputs, targets):
        batch_size, num_classes = outputs.size()
        targets = targets.view(-1, 1)  # Reshape targets to (batch_size, 1)
        
        mask = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1).to(outputs.device)
        mask = mask < targets

        loss = F.binary_cross_entropy_with_logits(outputs, mask.float())
        return loss