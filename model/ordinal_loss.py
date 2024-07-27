import torch
import torch.nn as nn
import torch.nn.functional as F
from logs.logging_config import logger_debug

class OrdinalLoss(nn.Module):
    def __init__(self):
        super(OrdinalLoss, self).__init__()

    def forward(self, outputs, targets):
        batch_size, num_classes = outputs.size()
        targets = targets.view(-1, 1)  # Reshape targets to (batch_size, 1)
        logger_debug.debug(f'[OrdinalLoss] targets = {targets}')
        
        mask = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1).to(outputs.device)
        logger_debug.debug(f'[OrdinalLoss] init mask = {mask}')
        mask = mask < targets
        logger_debug.debug(f'[OrdinalLoss] mask = {mask}')

        loss = F.binary_cross_entropy_with_logits(outputs, mask.float())
        return loss