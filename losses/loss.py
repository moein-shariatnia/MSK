import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self, weighted=True):
        super().__init__()
        self.weighted = weighted
        reduction = 'none' if weighted else 'mean'
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, preds, targets):
        loss = self.criterion(preds, targets)
        if self.weighted:
            batch, channel, size, _ = targets.size()
            num_elements = size ** 2
            mask = (targets == 0.).float()
            pos_ratios = mask.sum(dim=(2, 3)) / num_elements
            neg_ratios = 1 - pos_ratios
            weights = torch.where(targets != 0., pos_ratios.view(batch, channel, 1, 1), neg_ratios.view(batch, channel, 1, 1))

            loss = (weights * loss).mean()
        
        return loss