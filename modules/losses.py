import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def get_loss(loss_type, weight=None, gamma=2):
    if loss_type=='cross_entropy':
        loss_fnc = nn.CrossEntropyLoss()
    elif loss_type=='weighted_cross_entropy':
        loss_fnc = nn.CrossEntropyLoss(weight=weight)
    elif loss_type=='focal':
        loss_fnc = FocalLoss(gamma=gamma)

    return loss_fnc