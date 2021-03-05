import torch
from torch import nn

from torch.nn import CosineEmbeddingLoss, HingeEmbeddingLoss, MarginRankingLoss


class ContrastiveLoss(nn.Module):
    def __init__(self, m=1.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, x1, x2, y):
        """
        Hinge embedding loss for 0-1, 2 class
        :param x1:
        :param x2:
        :param y: 0 - 1
        :return:
        """

        dist = torch.pairwise_distance(x1, x2, p=2)
        return torch.mean(torch.clamp(self.m - dist, min=0.0) * (1 - y) + dist * y)
