import torch
from torch import nn
from utils.base import PairLoss


# from torch.nn import CosineEmbeddingLoss, HingeEmbeddingLoss, MarginRankingLoss
# These losses are for 1/(-1) labels


class ContrastiveLoss(PairLoss):
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

    @staticmethod
    def predict(y1, y2, threshold):
        """
        Prediction
        :param y1: (N,*)
        :param y2:
        :param threshold: float
        :return: (N)
        """
        return torch.where(torch.le(torch.pairwise_distance(y1, y2), threshold), 1, 0)
