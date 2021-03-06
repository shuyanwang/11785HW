import torch
# from torch import nn
from utils.base import PairLoss


# from torch.nn import CosineEmbeddingLoss, HingeEmbeddingLoss, MarginRankingLoss
# These nn losses are for 1/(-1) labels


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
        dist = torch.pairwise_distance(y1, y2, p=2)
        return torch.where(torch.le(dist, threshold), 1, 0)


# noinspection PyAttributeOutsideInit
class AdaptiveContrastiveLoss(PairLoss):
    def __init__(self, m=1.0):
        super().__init__()
        self.m = m
        self.lr = 1e-2
        mean_pos = 15 * torch.ones(1)  # running average
        mean_neg = 25 * torch.ones(1)  # Init API in the future

        self.register_buffer('mean_pos', mean_pos)
        self.register_buffer('mean_neg', mean_neg)

    def forward(self, x1, x2, y):
        """
        Hinge embedding loss for 0-1, 2 class
        :param x1:
        :param x2:
        :param y: 0 - 1
        :return:
        """

        dist = torch.pairwise_distance(x1, x2, p=2)
        mask_pos = y.ge(0.5)
        pos_mean_new = torch.mean(torch.masked_select(dist, mask_pos))
        neg_mean_new = torch.mean(torch.masked_select(dist, torch.logical_not(mask_pos)))

        self.mean_pos = self.mean_pos * (1 - self.lr) + pos_mean_new * self.lr
        self.mean_neg = self.mean_neg * (1 - self.lr) + neg_mean_new * self.lr

        return torch.mean(torch.clamp(self.m - dist, min=0.0) * (1 - y) + dist * y)

    def predict(self, y1, y2, *args):
        """
        Prediction
        :param y1: (N,*)
        :param y2:
        :return: (N)
        """

        threshold = (self.mean_pos.item() + self.mean_neg.item()) / 2

        dist = torch.pairwise_distance(y1, y2, p=2)
        return torch.where(torch.le(dist, threshold), 1, 0)


class CosineLoss(PairLoss):
    def __init__(self, m=1.0):
        super().__init__()
        self.m = m

    def forward(self, x1, x2, y):
        """
        Hinge embedding loss for 0-1, 2 class
        :param x1:
        :param x2:
        :param y: 0 - 1
        :return:
        """
        similarity = torch.cosine_similarity(x1, x2)
        return torch.mean(
                torch.clamp(similarity - self.m, min=0.0) * (1 - y) + (1 - similarity) * y)

    @staticmethod
    def predict(y1, y2, threshold):
        """
        Prediction
        :param y1: (N,*)
        :param y2:
        :param threshold: float
        :return: (N)
        """
        return torch.where(torch.ge(torch.cosine_similarity(y1, y2), threshold), 1, 0)
