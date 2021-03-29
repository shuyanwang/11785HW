import torch
from torch import nn
from utils.base import PairLoss, TripletLoss
from torch.nn import functional


class CenterLoss(nn.Module):
    """
    The original code is from the bootcamp (allowed according to course policy).
    I also used mask_select in place of loops -> performance improvement
    """

    def __init__(self, params):
        super(CenterLoss, self).__init__()
        self.num_classes = params.output_channels
        self.feat_dim = params.feature_dims
        self.classes = torch.arange(self.num_classes).long().to(params.device)

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distances = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) \
                    + torch.pow(self.centers, 2).sum(
                dim=1, keepdim=True).expand(self.num_classes, batch_size).transpose()

        distances -= 2 * x @ self.centers.t()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        return torch.mean(torch.masked_select(distances, mask))


class CrossEntropyCenterLoss(nn.Module):

    def __init__(self, params):
        super(CrossEntropyCenterLoss, self).__init__()
        self.lambDA = params.lambDA
        self.center = CenterLoss(params)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feature, y, label: torch.Tensor) -> torch.Tensor:
        return self.CE(y, label) + self.lambDA * self.center(feature, label)


class AdaptiveTripletMarginLoss(TripletLoss):
    def score(self, y1, y2):
        dist = torch.pairwise_distance(y1, y2)
        return 1 / (1 + dist)

    def __init__(self, m=1.0):
        super().__init__()
        self.m = m
        self.lr = 1e-2
        mean_pos = 15 * torch.ones(1)  # running average
        mean_neg = 25 * torch.ones(1)  # Init API in the future

        self.register_buffer('mean_pos', mean_pos)
        self.register_buffer('mean_neg', mean_neg)

    def forward(self, y0: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor) -> torch.Tensor:
        dist_pos = torch.pairwise_distance(y0, y_pos)
        dist_neg = torch.pairwise_distance(y0, y_neg)

        self.mean_pos = self.mean_pos * (1 - self.lr) + torch.mean(dist_pos) * self.lr
        self.mean_neg = self.mean_neg * (1 - self.lr) + torch.mean(dist_neg) * self.lr

        return torch.mean(torch.clamp(dist_pos - dist_neg + self.m, min=0.0))

    def predict(self, y1, y2, *args):
        threshold = (self.mean_pos.item() + self.mean_neg.item()) / 2

        dist = torch.pairwise_distance(y1, y2)
        return torch.where(torch.le(dist, threshold), 1, 0)


class SwapTripletCosineLoss(TripletLoss):
    def score(self, y1, y2):
        return torch.cosine_similarity(y1, y2)

    @staticmethod
    def dist(y0, y1):
        return torch.clamp(1 - torch.cosine_similarity(y0, y1), min=0.0)

    def __init__(self, m=0.1):
        super().__init__()
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=self.dist, margin=m,
                                                     swap=True)

    def forward(self, y0: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor) -> torch.Tensor:
        return self.loss(y0, y_pos, y_neg)


class SwapTripletMarginLoss(TripletLoss):
    def score(self, y1, y2):
        dist = torch.pairwise_distance(y1, y2)
        return 1 / (1 + dist)

    def __init__(self, m=1.0):
        super().__init__()
        self.loss = nn.TripletMarginWithDistanceLoss(margin=m, swap=True)

    def forward(self, y0: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor) -> torch.Tensor:
        return self.loss(y0, y_pos, y_neg)


# noinspection PyAttributeOutsideInit
class TripletCosineLoss(TripletLoss):
    def score(self, y1, y2):
        return torch.cosine_similarity(y1, y2)

    def __init__(self, m=1.0):
        super().__init__()
        self.m = m

    def forward(self, y0: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor) -> torch.Tensor:
        dist_pos = 1 - torch.cosine_similarity(y0, y_pos)
        dist_neg = 1 - torch.cosine_similarity(y0, y_neg)

        return torch.mean(torch.clamp(dist_pos - dist_neg + self.m, min=0.0))

    def predict(self, y1, y2, *args):
        pass


# noinspection PyAttributeOutsideInit
class AdaptiveContrastiveLoss(PairLoss):
    def score(self, y1, y2):
        dist = torch.pairwise_distance(y1, y2)
        return 1 / (1 + dist)

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


# noinspection PyAttributeOutsideInit
class AdaptiveCosineLoss(PairLoss):
    def score(self, y1, y2):
        return torch.cosine_similarity(y1, y2)

    def __init__(self, m=1.0):
        super().__init__()
        self.m = m
        self.lr = 1e-2  # could be modified by outside
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

        similarity = torch.cosine_similarity(x1, x2)
        mask_pos = y.ge(0.5)
        pos_mean_new = torch.mean(torch.masked_select(similarity, mask_pos))
        neg_mean_new = torch.mean(torch.masked_select(similarity, torch.logical_not(mask_pos)))

        self.mean_pos = self.mean_pos * (1 - self.lr) + pos_mean_new * self.lr
        self.mean_neg = self.mean_neg * (1 - self.lr) + neg_mean_new * self.lr

        return torch.mean(
                torch.clamp(similarity - self.m, min=0.0) * (1 - y) + (1 - similarity) * y)

    def predict(self, y1, y2, *args):
        """
        Prediction
        :param y1: (N,*)
        :param y2:
        :return: (N)
        """

        threshold = (self.mean_pos.item() + self.mean_neg.item()) / 2

        return torch.where(torch.ge(torch.cosine_similarity(y1, y2), threshold), 1, 0)


class BWayLoss(nn.Module):
    def score(self, y1, y2):
        dist = torch.pairwise_distance(y1, y2)
        return 1 / (1 + dist)

    def __init__(self):
        super(BWayLoss, self).__init__()

    def forward(self, y1, y2):
        """

        :param y1: B,F
        :param y2: B,F
        :return:
        """
        dot = torch.einsum('if,jf->ij', y1, y2)
        # dot[i,j] is the dot product of y1_i and y2_j

        return -torch.sum(functional.log_softmax(dot, dim=1))
