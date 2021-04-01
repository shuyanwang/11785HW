from utils.base import *
import torch
from torch import nn


class ModelHW3(nn.Module):
    def __init__(self, params):
        super(ModelHW3, self).__init__()
        self.params = params

    def forward(self, x: torch.Tensor, lengths: tuple):
        """

        :param x: padded (B,T,C)
        :param lengths:
        :return:
        """
        pass
