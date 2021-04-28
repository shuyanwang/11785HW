from utils.base import *
import torch
from torch import nn


class Encoder(nn.Module, ABC):
    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths):
        """

        :param x: padded (B,Tin,40)
        :param lengths: (B,)
        :return: # (B,Te,H), lengths_out: hidden states of the last layer
        """
        pass


class Decoder(nn.Module, ABC):
    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, encoded: torch.Tensor, encoded_lengths, gt=None, gt_lengths=None):
        """

        :param encoded: padded (B,Te,C)
        :param encoded_lengths: (B,)
        :param gt: (B,Tgt) used for teacher forcing
        :param gt_lengths: (B,) used for teacher forcing
        :return: # (B,To,): hidden states of the last layer
        """
        pass
