from typing import List
import numpy as np
from utils.base import *
from HW4 import ParamsHW4
import torch
from torch import nn, Tensor


class Encoder(nn.Module, ABC):
    def __init__(self, params: ParamsHW4):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths):
        """

        :param x: padded (B,Tin,40)
        :param lengths: (B,) list
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


class PBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """

        :param input_dim: x.shape[2]
        :param hidden_dim:
        """
        super(PBLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=1, bidirectional=True,
                           batch_first=True)

    def forward(self, x: Tensor, lengths: List):
        """

        :param x: (B,T,C) padded -> just a tensor
        :param lengths
        :return: (B,T//2,H*2), lengths
        """
        B, T, C = x.shape

        if T // 2 == 1:
            x = x[:, :-1, :]

        x = x.contiguous().view(B, T // 2, C * 2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        lengths = [length // 2 for length in lengths]
        return x, lengths


class Encoder1(Encoder):
    def __init__(self, params):
        super(Encoder1, self).__init__(params)

        self.first = nn.LSTM(params.input_dims, params.hidden_encoder, batch_first=True)
        rnn = []
        # 1 + 3

        for i in range(params.layer_encoder):
            rnn.append(PBLSTM(params.hidden_encoder * 2, params.hidden_encoder))

        self.rnn = nn.ModuleList(rnn)

        self.key_network = nn.Linear(params.hidden_encoder * 2, params.hidden_decoder)
        self.value_network = nn.Linear(params.hidden_encoder * 2, params.hidden_decoder)

    def forward(self, x: torch.Tensor, lengths):
        """

        :param x: B,T,C
        :param lengths: List
        :return: x, k ,v, lengths
        """
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        x = self.first(x)[0]

        for layer in self.rnn:
            x, lengths = layer(x, lengths)

        k = self.key_network(x)  # (B,T//(2**num_layer),hidden_decoder)
        v = self.value_network(x)  # same as k

        return x, k, v, lengths


class Attention(nn.Module, ABC):
    def __init__(self, param: ParamsHW4):
        super(Attention, self).__init__()
        self.param = param

    @abstractmethod
    def forward(self, query, key, value, mask):
        pass


class DotAttention(Attention):
    """
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    """

    def __init__(self, param):
        super(DotAttention, self).__init__(param)

    def forward(self, query, key, value, lengths):
        """

        :param query: (B,hd)
        :param key: (B,Tout_e,hd)
        :param value: (B,Tout_e,hd)
        :param lengths: List
        :return: context, attention (for visualization)
        """

        query = torch.unsqueeze(query, -1)  # (B,hd,1)
        energy = torch.bmm(key, query).squeeze(-1)  # (B,Toe)

        mask = torch.arange(energy.shape[1]).unsqueeze(0) >= torch.as_tensor(energy).unsqueeze(1)
        mask.to(self.param.device)
        energy = torch.masked_fill(energy, mask, -1e9)

        attention = torch.softmax(energy, dim=1).unsqueeze(1)  # (B,1,Toe)

        context = torch.bmm(attention, value).squeeze(1)  # (B,hd)

        return context, attention
