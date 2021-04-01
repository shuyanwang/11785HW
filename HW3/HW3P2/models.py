from utils.base import *
import torch
from torch import nn


class ModelHW3(nn.Module, ABC):
    def __init__(self, params):
        super(ModelHW3, self).__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: tuple):
        """

        :param x: padded (B,T,C)
        :param lengths:
        :return: # (T,B,C): logits
        """
        pass


class Model1(ModelHW3):
    def __init__(self, params):
        super(Model1, self).__init__(params)
        self.conv1 = nn.Conv1d(params.input_dims[0], params.conv_size, kernel_size=1)
        self.rnn = nn.GRU(params.conv_size, params.hidden_size, params.num_layer, batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths: tuple):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths
