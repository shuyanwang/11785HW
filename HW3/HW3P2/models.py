from utils.base import *
import torch
from torch import nn


class ModelHW3(nn.Module, ABC):
    def __init__(self, params):
        super(ModelHW3, self).__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths):
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

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class Model2(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = nn.Conv1d(params.input_dims[0], params.conv_size, kernel_size=1)
        self.rnn = nn.LSTM(params.conv_size, params.hidden_size, params.num_layer, batch_first=True,
                           dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class Model3(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = nn.Conv1d(params.input_dims[0], params.conv_size, kernel_size=3)
        self.rnn = nn.GRU(params.conv_size, params.hidden_size, params.num_layer, batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        lengths_x = lengths - 2
        x = nn.utils.rnn.pack_padded_sequence(x, lengths_x, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


"""
My Code from HW2P2
"""


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, down_sample=False):
        super().__init__()

        self.conv1 = nn.Conv1d(cin, cout, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(cout)

        self.conv2 = nn.Conv1d(cout, cout, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(cout)

        self.downsample = nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(cout)) if down_sample else None

    def forward(self, x: torch.Tensor):
        out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))

        down_sampled = self.downsample(x) if self.downsample is not None else x

        return torch.relu(out + down_sampled)


class ResNetK3S1(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 32
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(32, layers[0], 1),
                           self.residual_layer(64, layers[1], 1),
                           self.residual_layer(128, layers[2], 1),
                           self.residual_layer(256, layers[3], 1)]

        self.residual_layers = nn.Sequential(*residual_layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def residual_layer(self, cout, num_blocks, stride):

        if num_blocks == 0:
            return nn.Identity()

        layers = [ResBlock(self.cin, cout, stride, stride > 1 or self.cin != cout)]
        self.cin = cout
        for _ in range(1, num_blocks):
            layers.append(ResBlock(self.cin, cout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return: If embedding, return the flattened feature vector
        """
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        x = self.residual_layers(x)

        return x


class ResNet10(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetK3S1([2, 2, 2, 2], params.input_dims[0])

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Model4(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNet10(params)

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)
        # print(torch.max(lengths))
        # for (x_i, l_i) in zip(x, lengths):
        #     print(x_i.shape[0], l_i)

        lengths = torch.div(lengths, 2, rounding_mode='floor').long()

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class Model5(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1([3, 4, 6, 3], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)
        # print(torch.max(lengths))
        # for (x_i, l_i) in zip(x, lengths):
        #     print(x_i.shape[0], l_i)

        lengths = torch.div(lengths, 2, rounding_mode='floor').long()

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths
