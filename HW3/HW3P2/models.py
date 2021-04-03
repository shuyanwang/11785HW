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


class ResNetK3S1C64(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1),
                           self.residual_layer(512, layers[3], 1)]

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


class ResNetK3S1C64L3(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1)]

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


class Model6(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1C64([2, 2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(512, params.hidden_size, params.num_layer,
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


class Model7(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1C64L3([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)

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


class Model8(ModelHW3):
    """
    Orthogonal Init
    """

    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1C64L3([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

        for name, W in self.rnn.named_parameters():
            if 'weight_hh' in name:
                for i in range(3):
                    nn.init.orthogonal_(
                            W[i * self.params.hidden_size:(i + 1) * self.params.hidden_size, :])

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)

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


class ResNetK5S1C64L3(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1)]

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


class Model9(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK5S1C64L3([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)

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


class ResNetK3S1C64L3NoPool(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1)]

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
        x = torch.relu(self.bn(self.conv(x)))
        x = self.residual_layers(x)

        return x


class Model10(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1C64L3NoPool([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class ResNetK1S1C64L3(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1)]

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


class Model11(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK1S1C64L3([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)
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


class ResBlock2D(nn.Module):
    def __init__(self, cin, cout, stride=1, down_sample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)

        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)

        self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)) if down_sample else None

    def forward(self, x: torch.Tensor):
        out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))

        down_sampled = self.downsample(x) if self.downsample is not None else x

        return torch.relu(out + down_sampled)


class ResNet2D(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv2d(cin, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cin = 32
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(32, layers[0], 1),
                           self.residual_layer(64, layers[1], 1),
                           self.residual_layer(128, layers[2], 1)]

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

        layers = [ResBlock2D(self.cin, cout, stride, stride > 1 or self.cin != cout)]
        self.cin = cout
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(self.cin, cout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return: If embedding, return the flattened feature vector
        """
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        x = self.residual_layers(x)

        return x


class Model12(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNet2D([2, 2, 2], 1)

        self.rnn = nn.GRU(128 * 20, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        # x: (B,T,C)
        x = torch.unsqueeze(x, 1)  # (B,1,T,C)
        x = self.conv1(x)
        x = torch.relu(x)  # (B,D,T,C)
        x = torch.transpose(x, 1, 2)  # (B,T,D,C)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))  # (B,T,D * C)

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


class ResNetK3S1C64L3Init(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv1d(cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1)]

        self.residual_layers = nn.Sequential(*residual_layers)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
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


class Model13(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNetK3S1C64L3Init([2, 2, 2], params.input_dims[0])

        self.rnn = nn.GRU(256, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        x = self.conv1(torch.transpose(x, 1, 2))
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)  # (B,T,C)

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


class ResNet2DFC(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv2d(cin, 64, kernel_size=(41, 11), stride=(2, 2), padding=(20, 0),
                              bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1))

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1),
                           self.residual_layer(512, layers[3], 1)]

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

        layers = [ResBlock2D(self.cin, cout, stride, stride > 1 or self.cin != cout)]
        self.cin = cout
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(self.cin, cout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (B,1,T,C)
        :return: If embedding, return the flattened feature vector
        """
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        x = self.residual_layers(x)

        return x


class Model14(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNet2DFC([2, 2, 2, 2], 1)
        self.fc1 = nn.Linear(512 * 8, params.conv_size)

        self.rnn = nn.GRU(params.conv_size, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        # x: (B,T,C)
        x = torch.unsqueeze(x, 1)  # (B,1,T,C)
        x = self.conv1(x)
        x = torch.relu(x)  # (B,D,T,C)
        x = torch.transpose(x, 1, 2)  # (B,T,D,C)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))  # (B,T,D * C)
        x = self.fc1(x)

        lengths = torch.div(lengths, 4, rounding_mode='floor').long()

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class ResNet2DFC2(nn.Module):
    def __init__(self, layers, cin):
        super().__init__()

        self.conv = nn.Conv2d(cin, 64, kernel_size=(41, 11), stride=(2, 2), padding=(20, 0),
                              bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1))

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 2),
                           self.residual_layer(256, layers[2], 2),
                           self.residual_layer(512, layers[3], 2)]

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

        layers = [ResBlock2D(self.cin, cout, stride, stride > 1 or self.cin != cout)]
        self.cin = cout
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(self.cin, cout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (B,1,T,C)
        :return: If embedding, return the flattened feature vector
        """
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        x = self.residual_layers(x)

        return x


class Model15(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = ResNet2DFC2([2, 2, 2, 2], 1)
        self.fc1 = nn.Linear(512, params.conv_size)

        self.rnn = nn.GRU(params.conv_size, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.linear = nn.Linear(params.hidden_size, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        # x: (B,T,C)
        x = torch.unsqueeze(x, 1)  # (B,1,T,C)
        x = self.conv1(x)
        x = torch.relu(x)  # (B,D,T,C)
        x = torch.transpose(x, 1, 2)  # (B,T,D,C)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))  # (B,T,D * C)
        x = self.fc1(x)

        lengths = torch.div(lengths, 32, rounding_mode='floor').long()

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.linear(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths


class LayerResBlock(nn.Module):
    def __init__(self):
        super(LayerResBlock, self).__init__()
        self.norm = nn.LayerNorm(20)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.conv = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))

    def forward(self, x):
        y = self.drop(self.gelu(self.norm(x)))
        return x + self.conv(y)


class Model16(ModelHW3):
    def __init__(self, params):
        super().__init__(params)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=(2, 2), padding=(1, 1))
        self.res_blocks = nn.Sequential(*[LayerResBlock() for _ in range(7)])
        self.fc1 = nn.Linear(32 * 20, self.params.conv_size)

        self.rnn = nn.GRU(params.conv_size, params.hidden_size, params.num_layer,
                          batch_first=True,
                          dropout=params.dropout, bidirectional=params.bi)
        self.fc2 = nn.Linear(params.hidden_size, 128)
        self.drop = nn.Dropout(params.dropout)
        self.fc3 = nn.Linear(128, params.output_channels)

    def forward(self, x: torch.Tensor, lengths):
        # x: (B,T,C)
        x = torch.unsqueeze(x, 1)  # (B,1,T,C)
        x = self.conv1(x)  # (B,D,T,C)
        x = torch.relu(x)  # (B,D,T,C)
        x = self.res_blocks(x)  # (B,D,T,C)
        x = torch.transpose(x, 1, 2)  # (B,T,D,C)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))  # (B,T,D * C)
        x = self.fc1(x)  # (B.T,Conv_size)

        lengths = torch.div(lengths, 2, rounding_mode='floor').long()

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)
        if self.params.bi:
            x = x[:, :, :self.params.hidden_size] + x[:, :, self.params.hidden_size:]

        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, 2)
        return x, out_lengths
