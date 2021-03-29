from typing import List

import numpy as np
from utils.base import *
from hw2_classification import ParamsHW2Classification
import torch.nn.functional as F
from dataclasses import asdict, dataclass

"""
Note1:
    The ResNet architecture is from https://arxiv.org/abs/1512.03385 (ResNet) and the
    pytorch doc at https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
    It is a simplified version of the ResNet and it adapts to HW2.
    
    As per course policy at https://piazza.com/class/khtqzctrciu1fp?cid=71,
    I looked at some public resources, including:
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
    https://github.com/KaimingHe/deep-residual-networks

Note 2:
    The EfficientNet architecture is from https://arxiv.org/abs/1905.11946
    As per course policy, I looked at open source code at 
    https://github.com/lukemelas/EfficientNet-PyTorch. 
    However, I learned the algorithm flow and wrote my own code at the end,
    including adaptation to HW2 and changes in data structures,
    as well as other implementation techniques. For example, dropout on connections are
    written as a method in order to better utilize nn.Module's properties. There are also
    significant simplifications, including the same-padding conv2d imitating tensorflow
"""


class ResBlock(nn.Module):
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


class ResNetK3S1(nn.Module):
    def __init__(self, layers, embedding=False,
                 num_classes=4000):
        super().__init__()
        self.embedding = embedding

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value

        residual_layers = [self.residual_layer(64, layers[0], 1),
                           self.residual_layer(128, layers[1], 1),
                           self.residual_layer(256, layers[2], 1),
                           self.residual_layer(512, layers[3], 1)]

        self.residual_layers = nn.Sequential(*residual_layers)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        if not embedding:
            self.linear = nn.Linear(512, num_classes)
        else:
            self.linear = None

        #### He Initialization
        # from https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
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

        x = torch.flatten(self.pool2(x), 1)

        if self.embedding:
            return F.relu(x)

        return self.linear(x)

    def features_no_pool(self, x):
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        x = torch.flatten(x, 1)
        return x


class ResNet34K3S1RC(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetK3S1([3, 4, 6, 3], embedding=True, num_classes=params.feature_dims)
        self.fc = nn.Linear(params.feature_dims, params.output_channels)

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        return features, self.fc(torch.relu(features))


@dataclass
class EfficientNetParams:
    width_coefficients: float = field(default=1.4)
    depth_coefficients: float = field(default=1.8)
    batch_norm_momentum: float = field(default=1e-2)
    batch_norm_epsilon: float = field(default=1e-3)
    drop_connect_rate: float = field(default=0.2)
    depth_divisor: int = field(default=8)
    cin: int = field(default=3)
    cout_conv0: int = field(default=48)
    cout_conv1: int = field(default=1792)


@dataclass
class MBBlockParams:
    num_repeat: int = field(default=1)  # repeat start at 2
    kernel_size: int = field(default=3)
    stride: int = field(default=1)
    expand_ratio: int = field(default=6)
    input_filters: int = field(default=1)
    output_filters: int = field(default=1)
    se_ratio: float = field(default=0.25)
    id_skip: bool = field(default=False)


basicParamsList = [
    MBBlockParams(expand_ratio=1, input_filters=32, output_filters=16),
    MBBlockParams(num_repeat=2, stride=2, input_filters=16, output_filters=24),
    MBBlockParams(num_repeat=2, stride=2, input_filters=24, output_filters=40, kernel_size=5),
    MBBlockParams(num_repeat=3, stride=2, input_filters=40, output_filters=80),
    MBBlockParams(num_repeat=3, input_filters=80, output_filters=112, kernel_size=5),
    MBBlockParams(num_repeat=4, stride=2, input_filters=112, output_filters=192, kernel_size=5),
    MBBlockParams(num_repeat=1, stride=1, input_filters=192, output_filters=320),
]


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


def round_filter(filters, width_coefficients, depth_divisor: int):
    filters = int(filters * width_coefficients)
    return max(depth_divisor,
               (int(filters + depth_divisor / 2) // depth_divisor) * depth_divisor)


def round_depth(repeats, depth_coefficients):
    return int(np.ceil(repeats * depth_coefficients))


class EfficientNetB4(Model):
    """
    Caller shell for Model API
    """

    def __init__(self, params):
        super().__init__(params)
        e_params = EfficientNetParams()
        self.net = EfficientNet(params, e_params, basicParamsList)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MBConvBlock(nn.Module):
    def drop_connect(self, x):
        if not self.training:
            return x

        distribution = np.random.binomial(p=1 - self.dropout_connect_rate,
                                          size=(x.shape[0], 1, 1, 1), n=1)
        distribution_tensor = torch.as_tensor(distribution, device=self.device)
        return x / (1 - self.dropout_connect_rate) * distribution_tensor

    def __init__(self, params: ParamsHW2Classification, e_params: EfficientNetParams,
                 b_params: MBBlockParams):
        super().__init__()
        self.device = params.device  # needed in dropConnect
        self.block_params = b_params  # needed in forward
        self.dropout_connect_rate = 0  # also needed in forward; will be modified before use

        cout = b_params.input_filters * b_params.expand_ratio

        if b_params.expand_ratio > 1:
            self.conv0 = SamePaddingConv2D(cin=b_params.input_filters,
                                           cout=cout, kernel_size=1, size=params.size)
            self.bn0 = nn.BatchNorm2d(num_features=cout, eps=e_params.batch_norm_epsilon,
                                      momentum=e_params.batch_norm_momentum)

        self.conv1 = SamePaddingConv2D(cin=cout, cout=cout, groups=cout,
                                       kernel_size=b_params.kernel_size,
                                       stride=b_params.stride, size=params.size)

        self.bn1 = nn.BatchNorm2d(num_features=cout, eps=e_params.batch_norm_epsilon,
                                  momentum=e_params.batch_norm_momentum)

        num_squeezed_channels = max(1, int(b_params.input_filters * b_params.se_ratio))
        self.conv2 = SamePaddingConv2D(cin=cout, cout=num_squeezed_channels, size=1, kernel_size=1)
        self.conv3 = SamePaddingConv2D(cin=num_squeezed_channels, cout=cout, size=1, kernel_size=1)

        self.conv4 = SamePaddingConv2D(cin=cout, cout=b_params.output_filters, kernel_size=1,
                                       size=int(np.ceil(params.size / b_params.stride)))
        self.bn2 = nn.BatchNorm2d(num_features=b_params.output_filters,
                                  eps=e_params.batch_norm_epsilon,
                                  momentum=e_params.batch_norm_momentum)

    def forward(self, x):
        x0 = x
        if self.block_params.expand_ratio > 1:
            x = swish(self.bn0(self.conv0(x)))

        x = swish(self.bn1(self.conv1(x)))

        x_squeezed = self.conv3(swish(self.conv2(F.adaptive_avg_pool2d(x, (1, 1)))))
        x = torch.sigmoid(x_squeezed) * x

        x = self.bn2(self.conv4(x))

        if self.block_params.id_skip and self.block_params.stride == 1 \
                and self.block_params.input_filters == self.block_params.output_filters:
            x = self.drop_connect(x)
            x = x + x0
        return x


class EfficientNet(nn.Module):
    def __init__(self, params: ParamsHW2Classification,
                 e_params: EfficientNetParams,
                 basic_block_args):
        super().__init__()
        self.params = params
        self.e_params = e_params

        image_size = params.size
        self.conv0 = SamePaddingConv2D(size=image_size, cin=e_params.cin,
                                       cout=e_params.cout_conv0,
                                       kernel_size=3, stride=2)
        self.bn0 = nn.BatchNorm2d(num_features=e_params.cout_conv0,
                                  momentum=e_params.batch_norm_momentum,
                                  eps=e_params.batch_norm_epsilon)
        image_size = image_size // 2

        blocks: List[MBConvBlock] = []
        for block_args in basic_block_args:
            block_args = MBBlockParams(**asdict(block_args))
            block_args.input_filters = round_filter(block_args.input_filters,
                                                    e_params.width_coefficients,
                                                    e_params.depth_divisor)
            block_args.output_filters = round_filter(block_args.output_filters,
                                                     e_params.width_coefficients,
                                                     e_params.depth_divisor)
            block_args.num_repeat = round_depth(block_args.num_repeat,
                                                e_params.depth_coefficients)

            blocks.append(MBConvBlock(params, e_params, block_args))
            image_size = int(np.ceil(image_size / block_args.stride))
            if block_args.num_repeat > 1:
                block_args = MBBlockParams(**asdict(block_args))
                block_args.stride = 1
                block_args.input_filters = block_args.output_filters
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(params, e_params, block_args))

        n = len(blocks)
        for i in range(n):
            blocks[i].dropout_connect_rate = self.e_params.drop_connect_rate * i / n

        self.MBBlocks = nn.Sequential(*blocks)

        self.conv1 = SamePaddingConv2D(size=image_size, cin=448, cout=e_params.cout_conv1,
                                       kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=e_params.cout_conv1,
                                  momentum=e_params.batch_norm_momentum,
                                  eps=e_params.batch_norm_epsilon)
        self.fc = nn.Linear(e_params.cout_conv1, params.output_channels)

    def forward(self, x: torch.Tensor):
        x = swish(self.bn0(self.conv0(x)))
        x = self.MBBlocks(x)
        x = swish(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.fc(F.dropout(torch.flatten(x, start_dim=1), self.params.dropout))


class SamePaddingConv2D(nn.Module):
    """
    Why PyTorch still hasn't implement this officially?
    """

    def __init__(self, size, cin, cout, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super(SamePaddingConv2D, self).__init__()
        assert kernel_size % 2 == 1  # use odd filter
        eq_filter_size = (kernel_size - 1) * dilation + 1
        out_size = (size + stride - 1) // stride
        padding = max(0, (out_size - 1) * stride + eq_filter_size - size)

        self.layer = nn.Conv2d(cin, cout, kernel_size, stride, padding // 2, dilation, groups, bias)

    def forward(self, x):
        return self.layer(x)
