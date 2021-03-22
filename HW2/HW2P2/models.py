from typing import List

import numpy as np
from utils.base import *
from hw2_classification import ParamsHW2Classification
import torch.nn.functional as F
from dataclasses import asdict


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
    """
    Note1:
        The architecture is from https://arxiv.org/abs/1512.03385 (ResNet) and the
        pytorch doc at https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
        It is a simplified version of the ResNet and it adapts to HW2

    Note2:
        As per course policy at https://piazza.com/class/khtqzctrciu1fp?cid=71,
        I looked at some public resources, including:
        https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
        https://github.com/KaimingHe/deep-residual-networks
    """

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
    kernel_size: int = field(default=1)
    stride: int = field(default=1)
    expand_ratio: int = field(default=1)
    input_filters: int = field(default=1)
    output_filters: int = field(default=1)
    se_ratio: float = field(default=-1.0)
    id_skip: bool = field(default=False)


def Swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


def roundFilter(filters, width_coefficients, depth_divisor: int):
    filters = int(filters * width_coefficients)
    return np.max(depth_divisor,
                  (int(filters + depth_divisor / 2) // depth_divisor) * depth_divisor)


def roundDepth(repeats, depth_coefficients):
    return np.ceil(repeats * depth_coefficients)


class EfficientNetB4(Model):
    """
    Caller shell for Model API
    """

    def __init__(self, params):
        super().__init__(params)
        eParams = EfficientNetParams()
        self.net = EfficientNet(params, eParams)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MBConvBlock(nn.Module):
    def dropConnect(self, x):
        if not self.training:
            return x

        distribution = np.random.binomial(p=1 - self.dropout_connect_rate,
                                          size=(x.shape[0], 1, 1, 1), n=1)
        distribution_tensor = torch.as_tensor(distribution, device=self.device)
        return x / (1 - self.dropout_connect_rate) * distribution_tensor

    def __init__(self, params: ParamsHW2Classification, efficientNetParams: EfficientNetParams,
                 blockParams: MBBlockParams):
        super().__init__()
        self.device = params.device
        self._block_args = blockParams
        self._bn_mom = efficientNetParams.batch_norm_momentum
        self._bn_eps = efficientNetParams.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
                0 < self._block_args.se_ratio <= 1)
        self.id_skip = blockParams.id_skip

        cin = self._block_args.input_filters
        cout = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            self._expand_conv = SamePaddingConv2D(cin=cin, cout=cout, kernel_size=1,
                                                  size=params.size)
            self._bn0 = nn.BatchNorm2d(num_features=cout, momentum=self._bn_mom, eps=self._bn_eps)

        self._depthwise_conv = SamePaddingConv2D(cin=cout, cout=cout, groups=cout,
                                                 kernel_size=self._block_args.kernel_size,
                                                 stride=self._block_args.stride, size=params.size)
        self._bn1 = nn.BatchNorm2d(num_features=cout, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = int(np.ceil(params.size / self._block_args.stride))

        if self.has_se:
            num_squeezed_channels = max(1, int(
                    self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = SamePaddingConv2D(cin=cout, cout=num_squeezed_channels, size=1,
                                                kernel_size=1)
            self._se_expand = SamePaddingConv2D(cin=num_squeezed_channels, cout=cout, size=1,
                                                kernel_size=1)

        final_cout = self._block_args.output_filters
        self._project_conv = SamePaddingConv2D(cin=cout, cout=final_cout, kernel_size=1,
                                               size=image_size)
        self._bn2 = nn.BatchNorm2d(num_features=final_cout, momentum=self._bn_mom, eps=self._bn_eps)
        self.dropout_connect_rate = 0

    def forward(self, x):
        x0 = x
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = Swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = Swish(x)

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, (1, 1))
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = Swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        if self.id_skip and self._block_args.stride == 1 and self._block_args.input_filters == \
                self._block_args.output_filters:
            x = self.dropConnect(x)
            x = x + x0
        return x


class EfficientNet(nn.Module):
    """
    The architecture is from https://arxiv.org/abs/1905.11946
    As per course policy @ https://piazza.com/class/khtqzctrciu1fp?cid=71, I looked at open
    source code at https://github.com/lukemelas/EfficientNet-PyTorch. However, I learned the
    algorithm flow and wrote my own code at the end, including adaptation to HW2 and changes in data
    structures, as well as other implementation techniques. For example, dropout should be written
    as a method in order to better utilize nn.Module's properties.
    """

    def __init__(self, params: ParamsHW2Classification,
                 efficientNetParams: EfficientNetParams):
        super().__init__()
        self.params = params
        self.efficientNetParams = efficientNetParams

        image_size = params.size
        self.conv0 = SamePaddingConv2D(size=image_size, cin=efficientNetParams.cin,
                                       cout=efficientNetParams.cout_conv0,
                                       kernel_size=3, stride=2)
        self.bn0 = nn.BatchNorm2d(num_features=efficientNetParams.cout_conv0,
                                  momentum=efficientNetParams.batch_norm_momentum,
                                  eps=efficientNetParams.batch_norm_epsilon)
        image_size = image_size // 2

        blocks: List[MBConvBlock] = []
        blocks_args = []
        for block_args in blocks_args:
            block_args = MBBlockParams(**asdict(block_args))
            block_args.input_filters = roundFilter(block_args.input_filters,
                                                   efficientNetParams.width_coefficients,
                                                   efficientNetParams.depth_divisor)
            block_args.output_filters = roundFilter(block_args.output_filters,
                                                    efficientNetParams.width_coefficients,
                                                    efficientNetParams.depth_divisor)
            block_args.num_repeat = roundDepth(block_args.num_repeat,
                                               efficientNetParams.depth_coefficients)

            blocks.append(MBConvBlock(params, efficientNetParams, block_args))
            image_size = int(np.ceil(image_size / block_args.stride))
            if block_args.num_repeat > 1:
                block_args = MBBlockParams(**asdict(block_args))
                block_args.stride = 1
                block_args.input_filters = block_args.output_filters
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(params, efficientNetParams, block_args))

        N = len(blocks)
        for i in range(N):
            blocks[i].dropout_connect_rate = self.efficientNetParams.drop_connect_rate * i / N

        self.MBBlocks = nn.Sequential(*blocks)

        self.conv1 = SamePaddingConv2D(size=image_size, cin=448, cout=efficientNetParams.cout_conv1,
                                       kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=efficientNetParams.cout_conv1,
                                  momentum=efficientNetParams.batch_norm_momentum,
                                  eps=efficientNetParams.batch_norm_epsilon)
        self.fc = nn.Linear(efficientNetParams.cout_conv1, params.output_channels)

    def forward(self, x: torch.Tensor):
        x = Swish(self.bn0(self.conv0(x)))
        x = self.MBBlocks(x)
        x = Swish(self.bn1(self.conv1(x)))
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
