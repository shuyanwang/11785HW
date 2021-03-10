import torch
import torch.nn as nn
from torch import Tensor

from utils.base import Model
from torchvision.models.resnet import BasicBlock
from hw2_classification import ParamsHW2Classification
from typing import Union, Type, Optional, Callable


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PerceptronLayer(nn.Module):
    # Added dropout
    def __init__(self, c_in, c_out, dropout=0.5):
        super(PerceptronLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out),
                                   nn.BatchNorm1d(c_out), nn.Dropout(dropout), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class MLP19(Model):
    def __init__(self, params):
        super().__init__(params)
        self.l1 = PerceptronLayer(self.params.input_dims[0], 4096, params.dropout)
        self.l2 = PerceptronLayer(4096, 4096, params.dropout)
        self.l3 = PerceptronLayer(4096, 4096, params.dropout)
        self.l4 = PerceptronLayer(4096, 4096, params.dropout)
        self.l5 = PerceptronLayer(4096, 4096, params.dropout)
        self.l6 = PerceptronLayer(4096, 4096, params.dropout)
        self.l7 = PerceptronLayer(4096, 4096, params.dropout)

        self.l8 = nn.Linear(4096, self.params.output_channels)

    def forward(self, x: torch.Tensor):
        x1 = self.l1(torch.flatten(x, start_dim=1))
        x3 = self.l3(x1 + self.l2(x1))
        x5 = self.l5(x3 + self.l4(x3))
        x7 = self.l7(x5 + self.l6(x5))
        return self.l8(x7)


class ResNet(nn.Module):
    """
    Note1:
        The architecture is from https://arxiv.org/abs/1512.03385 (ResNet) and the
        pytorch release at https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

        The model below is used to generate the best results submitted;
        it is a simplified version of the official ResNet and modified to adapt to HW2

    Note2:
    As per course policy at https://piazza.com/class/khtqzctrciu1fp?cid=71,
    I looked at some public resources, including:
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
    https://github.com/KaimingHe/deep-residual-networks
    """

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers, embedding=False,
                 num_classes=4000):
        super().__init__()
        self.embedding = embedding

        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cin = 64
        # The number of channels feeding into the next residual layer
        # subject to update in residual layers: the proceeding layer takes this value
        residual_layers = [self.residual_layer(block, 64, layers[0], 1),
                           self.residual_layer(block, 128, layers[1], 2),
                           self.residual_layer(block, 256, layers[2], 2),
                           self.residual_layer(block, 512, layers[3], 2)]

        self.residual_layers = nn.ModuleList(residual_layers)  # Register the layers

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        if not embedding:
            self.linear = nn.Linear(block.expansion * 512, num_classes)
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

    def residual_layer(self, block, cout, num_blocks, stride):

        #### For bottleneck blocks, we need to down-sample
        # because we need cin to match cout for addition
        down_sample = None
        if stride > 1 or self.cin != cout * block.expansion:
            down_sample = nn.Sequential(
                    nn.Conv2d(self.cin, cout * block.expansion, kernel_size=1, stride=stride,
                              bias=False), nn.BatchNorm2d(cout * block.expansion))

        layers = [block(self.cin, cout, stride, down_sample)]
        self.cin = cout * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.cin, cout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return: If embedding, return the flattened feature vector
        """
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        x = torch.flatten(self.pool2(x), 1)

        if self.embedding:
            return x

        return self.linear(x)


class ResNet101(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(Bottleneck, [3, 4, 23, 3])

    def forward(self, x):
        return self.net(x)


class ResNet34(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        return self.net(x)


class ResNet18(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        return self.net(x)


class ResNet10(Model):
    def __init__(self, params: ParamsHW2Classification):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1])

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet152(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(Bottleneck, [3, 8, 36, 3])

    def forward(self, x):
        return self.net(x)


class ResNet101E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], embedding=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet18E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], embedding=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet34E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [3, 4, 6, 3], embedding=True)

    def forward(self, x):
        return self.net(x)


class ResNet10E(Model):
    def __init__(self, params: ParamsHW2Classification):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet10_256(Model):
    def __init__(self, params):
        super(ResNet10_256, self).__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=True, num_classes=256)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet10_2048(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=True, num_classes=2048)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet10_1024(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=True, num_classes=1024)

    def forward(self, x: torch.Tensor):
        return self.net(x)
