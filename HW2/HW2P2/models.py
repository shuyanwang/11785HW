import torch
import torch.nn as nn
from utils.base import Model
from hw2_classification import ParamsHW2Classification
from typing import Union, Type
import torch.nn.functional as F


class BasicBlock(nn.Module):
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


class BottleNeckBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, down_sample=False):
        super().__init__()
        # BottleNeckBlock: 3-layered: cin->4*cout
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)

        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)

        self.conv3 = nn.Conv2d(cout, cout * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout * 4)

        self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout * 4)) if down_sample else None

    def forward(self, x: torch.Tensor):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        down_sampled = self.downsample(x) if self.downsample is not None else x

        return torch.relu(out + down_sampled)


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
        pytorch doc at https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
        It is a simplified version of the ResNet and it adapts to HW2

    Note2:
        As per course policy at https://piazza.com/class/khtqzctrciu1fp?cid=71,
        I looked at some public resources, including:
        https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
        https://github.com/KaimingHe/deep-residual-networks
    """

    def __init__(self, block: Type[Union[BasicBlock, BottleNeckBlock]], layers, embedding=False,
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
            self.linear = nn.Linear((4 if block.__name__ == 'BottleNeckBlock' else 1) * 512,
                                    num_classes)
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

        if num_blocks == 0:
            return nn.Identity()

        #### For BottleNeckBlock blocks, we need to down-sample
        # because we need cin to match cout for addition

        expansion = 4 if block.__name__ == 'BottleNeckBlock' else 1

        layers = [block(self.cin, cout, stride, stride > 1 or self.cin != cout * expansion)]
        self.cin = cout * expansion
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
            return F.normalize(x)

        return self.linear(x)

    def features_no_pool(self, x):
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        x = torch.flatten(x, 1)
        return x


class ResNet101(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BottleNeckBlock, [3, 4, 23, 3])

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
        self.net = ResNet(BottleNeckBlock, [3, 8, 36, 3])

    def forward(self, x):
        return self.net(x)


class ResNet101E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(block=BottleNeckBlock, layers=[3, 4, 23, 3], embedding=True)

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


class ResNet8E(Model):
    def __init__(self, params: ParamsHW2Classification):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 0], embedding=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet10_256(Model):
    def __init__(self, params):
        super(ResNet10_256, self).__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=False, num_classes=256)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet10C(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=False,
                          num_classes=params.feature_dims)
        self.fc = nn.Linear(params.feature_dims, params.output_channels)

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        return features, self.fc(torch.relu(features))


class ResNet10EN(Model):
    def __init__(self, params: ParamsHW2Classification):
        super().__init__(params)
        self.net = ResNet(BasicBlock, [1, 1, 1, 1], embedding=True)

    def forward(self, x: torch.Tensor):
        return self.net.features_no_pool(x)


class ResNetK3(nn.Module):
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

    def __init__(self, block: Type[Union[BasicBlock, BottleNeckBlock]], layers, embedding=False,
                 num_classes=4000):
        super().__init__()
        self.embedding = embedding

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
            self.linear = nn.Linear((4 if block.__name__ == 'BottleNeckBlock' else 1) * 512,
                                    num_classes)
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

        if num_blocks == 0:
            return nn.Identity()

        #### For BottleNeckBlock blocks, we need to down-sample
        # because we need cin to match cout for addition

        expansion = 4 if block.__name__ == 'BottleNeckBlock' else 1

        layers = [block(self.cin, cout, stride, stride > 1 or self.cin != cout * expansion)]
        self.cin = cout * expansion
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
            return F.relu(x)

        return self.linear(x)

    def features_no_pool(self, x):
        x = self.pool1(torch.relu(self.bn(self.conv(x))))
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        x = torch.flatten(x, 1)
        return x


class ResNet34K3C(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetK3(BasicBlock, [3, 4, 6, 3], embedding=True,
                            num_classes=params.feature_dims)
        self.fc = nn.Linear(params.feature_dims, params.output_channels)

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        return features, self.fc(torch.relu(features))


class ResNet34K3RC(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetK3(BasicBlock, [3, 4, 6, 3], embedding=True,
                            num_classes=params.feature_dims)
        self.fc = nn.Linear(params.feature_dims, params.output_channels)

    def forward(self, x: torch.Tensor):
        features = torch.relu(self.net(x))
        return features, self.fc(features)
