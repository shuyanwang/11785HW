import torch
import torch.nn as nn
from utils.base import Model
from torchvision.models import resnet, vgg, squeezenet, densenet
from hw2_classification import ParamsHW2Classification


class PerceptronLayer(nn.Module):
    # Added dropout
    def __init__(self, c_in, c_out, dropout=0.5):
        super(PerceptronLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out),
                                   nn.BatchNorm1d(c_out), nn.Dropout(dropout), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class ResNet101(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3],
                                 num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


class ResNet34(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3],
                                 num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


class ResNet18(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2],
                                 num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


class VGG11BN(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = vgg.VGG(
                vgg.make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                                True), num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


class SqueezeNet(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = squeezenet.SqueezeNet(version='1_1', num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


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


class ResNetEmbedding(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout=0.5):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.d1 = nn.Dropout(dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.d2 = nn.Dropout(dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.d3 = nn.Dropout(dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an
        # identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.d1(x)
        x = self.layer2(x)
        x = self.d2(x)
        x = self.layer3(x)
        x = self.d3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class DenseNet121(Model):
    def __init__(self, params):
        super(DenseNet121, self).__init__(params)
        self.net = densenet.DenseNet(num_classes=self.params.output_channels,
                                     drop_rate=self.params.dropout)

    def forward(self, x):
        return self.net(x)


class ResNet10(Model):
    def __init__(self, params: ParamsHW2Classification):
        super(ResNet10, self).__init__(params)
        self.net = resnet.ResNet(resnet.BasicBlock, [1, 1, 1, 1], self.params.output_channels)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet152(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3],
                                 num_classes=self.params.output_channels)

    def forward(self, x):
        return self.net(x)


class WideResNet101(Model):
    def __init__(self, params):
        super(WideResNet101, self).__init__(params)
        self.net = resnet.ResNet(width_per_group=64 * 2, num_classes=4000, block=resnet.Bottleneck,
                                 layers=[3, 4, 23, 3])

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet101E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetEmbedding(num_classes=4000,
                                   block=resnet.Bottleneck, layers=[3, 4, 23, 3])

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNet18E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = ResNetEmbedding(num_classes=4000,
                                   block=resnet.BasicBlock, layers=[2, 2, 2, 2])

    def forward(self, x: torch.Tensor):
        return self.net(x)
