import torch
import torch.nn as nn
from utils.base import Model
from hw2_classification import ParamsHW2Classification


#### Note: The architecture is from https://arxiv.org/abs/1512.03385 (ResNet)
#### Note2: As per course policy at https://piazza.com/class/khtqzctrciu1fp?cid=71,
# I looked at some public resources, including:
# https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
# https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
# Note3: The code below is composed by myself.

class ResidualBlock(nn.Module):
    def __init__(self, c_width,down_sample):
        """
        Bottleneck block.
        :param c_width:
        :param down_sample: for handling 256->
        """
        super(ResidualBlock, self).__init__()
        c_in = c_width * 4

        self.c1 = nn.Conv2d(c_in, c_width, kernel_size=1, bias=False),
        self.b1 = nn.BatchNorm2d(c_width)
        self.c2 = nn.Conv2d(c_width, c_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c_width)
        self.c3 = nn.Conv2d(c_width, c_in, kernel_size=1, bias=False),
        self.b3 = nn.BatchNorm2d(c_in)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.fu
