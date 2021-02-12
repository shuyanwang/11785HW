import torch.nn as nn
import torch.nn.functional as functional
import torch


class MLP4(nn.Module):
    def __init__(self, K):
        super(MLP4, self).__init__()
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 512)
        self.l2 = PerceptronLayer(512, 256)
        self.l3 = PerceptronLayer(768, 256)
        self.l4 = PerceptronLayer(512, 256)
        self.l5 = PerceptronLayer(768, 256)
        self.l6 = PerceptronLayer(768, 256)
        self.l7 = PerceptronLayer(768, 256)
        self.out = nn.Linear(256, 71)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(torch.cat([x1, x2], dim=1))
        x4 = self.l4(torch.cat([x2, x3], dim=1))
        x5 = self.l5(torch.cat([x2, x3, x4], dim=1))
        x6 = self.l6(torch.cat([x3, x4, x5], dim=1))
        x7 = self.l5(torch.cat([x4, x5, x6], dim=1))
        return self.out(x7)


class PerceptronLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(PerceptronLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out),
                                   nn.BatchNorm1d(c_out), nn.Dropout(), nn.ReLU())

    def forward(self, x):
        return self.layer(x)
