import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils.base import Model
from typing import List


class MLP1(nn.Module):
    def __init__(self, K):
        super(MLP1, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(40 * (2 * K + 1), 1024),
                                 nn.BatchNorm1d(1024), nn.ReLU(),
                                 nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                 nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 71))

    def forward(self, x):
        return self.mlp(x)


class MLP2(nn.Module):
    def __init__(self, K):
        super(MLP2, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(40 * (2 * K + 1), 1024),
                                 nn.BatchNorm1d(1024), nn.ReLU(),
                                 nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 71))

    def forward(self, x):
        return self.mlp(x)


class MLP3(nn.Module):
    def __init__(self, K):
        super(MLP3, self).__init__()
        self.l1 = nn.Linear(40 * (2 * K + 1), 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(768, 256)
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256, 71)

        self.b1 = nn.BatchNorm1d(512)
        self.b2 = nn.BatchNorm1d(256)
        self.b3 = nn.BatchNorm1d(256)
        self.b4 = nn.BatchNorm1d(256)

    def forward(self, x):
        x1 = functional.relu(self.b1(self.l1(x)))  # B,512
        x2 = functional.relu(self.b2(self.l2(x1)))  # B,256
        x3 = functional.relu(self.b3(self.l3(torch.cat([x1, x2], dim=1))))  # B,256
        x4 = functional.relu(self.b4(self.l4(torch.cat([x2, x3], dim=1))))  # B,256
        return self.l5(x4)


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
    # Added dropout
    def __init__(self, c_in, c_out):
        super(PerceptronLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_features=c_in, out_features=c_out),
                                   nn.BatchNorm1d(c_out), nn.Dropout(), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class MLP5(nn.Module):
    def __init__(self, K):
        super(MLP5, self).__init__()
        self.mlp = nn.Sequential(PerceptronLayer(40 * (2 * K + 1), 1024),
                                 PerceptronLayer(1024, 1024),
                                 # PerceptronLayer(1024, 1024),
                                 PerceptronLayer(1024, 512),
                                 PerceptronLayer(512, 256),
                                 PerceptronLayer(256, 71))

    def forward(self, x):
        return self.mlp(x)


class MLP6(nn.Module):
    def __init__(self, K):
        super(MLP6, self).__init__()
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 1024)
        self.l2 = PerceptronLayer(1024, 1024)
        self.l3 = PerceptronLayer(1024, 1024)
        self.l4 = PerceptronLayer(1024, 512)
        self.l5 = PerceptronLayer(512, 256)
        self.l6 = PerceptronLayer(256, 71)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l1 + l2)
        l4 = self.l4(l2 + l3)
        return self.l6(self.l5(l4))


class MLP7(nn.Module):
    def __init__(self, K):
        super(MLP7, self).__init__()
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 1024)
        self.l2 = PerceptronLayer(1024, 1024)
        self.l3 = PerceptronLayer(1024, 1024)
        self.l4 = PerceptronLayer(1024, 1024)
        self.l5 = PerceptronLayer(1024, 1024)
        self.l6 = PerceptronLayer(1024, 512)
        self.l7 = PerceptronLayer(512, 256)
        self.l8 = PerceptronLayer(256, 71)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l1 + l2)
        l4 = self.l4(l2 + l3)
        l5 = self.l5(l3 + l4)
        l6 = self.l6(l4 + l5)
        return self.l8(self.l7(l6))


class MLPSkipConnections(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(MLPSkipConnections, self).__init__()
        self.layers = [PerceptronLayer(in_channels, in_channels) for _ in range(1, num_layers)]
        self.layers.append(PerceptronLayer(in_channels, out_channels))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x1 = self.layers[0](x)
        for i in range(1, len(self.layers)):
            in_vector = x + x1
            x = x1
            x1 = self.layers[i](in_vector)
        return x1


class MLP8(nn.Module):
    def __init__(self, K):
        super(MLP8, self).__init__()
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 1024)
        self.skip = MLPSkipConnections(1024, 512, 9)
        self.l2 = PerceptronLayer(512, 256)
        self.l3 = PerceptronLayer(256, 71)

    def forward(self, x):
        l1 = self.l1(x)
        skip = self.skip(l1)
        return self.l3(self.l2(skip))


class MLP9(nn.Module):
    def __init__(self, K):
        super(MLP9, self).__init__()
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 2048)
        self.skip = MLPSkipConnections(2048, 1024, 5)
        self.l2 = PerceptronLayer(1024, 256)
        self.l3 = PerceptronLayer(256, 71)

    def forward(self, x):
        l1 = self.l1(x)
        skip = self.skip(l1)
        return self.l3(self.l2(skip))


# consider using broader networks with a smaller B

#### From MLP10, models should inherit Model (in base.py), not nn.Module
# This is for visualizing graphs
# To train/validate/test MLP1-9, change the base class and add the input-dims property
# Performance should not be affected, as input_dims are only used for creating a dummy input

class MLP10(Model):
    @property
    def input_dims(self):
        return [40 * (2 * self.K + 1)]  # Batch dim not included

    def __init__(self, K):
        super(MLP10, self).__init__()
        self.K = K
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 4096)
        self.skip = MLPSkipConnections(4096, 2048, 5)
        self.l2 = PerceptronLayer(2048, 1024)
        self.l3 = PerceptronLayer(1024, 256)
        self.l4 = PerceptronLayer(256, 71)

    def forward(self, x):
        return self.l4(self.l3(self.l2(self.skip(self.l1(x)))))


class MLP11(Model):
    @property
    def input_dims(self) -> List:
        return [40 * (2 * self.K + 1)]

    def __init__(self, K):
        super(MLP11, self).__init__()
        self.K = K
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 4096)
        self.l2 = PerceptronLayer(4096, 4096)
        self.l3 = PerceptronLayer(4096, 4096)
        self.l4 = PerceptronLayer(4096, 4096)
        self.l5 = PerceptronLayer(4096, 4096)

        self.classifier = nn.Sequential(PerceptronLayer(4096, 2048),
                                        PerceptronLayer(2048, 1024),
                                        PerceptronLayer(1024, 256),
                                        PerceptronLayer(256, 71))

    def forward(self, x):
        x1 = self.l1(x)
        x3 = self.l3(x1 + self.l2(x1))
        x5 = self.l5(x3 + self.l4(x3))
        return self.classifier(x5)


class MLP12(Model):
    @property
    def input_dims(self) -> List:
        return [40 * (2 * self.K + 1)]

    def __init__(self, K):
        super().__init__()
        self.K = K
        self.l1 = PerceptronLayer(40 * (2 * K + 1), 4096)
        self.l2 = PerceptronLayer(4096, 4096)
        self.l3 = PerceptronLayer(4096, 4096)
        self.l4 = PerceptronLayer(4096, 4096)
        self.l5 = PerceptronLayer(4096, 4096)
        self.l6 = PerceptronLayer(4096, 4096)
        self.l7 = PerceptronLayer(4096, 4096)

        self.l8 = PerceptronLayer(4096, 71)

    def forward(self, x):
        x1 = self.l1(x)
        x3 = self.l3(x1 + self.l2(x1))
        x5 = self.l5(x3 + self.l4(x3))
        x7 = self.l7(x5 + self.l6(x5))
        return self.l8(x7)
