import torch.nn as nn
import torch.nn.functional as functional
import torch


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
