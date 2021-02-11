import torch.nn as nn


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
