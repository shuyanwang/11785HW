import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co

data_dir = 'E:/11785_data/HW1'


class HyperParameters:
    context_K = 11
    batch_size = 12
    lr = 1e-3


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

    def forward(self):
        pass


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, X, Y, context_K=HyperParameters.context_K):
        super(TrainSet, self).__init__()
        self.K = context_K
        pad_size = (context_K, 40)
        self.X = []
        self.Y = []

        for x, y in zip(X, Y):
            x_padded = torch.cat([torch.zeros(pad_size), x, torch.zeros(pad_size)])  # (K+?+K)*40
            for i in range(y.shape[0]):
                self.X.append(x_padded[i:i + 2 * context_K + 1])
                self.Y.append(y[i])

        self.length = len(self.Y)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.length

    # @staticmethod
    # def collate_fn(batch):
    #     pass


class TestSet(torch.utils.data.Dataset):
    def __init__(self, X, context_K=HyperParameters.context_K):
        super(TestSet, self).__init__()
        self.K = context_K
        pad_size = (context_K, 40)
        self.X = []

        for x in X:
            x_padded = torch.cat([torch.zeros(pad_size), x, torch.zeros(pad_size)])  # (K+?+K)*40
            for i in range(x.shape[0]):
                self.X.append(x_padded[i:i + 2 * context_K + 1])

        self.length = len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index]

    def __len__(self):
        return self.length


def main():
    train_X = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle=True)  # N*?*40
    train_Y = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)  # N*?

    valid_X = np.load(os.path.join(data_dir, 'dev.npy'), allow_pickle=True)  # N*?*40
    valid_Y = np.load(os.path.join(data_dir, 'dev_labels.npy'), allow_pickle=True)  # N*?

    test_X = np.load(os.path.join(data_dir, 'test.npy'), allow_pickle=True)

    train_loader = torch.utils.data.DataLoader(
        TrainSet(train_X, train_Y), batch_size=HyperParameters.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        TrainSet(valid_X, valid_Y), batch_size=HyperParameters.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        TestSet(test_X), batch_size=HyperParameters.batch_size, shuffle=False)


if __name__ == '__main__':
    main()
