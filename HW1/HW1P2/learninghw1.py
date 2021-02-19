import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co
from utils.base import Params, Learning
from tqdm import tqdm

num_workers = 6


class ParamsHW1(Params):
    def __init__(self, K=15, B=32768, lr=1e-3, max_epoch=201, is_double=False,
                 data_dir='E:/11785_data/HW1'):
        super(ParamsHW1, self).__init__(B=B, lr=lr, max_epoch=max_epoch,
                                        data_dir=data_dir, is_double=is_double)
        self.K = K
        self.str = 'k=' + str(self.K) + 'b=' + str(self.B) + 'lr=' + str(
            self.lr) + ('_double_' if is_double else '_float_')

    def __str__(self):
        return self.str


class DatasetHW1(torch.utils.data.Dataset):
    def __init__(self, X_dir, Y_dir, context_K, data_type=torch.float):
        super(DatasetHW1, self).__init__()
        X = np.load(X_dir, allow_pickle=True)
        self.test = Y_dir is None
        self.N = X.shape[0]
        self.K = context_K
        self.look_up = []
        pad_size = (self.K, 40)
        self.X = []  # N,[(K+?+K),40]
        for u_id, x in enumerate(X):
            self.X.append(torch.cat([torch.zeros(pad_size), torch.as_tensor(x, dtype=data_type),
                                     torch.zeros(pad_size)], dim=0))
            for frame_id in range(x.shape[0]):
                self.look_up.append((u_id, frame_id))
        self.X = np.asarray(self.X, dtype=object)

        if not self.test:
            Y = np.load(Y_dir, allow_pickle=True)
            self.Y = [torch.as_tensor(y, dtype=torch.long) for y in Y]
            self.Y = np.asarray(self.Y, dtype=object)
        print(X_dir, self.__len__())

    def __getitem__(self, index) -> T_co:
        u_id, frame_id = self.look_up[index]
        if self.test:
            return torch.flatten(self.X[u_id][frame_id:self.K * 2 + frame_id + 1])
        else:
            return (torch.flatten(self.X[u_id][frame_id:self.K * 2 + frame_id + 1]),
                    self.Y[u_id][frame_id])

    def __len__(self):
        return len(self.look_up)


class LearningHW1(Learning):
    def __init__(self, params: ParamsHW1, model):
        super(LearningHW1, self).__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)
        print(str(self))

        self.train_X = os.path.join(params.data_dir, 'train.npy')  # N,?,40
        self.train_Y = os.path.join(params.data_dir, 'train_labels.npy')  # N,?
        self.valid_X = os.path.join(params.data_dir, 'dev.npy')  # N,?,40
        self.valid_Y = os.path.join(params.data_dir, 'dev_labels.npy')
        self.test_X = os.path.join(params.data_dir, 'test.npy')

        self.dtype = torch.double if params.is_double else torch.float

    def _load_train(self):
        self.train_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.train_X, self.train_Y, self.params.K, self.dtype),
            batch_size=self.params.B,
            shuffle=True, pin_memory=True, num_workers=num_workers)

    def _load_valid(self):
        self.valid_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.valid_X, self.valid_Y, self.params.K, self.dtype),
            batch_size=self.params.B,
            shuffle=False, pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        self.test_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.test_X, None, self.params.K, self.dtype), batch_size=1, shuffle=False,
            pin_memory=True, num_workers=num_workers)

    def test(self):
        if self.test_loader is None:
            self._load_test()
        # print('testing...')
        with open('results/' + str(self) + '.csv', 'w') as f:
            f.write('id,label')
            with torch.cuda.device(self.device):
                with torch.no_grad():
                    self.model.eval()
                    for i, item in enumerate(tqdm(self.test_loader)):
                        x = item.to(self.device)
                        label = torch.argmax(self.model(x), dim=1).item()
                        f.write('\n' + str(i) + ',' + str(label))
