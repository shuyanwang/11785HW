import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co
from utils.base import Params, Learning

num_workers = 6


class ParamsHW1(Params):
    def __init__(self, K, B, lr=1e-4, max_epoch=41, is_double=False):
        super(ParamsHW1, self).__init__(B=B, lr=lr, max_epoch=max_epoch,
                                        data_dir='E:/11785_data/HW1', is_double=is_double)
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
    def __init__(self, params: ParamsHW1, model: nn.Module):
        super(LearningHW1, self).__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)
        print(str(self))

        self.train_X = os.path.join(params.data_dir, 'train.npy')  # N,?,40
        self.train_Y = os.path.join(params.data_dir, 'train_labels.npy')  # N,?
        self.valid_X = os.path.join(params.data_dir, 'dev.npy')  # N,?,40
        self.valid_Y = os.path.join(params.data_dir, 'dev_labels.npy')
        self.test_X = os.path.join(params.data_dir, 'test.npy')

    def load_train(self):
        self.train_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.train_X, self.train_Y, self.params.K), batch_size=self.params.B,
            shuffle=True, pin_memory=True, num_workers=num_workers)

    def load_valid(self):
        self.valid_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.valid_X, self.valid_Y, self.params.K), batch_size=self.params.B,
            shuffle=False, pin_memory=True, num_workers=num_workers)

    def load_test(self):
        self.test_loader = torch.utils.data.DataLoader(
            DatasetHW1(self.test_X, None, self.params.K), batch_size=1, shuffle=False,
            pin_memory=True, num_workers=num_workers)

    def load_model(self, epoch=5):
        loaded = torch.load('checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar')
        self.init_epoch = loaded['epoch']
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])

    def save_model(self, epoch, loss_item):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_item,
        }, 'checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar')

    def test(self):
        assert self.test_loader is not None
        print('testing...')
        with open('results/' + str(self) + '.csv', 'w') as f:
            f.write('id,label')
            with torch.cuda.device(self.device):
                with torch.no_grad():
                    self.model.eval()
                    for i, item in enumerate(self.test_loader):
                        x = item.to(self.device)
                        label = torch.argmax(self.model(x), dim=1).item()
                        f.write('\n' + str(i) + ',' + str(label))
