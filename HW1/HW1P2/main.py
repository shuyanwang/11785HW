import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field

data_dir = 'E:/11785_data/HW1'


@dataclass
class HyperParameters:
    K: int = field()
    B: int = field()
    lr: float = field(default=1e-3)
    max_epoch: int = field(default=16)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_dir, Y_dir, context_K):
        super(Dataset, self).__init__()
        X = np.load(X_dir, allow_pickle=True)
        self.test = Y_dir is None
        self.N = X.shape[0]
        self.K = context_K
        self.Y = np.load(Y_dir, allow_pickle=True) if not self.test else None  # N,?
        self.look_up = []
        pad_size = (self.K, 40)
        self.X = []  # N,[(K+?+K),40]
        for u_id, x in enumerate(X):
            self.X.append(torch.cat([torch.zeros(pad_size),
                                     torch.from_numpy(x), torch.zeros(pad_size)], dim=0))
            for frame_id in range(x.shape[0]):
                self.look_up.append((u_id, frame_id))
        print(X_dir, self.__len__())

    def __getitem__(self, index) -> T_co:
        u_id, frame_id = self.look_up[index]
        if self.test:
            return torch.flatten(self.X[u_id][frame_id:self.K * 2 + frame_id + 1])
        else:
            return (torch.flatten(self.X[u_id][frame_id:self.K * 2 + frame_id + 1]),
                    int(self.Y[u_id][frame_id]))

    def __len__(self):
        return len(self.look_up)


class Learning:
    def __init__(self, params: HyperParameters):
        self.params = params
        self.str = 'k=' + str(self.params.K) + 'b=' + str(self.params.B) + 'lr=' + str(
            self.params.lr)

        print(str(self))

        self.writer = SummaryWriter(comment=str(self))
        self.train_X = os.path.join(data_dir, 'train.npy')  # N,?,40
        self.train_Y = os.path.join(data_dir, 'train_labels.npy')  # N,?
        self.valid_X = os.path.join(data_dir, 'dev.npy')  # N,?,40
        self.valid_Y = os.path.join(data_dir, 'dev_labels.npy')
        self.test_X = os.path.join(data_dir, 'test.npy')

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = nn.Sequential(nn.Linear(40 * (2 * self.params.K + 1), 1024),
                                   nn.BatchNorm1d(1024), nn.ReLU(),
                                   nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                   nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Linear(256, 71)).cuda()
        self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.init_epoch = 1

        self.device = torch.device("cuda:0")

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def __str__(self):
        return self.str

    def load_train(self):
        self.train_loader = torch.utils.data.DataLoader(
            Dataset(self.train_X, self.train_Y, self.params.K), batch_size=self.params.B,
            shuffle=True)

    def load_valid(self):
        self.valid_loader = torch.utils.data.DataLoader(
            Dataset(self.valid_X, self.valid_Y, self.params.K), batch_size=self.params.B,
            shuffle=False)

    def load_test(self):
        self.test_loader = torch.utils.data.DataLoader(
            Dataset(self.test_X, None, self.params.K), batch_size=1, shuffle=False)

    def load_model(self, epoch=5):
        loaded = torch.load('checkpoints/e=' + str(epoch) + str(self) + '.tar')
        self.init_epoch = loaded['epoch']
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])

    def save_model(self, epoch, loss_item):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_item,
        }, 'checkpoints/e=' + str(epoch) + str(self) + '.tar')

    def train(self):
        assert self.train_loader is not None
        print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch, self.params.max_epoch):
                total_loss = torch.zeros(1, device=self.device)
                total_acc = torch.zeros(1, device=self.device)
                for i, batch in enumerate(self.train_loader):
                    bx = batch[0].to(self.device)
                    by = batch[1].to(self.device)

                    prediction = self.model(bx)
                    loss = self.criterion(prediction, by)
                    total_loss += loss
                    y_prime = torch.argmax(prediction, dim=1)
                    total_acc += torch.count_nonzero(y_prime == by)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if i % 100 == 0:
                        print('epoch: ', epoch, 'iter: ', i)
                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Train', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Train', accuracy_item, epoch)
                print('Training Loss: ', loss_item, 'epoch: ', epoch)

                if epoch % 5 == 0:
                    self.save_model(epoch, loss_item)
                    self.evaluate(epoch)
                    self.model.train()

    def evaluate(self, epoch):
        assert self.valid_loader is not None
        print('Validating...')
        with torch.cuda.device(0):
            with torch.no_grad():
                self.model.eval()
                total_loss = torch.zeros(1, device=self.device)
                total_acc = torch.zeros(1, device=self.device)
                for i, batch in enumerate(self.valid_loader):
                    bx = batch[0].to(self.device)
                    by = batch[1].to(self.device)

                    prediction = self.model(bx)
                    loss = self.criterion(prediction, by)
                    total_loss += loss
                    y_prime = torch.argmax(prediction, dim=1)
                    total_acc += torch.count_nonzero(y_prime == by)

                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Validation', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Validation', accuracy_item, epoch)
                print('Validation loss', loss_item, 'Accuracy', accuracy_item, 'epoch: ', epoch)

    def test(self):
        assert self.test_loader is not None
        print('testing...')
        f = open('results/' + str(self) + '.csv', 'w')
        f.write('id,label')
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for i, item in enumerate(self.test_loader):
                    if i % 100000 == 0:
                        print('testing: ', i)
                    x = item.to(self.device)
                    label = torch.argmax(self.model(x), dim=1).item()
                    f.write('\n' + str(i) + ',' + str(label))
        f.close()


def main():
    for k in [15, 11, 7]:
        for b in [65536, 32768, 8192]:
            for lr in [1e-4, 1e-3, 1e-2]:
                learning = Learning(HyperParameters(k, b, lr))
                learning.load_train()
                learning.load_valid()
                learning.load_test()

                learning.train()
                learning.test()

                del learning


if __name__ == '__main__':
    main()
