import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co
from torch.utils.tensorboard import SummaryWriter

data_dir = 'E:/11785_data/HW1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

is_load_model = True

is_train = True


class HyperParameters:
    context_K = 11
    batch_size = 10000
    lr = 1e-3
    max_epoch = 100000


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_dir, Y_dir=None, context_K=HyperParameters.context_K):
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


def main():
    writer = SummaryWriter('./logs', comment=str(HyperParameters.context_K))

    train_X = os.path.join(data_dir, 'train.npy')  # N,?,40
    train_Y = os.path.join(data_dir, 'train_labels.npy')  # N,?

    valid_X = os.path.join(data_dir, 'dev.npy')  # N,?,40
    valid_Y = os.path.join(data_dir, 'dev_labels.npy')

    test_X = os.path.join(data_dir, 'test.npy')

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_X, train_Y), batch_size=HyperParameters.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        Dataset(valid_X, valid_Y), batch_size=HyperParameters.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(Dataset(test_X), batch_size=1, shuffle=False)

    model = nn.Sequential(nn.Linear(40 * (2 * HyperParameters.context_K + 1), 1024),
                          nn.BatchNorm1d(1024), nn.ReLU(),
                          nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                          nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                          nn.Linear(256, 71)).cuda()
    model.double()
    if is_load_model:
        model.load_state_dict(
            torch.load('checkpoints/model' + str(HyperParameters.context_K) + '.tar'))

    if is_train:
        train(model, train_loader, valid_loader, writer)

    test(model, test_loader)

    writer.flush()
    writer.close()


def train(model, train_loader, valid_loader, writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=HyperParameters.lr)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.cuda.device(0):
        model.train()
        for epoch in range(1, HyperParameters.max_epoch):
            # print('epoch: ', epoch)
            for i, batch in enumerate(train_loader):
                bx = batch[0].to(device)
                by = batch[1].to(device)

                prediction = model(bx)
                loss = criterion(prediction, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('epoch: ', epoch, 'iter: ', i)

            writer.add_scalar('Loss/Train', loss, epoch)
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'checkpoints/model' + str(HyperParameters.context_K) + '.tar')
                evaluate(valid_loader, model, criterion, epoch, writer)
                model.train()


def evaluate(valid_loader, model, criterion, epoch, writer):
    with torch.cuda.device(0):
        with torch.no_grad():
            model.eval()
            for bx, by in valid_loader:
                bx = bx.to(device)
                by = by.to(device)

                prediction = model(bx)
                loss = criterion(prediction, by)

                writer.add_scalar('Loss/Validation', loss, epoch)


def test(model, test_loader):
    f = open('results/model' + str(HyperParameters.context_K) + '.csv', 'w')
    f.write('id,label\n')
    with torch.cuda.device(0):
        with torch.no_grad():
            model.eval()
            for i, item in enumerate(test_loader):
                x = item.to(device)
                label = torch.argmax(model(x), dim=1).item()
                f.write(str(i) + str(int(label)))
    f.close()


if __name__ == '__main__':
    main()
