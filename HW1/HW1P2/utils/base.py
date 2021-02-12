import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataset import T_co
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Params(ABC):
    B: int = field()
    data_dir: str = field()
    lr: float = field(default=1e-3)
    max_epoch: int = field(default=41)
    is_double: int = field(default=False)
    device: torch.device = field(default=torch.device("cuda:0"))

    @abstractmethod
    def __str__(self):
        return ''


class Learning(ABC):
    def __init__(self, params, model: nn.Module, optimizer_handle, criterion_handle):

        self.params = params
        self.device = params.device
        self.str = str(params) + model.__class__.__name__

        self.writer = SummaryWriter('runs/' + str(self))

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = model.cuda(self.device)
        if params.is_double:
            self.model.double()

        self.optimizer = optimizer_handle(self.model.parameters(), lr=self.params.lr)
        self.criterion = criterion_handle().cuda(self.device)

        self.init_epoch = 1

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def __str__(self):
        return self.str

    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_valid(self):
        pass

    @abstractmethod
    def load_test(self):
        pass

    def load_model(self, epoch=5):
        loaded = torch.load('checkpoints/' + str(self) + '_e=' + str(epoch) + '.tar')
        self.init_epoch = loaded['epoch']
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])

    def save_model(self, epoch, loss_item):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_item,
        }, 'checkpoints/' + str(self) + '_e=' + str(epoch) + '.tar')

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

    @abstractmethod
    def test(self):
        pass

    def learn(self):
        self.load_train()
        self.load_valid()
        self.load_test()
        self.train()
        self.test()
