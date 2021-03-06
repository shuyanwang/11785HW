import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List


class PairLoss(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y1: torch.Tensor, y2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, y1, y2, *args):
        pass


@dataclass
class Params(ABC):
    B: int = field(default=1024)
    data_dir: str = field(default='')
    lr: float = field(default=1e-3)
    max_epoch: int = field(default=101)
    is_double: int = field(default=False)
    device: torch.device = field(default=torch.device("cuda:0"))
    dropout: float = field(default=0.)
    input_dims: tuple = field(default=(3, 64, 64))
    output_channels: int = field(default=4000)

    @abstractmethod
    def __str__(self):
        return ''


class Model(nn.Module, ABC):
    @abstractmethod
    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class Learning(ABC):
    def __init__(self, params, model: Model, optimizer_handle, criterion_handle,
                 draw_graph=False, string=None):

        self.params = params
        self.device = params.device
        self.str = model.__class__.__name__ + '_' + str(params) if string is None else string

        self.writer = SummaryWriter('runs/' + str(self))

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.label_to_class = None

        self.model = model.cuda(self.device)
        if params.is_double:
            self.model.double()

        if draw_graph:
            self.writer.add_graph(model, torch.rand([params.B] + list(params.input_dims),
                                                    device=self.device))

        self.optimizer = optimizer_handle(self.model.parameters(), lr=self.params.lr)
        self.criterion = criterion_handle().cuda(self.device)

        self.init_epoch = 0

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def __str__(self):
        return self.str

    @abstractmethod
    def _load_train(self):
        pass

    @abstractmethod
    def _load_valid(self):
        pass

    @abstractmethod
    def _load_test(self):
        pass

    def load_model(self, epoch=20):
        loaded = torch.load('checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar')
        self.init_epoch = loaded['epoch']
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])

        if 'loss_state_dict' in loaded:
            self.criterion.load_state_dict(loaded['loss_state_dict'])

    def save_model(self, epoch, loss_item):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.criterion.state_dict(),
        }, 'checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar')

    def train(self):
        if self.train_loader is None:
            self._load_train()

        print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                total_loss = torch.zeros(1, device=self.device)
                total_acc = torch.zeros(1, device=self.device)
                for i, batch in enumerate(tqdm(self.train_loader)):
                    bx = batch[0].to(self.device)
                    by = batch[1].to(self.device)

                    prediction = self.model(bx)

                    loss = self.criterion(prediction, by)
                    total_loss += loss
                    y_prime = torch.argmax(prediction, dim=1)
                    total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Train', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Train', accuracy_item, epoch)
                print('epoch: ', epoch, 'Training Loss: ', "%.5f" % loss_item,
                      'Accuracy: ', "%.5f" % accuracy_item)

                self._validate(epoch)
                self.model.train()

                if epoch % 5 == 0:
                    self.save_model(epoch, loss_item)

    def _validate(self, epoch):
        if self.valid_loader is None:
            self._load_valid()

        # print('Validating...')
        with torch.cuda.device(self.device):
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
                    total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Validation', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Validation', accuracy_item, epoch)
                print('epoch: ', epoch, 'Validation Loss: ', "%.5f" % loss_item,
                      'Accuracy: ', "%.5f" % accuracy_item)

    @abstractmethod
    def test(self):
        pass
