import os
import torch.utils.data

from utils.base import Params, Learning
from tqdm import tqdm
import torchvision
from torchvision.datasets.folder import pil_loader

from models import *
from losses import *

import argparse
import numpy as np

from hw2_verification_pair import HW2ValidPairSet

num_workers = 6


class ParamsHW2Verification(Params):
    def __init__(self, B, lr, device, flip, normalize,
                 erase, resize, positive, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification'
                          '/train_data', loss_lr=1e-2):

        self.loss_lr = loss_lr

        self.size = 64 if resize <= 0 else resize

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, output_channels=2,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))
        self.pos_p = positive
        self.str = 'verify_b=' + str(self.B) + 'p=' + str(self.pos_p) + 'loss_lr=' + str(
                self.loss_lr) + '_'

        transforms_train = []
        transforms_test = []

        if self.size != 64:
            self.str = self.str + 'r' + str(self.size)
            transforms_train.append(torchvision.transforms.Resize(self.size))
            transforms_test.append(torchvision.transforms.Resize(self.size))

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + 'f'

        transforms_train.append(torchvision.transforms.ToTensor())
        transforms_test.append(torchvision.transforms.ToTensor())

        if normalize:
            self.str = self.str + 'n'
            transforms_test.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            transforms_train.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        if erase:
            transforms_train.append(torchvision.transforms.RandomErasing())
            self.str = self.str + 'e'

        self.str = self.str + '_'

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class HW2VerificationSimple(Learning):
    def __init__(self, params: ParamsHW2Verification, model: Model):
        super().__init__(params, model, torch.optim.Adam, torch.nn.CrossEntropyLoss,
                         string='simple_' + model.__class__.__name__ + '_' + str(params))
        print(str(self))

    def predict(self, y1, y2):
        y1 = torch.argmax(y1, dim=1)
        y2 = torch.argmax(y2, dim=1)
        return torch.eq(y1, y2).int()

    def _load_train(self):
        pass

    def _load_valid(self):
        valid_set = HW2ValidPairSet(validation=True, transform=self.params.transforms_test)

        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        self.test_set = HW2ValidPairSet(validation=False, transform=self.params.transforms_test)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=1, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

    def _validate(self, epoch):
        if self.valid_loader is None:
            self._load_valid()

        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                total_acc = torch.zeros(1, device=self.device)

                for i, batch in enumerate(tqdm(self.valid_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    y1 = self.model(bx1)
                    y2 = self.model(bx2)

                    y_prime = self.predict(y1, y2)
                    total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Accuracy/Validation', accuracy_item, epoch)

    def test(self):
        if self.test_loader is None:
            self._load_test()

        with open('results/' + str(self) + '.csv', 'w') as f:
            f.write('Id,Category\n')
            with torch.cuda.device(self.device):
                with torch.no_grad():
                    self.model.eval()
                    for (i, item) in enumerate(tqdm(self.test_loader)):
                        x1 = item[0].to(self.device)
                        x2 = item[1].to(self.device)

                        y1 = self.model(x1)
                        y2 = self.model(x2)

                        f.write(self.test_set.items[i][3] + ' ' +
                                self.test_set.items[i][4] + ',' +
                                str(self.predict(y1, y2).item()) + '\n')

    def load_model(self, epoch=20, name=None):
        if name is None:
            loaded = torch.load('checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar')
        else:
            loaded = torch.load('checkpoints/' + name + 'e=' + str(epoch) + '.tar')
        self.init_epoch = loaded['epoch']
        model_states = loaded['model_state_dict']
        # del model_states['net.linear.weight']
        # del model_states['net.linear.bias']

        self.model.load_state_dict(model_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='ResNet101', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--resize', default=224, help='Resize Image', type=int)
    parser.add_argument('--save', default=5, type=int, help='Checkpoint interval')
    parser.add_argument('--pos', default='0.5', type=float,
                        help='Probability of choosing same class, otherwise randomly chosen')
    parser.add_argument('--loss_lr', default=1e-2, type=float)

    args = parser.parse_args()

    params = ParamsHW2Verification(B=args.batch, lr=args.lr,
                                   device='cuda:' + args.gpu_id, flip=args.flip,
                                   normalize=args.normalize, erase=args.erase,
                                   resize=args.resize, positive=args.pos)
    model = eval(args.model + '(params)')
    learner = HW2VerificationSimple(params, model)

    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)

    if args.test:
        learner._validate(learner.init_epoch)
        learner.test()


if __name__ == '__main__':
    main()
