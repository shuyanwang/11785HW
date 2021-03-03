import os
import torch.utils.data
from utils.base import Params, Learning
from tqdm import tqdm
import torchvision

from models import *

import argparse

num_workers = 8


class ParamsHW2Classification(Params):
    def __init__(self, B=1024, lr=1e-3, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification',
                 dropout=0.5, device='cuda:0', flip=False):
        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=dropout,
                         data_dir=data_dir, is_double=False, device=device)

        self.str = 'class_b=' + str(self.B) + 'lr=' + str(self.lr) + 'd=' + str(self.dropout)

        transforms = []
        if flip:
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + '_f'

        transforms.append(torchvision.transforms.ToTensor())
        self.transforms = torchvision.transforms.Compose(transforms)

    def __str__(self):
        return self.str


class HW2Classification(Learning):
    def __init__(self, params: ParamsHW2Classification, model):
        super().__init__(params, model, torch.optim.Adam, nn.CrossEntropyLoss)
        print(str(self))

    def _load_train(self):
        train_set = torchvision.datasets.ImageFolder(
                os.path.join(self.params.data_dir, 'train_data'), transform=self.params.transforms)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)
        self.label_to_class = train_set.classes

    def _load_valid(self):
        valid_set = torchvision.datasets.ImageFolder(
                os.path.join(self.params.data_dir, 'val_data'), transform=self.params.transforms)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        self.test_set = torchvision.datasets.ImageFolder(
                os.path.join(self.params.data_dir, 'test_data'), transform=self.params.transforms)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=1, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

    def test(self):
        if self.test_loader is None:
            self._load_test()

        if self.label_to_class is None:
            self._load_train()  # for class labels

        results = torch.zeros(len(self.test_set.imgs), dtype=torch.int)

        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                for (i, item) in enumerate(tqdm(self.test_loader)):
                    x = item[0].to(self.device)
                    labels = torch.argmax(self.model(x), dim=1)
                    file_id = int(self.test_set.imgs[i][0].split('\\')[-1].split('.')[0])
                    results[file_id] = int(self.label_to_class[labels.item()])

        with open('results/' + str(self) + '.csv', 'w') as f:
            f.write('id,label\n')
            for i, result in enumerate(results):
                f.write(str(i) + '.jpg,' + str(result.item()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', help='Batch Size', default=1024, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='ResNet34', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    params = ParamsHW2Classification(B=args.B, dropout=args.dropout, lr=args.lr,
                                     device='cuda:' + args.gpu_id)
    model = eval(args.model + '(params)')
    learner = HW2Classification(params, model)
    if args.epoch >= 0:
        learner.load_model(args.epoch)
    if args.train:
        learner.train()
    if args.test:
        learner.test()


if __name__ == '__main__':
    main()
