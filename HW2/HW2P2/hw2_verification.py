import os
import torch.utils.data
from torch.utils.data.dataset import T_co

from utils.base import Params, Learning
from tqdm import tqdm
import torchvision
from torchvision.datasets.folder import pil_loader

from models import *
from losses import *

import argparse
import numpy as np

num_workers = 8


class ParamsHW2Verification(Params):
    def __init__(self, B, lr, device, flip, normalize,
                 erase, resize, positive, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification'
                          '/train_data'):

        self.size = 64 if resize <= 0 else resize

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, output_channels=2,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))
        self.pos_p = positive
        self.str = 'verify_b=' + str(self.B) + 'p=' + str(self.pos_p)

        transforms_train = []
        transforms_test = []

        if self.size != 64:
            self.str = self.str + '_r' + str(self.size)
            transforms_train.append(torchvision.transforms.Resize(self.size))
            transforms_test.append(torchvision.transforms.Resize(self.size))

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + '_f'

        transforms_train.append(torchvision.transforms.ToTensor())
        transforms_test.append(torchvision.transforms.ToTensor())

        if normalize:
            self.str = self.str + '_n'
            transforms_test.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            transforms_train.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        if erase:
            transforms_train.append(torchvision.transforms.RandomErasing())
            self.str = self.str + '_e'

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class HW2TrainingPairSet(torch.utils.data.Dataset):
    def __init__(self, params):
        self.params: ParamsHW2Verification = params
        self.classes = dict()
        self.lookup = []  # label @ id
        self.offset = [0 for _ in range(4000)]  # num of images before current label
        for label in range(4000):
            self.classes[label] = []
            for img_name in os.listdir(os.path.join(self.params.data_dir, str(label))):
                self.lookup.append(label)
                self.classes[label].append(img_name)

        for label in range(1, 4000):
            self.offset[label] = self.offset[label - 1] + len(self.classes[label - 1])

        self.transform = self.params.transforms_train

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        label1 = self.lookup[index]
        item1 = self.classes[label1][index - self.offset[label1]]
        get_same_class = np.random.binomial(1, self.params.pos_p) == 1
        if get_same_class:
            label2 = label1
            item2 = self.classes[label1][np.random.randint(0, len(self.classes[label1]))]
        else:
            # randomly chosen (could be 1)
            label2 = np.random.randint(0, 4000)
            item2 = self.classes[label2][np.random.randint(0, len(self.classes[label2]))]

        return self.transform(pil_loader(os.path.join(self.params.data_dir, str(label1), item1))), \
               self.transform(pil_loader(os.path.join(self.params.data_dir, str(label2), item2))), \
               1 if label1 == label2 else 0


class HW2ValidPairSet(torch.utils.data.Dataset):
    def __getitem__(self, index) -> T_co:
        item = self.items[index]
        return self.transform(pil_loader(item[0])), self.transform(pil_loader(item[1])), item[2], \
               item[0], item[1]  # for testing

    def __len__(self):
        return len(self.items)

    def __init__(self, validation, transform):
        self.set_path = 'c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-verification'
        txt_path = 'verification_pairs_val.txt' if validation else 'verification_pairs_test.txt'
        self.items = []
        self.transform = transform
        with open(os.path.join(self.set_path, txt_path)) as f:
            pairs = f.readlines()
            for pair in pairs:
                pair = pair.split(' ')
                if validation:
                    self.items.append((os.path.join(self.set_path, pair[0]),
                                       os.path.join(self.set_path, pair[1]),
                                       1 if int(pair[2]) > 0 else 0, pair[0], pair[1]))
                else:
                    self.items.append((os.path.join(self.set_path, pair[0]),
                                       os.path.join(self.set_path, pair[1]),
                                       0, pair[0], pair[1]))


class HW2VerificationPair(Learning):
    def __init__(self, params: ParamsHW2Verification, model: Model, loss: PairLoss):
        super().__init__(params, model, torch.optim.Adam, loss,
                         string=loss.__name__ + '_' + model.__class__.__name__ + '_' + str(params))
        print(str(self))

    def predict(self, y1, y2):
        return self.criterion.predict(y1, y2)

    def _load_train(self):
        train_set = HW2TrainingPairSet(self.params)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)

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

    def train(self):
        if self.train_loader is None:
            self._load_train()

        print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                print('Epoch:', epoch)
                total_loss = torch.zeros(1, device=self.device)
                total_acc = torch.zeros(1, device=self.device)

                TP = torch.zeros(1, device=self.device)
                FP = torch.zeros(1, device=self.device)
                TN = torch.zeros(1, device=self.device)
                FN = torch.zeros(1, device=self.device)

                for i, batch in enumerate(tqdm(self.train_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    y1 = self.model(bx1)
                    y2 = self.model(bx2)

                    loss = self.criterion(y1, y2, by)
                    total_loss += loss
                    y_prime = self.predict(y1, y2)
                    total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                    TP += torch.count_nonzero(torch.logical_and(y_prime, by))
                    TN += torch.count_nonzero(torch.logical_and(
                            torch.logical_not(y_prime), torch.logical_not(by)))

                    FP += torch.count_nonzero(torch.logical_and(
                            y_prime, torch.logical_not(by)))
                    FN += torch.count_nonzero(torch.logical_and(
                            torch.logical_not(y_prime), by))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Train', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Train', accuracy_item, epoch)
                self.writer.add_scalar('Precision/Train', (TP / (TP + FP)).item(), epoch)
                self.writer.add_scalar('Recall/Train', (TP / (TP + FN).item()), epoch)
                self.writer.add_scalar('TPR/Train', (TP / (TP + FN)).item(), epoch)
                self.writer.add_scalar('FPR/Train', (FP / (TN + FP)).item(), epoch)

                # print('epoch: ', epoch, 'Training Loss: ', "%.5f" % loss_item,
                #       'Accuracy: ', "%.5f" % accuracy_item)

                self._validate(epoch)
                self.model.train()

                if epoch % 1 == 0:
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

                TP = torch.zeros(1, device=self.device)
                FP = torch.zeros(1, device=self.device)
                TN = torch.zeros(1, device=self.device)
                FN = torch.zeros(1, device=self.device)

                for i, batch in enumerate(tqdm(self.valid_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    y1 = self.model(bx1)
                    y2 = self.model(bx2)

                    loss = self.criterion(y1, y2, by)
                    total_loss += loss
                    y_prime = self.predict(y1, y2)
                    total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                    TP += torch.count_nonzero(torch.logical_and(y_prime, by))
                    TN += torch.count_nonzero(torch.logical_and(
                            torch.logical_not(y_prime), torch.logical_not(by)))

                    FP += torch.count_nonzero(torch.logical_and(
                            y_prime, torch.logical_not(by)))
                    FN += torch.count_nonzero(torch.logical_and(
                            torch.logical_not(y_prime), by))

                loss_item = total_loss.item() / (i + 1)
                accuracy_item = total_acc.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Validation', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Validation', accuracy_item, epoch)
                self.writer.add_scalar('Precision/Validation', (TP / (TP + FP)).item(), epoch)
                self.writer.add_scalar('Recall/Validation', (TP / (TP + FN).item()), epoch)
                self.writer.add_scalar('TPR/Validation', (TP / (TP + FN)).item(), epoch)
                self.writer.add_scalar('FPR/Validation', (FP / (TN + FP)).item(), epoch)
                # print('epoch: ', epoch, 'Validation Loss: ', "%.5f" % loss_item,
                #       'Accuracy: ', "%.5f" % accuracy_item)

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
                                self.test_set.items[i][4] + ' ' +
                                str(self.predict(y1, y2)) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='ResNet101', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--resize', default=224, help='Resize Image', type=int)
    parser.add_argument('--loss', default='AdaptiveContrastiveLoss')
    # parser.add_argument('--threshold', default='0.5', type=float)
    parser.add_argument('--pos', default='0.3', type=float,
                        help='Probability of choosing same class, otherwise randomly chosen')

    args = parser.parse_args()

    params = ParamsHW2Verification(B=args.batch, lr=args.lr,
                                   device='cuda:' + args.gpu_id, flip=args.flip,
                                   normalize=args.normalize, erase=args.erase,
                                   resize=args.resize, positive=args.pos)
    model = eval(args.model + '(params)')
    learner = HW2VerificationPair(params, model, eval(args.loss))

    if args.epoch >= 0:
        learner.load_model(args.epoch)
    if args.train:
        learner.train()
    if args.test:
        learner.test()


if __name__ == '__main__':
    main()
