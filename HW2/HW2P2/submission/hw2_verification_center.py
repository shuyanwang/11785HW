import os

import numpy as np
import torch.utils.data
import torchvision
from torchvision.datasets.folder import pil_loader

# noinspection PyUnresolvedReferences
from models import *
from losses import *
from utils.base import *

from sklearn.metrics import roc_auc_score

import argparse

num_workers = 4


class HW2ValidPairSet(torch.utils.data.Dataset):
    def __getitem__(self, index):
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
            pairs = f.read().splitlines()
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

        self.gt_array = np.zeros(len(self.items), dtype=int)
        for (i, item) in enumerate(self.items):
            self.gt_array[i] = item[2]


class ParamsHW2ClassificationCenter(Params):
    def __init__(self, B, lr, device, flip, normalize, rotate,
                 erase, resize, perspective, feature_dims, lambDA, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification'):

        self.size = 64 if resize <= 0 else resize
        self.feature_dims = feature_dims
        self.lambDA = lambDA

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=0, output_channels=4000,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))

        self.str = 'class_b=' + str(self.B) + 'lr=' + str(
                self.lr) + 'fd=' + str(feature_dims) + 'lam=' + str(self.lambDA) + '_'

        transforms_train = []
        transforms_test = []

        if self.size != 64:
            self.str = self.str + 'r' + str(self.size)
            transforms_train.append(torchvision.transforms.Resize(self.size))
            transforms_test.append(torchvision.transforms.Resize(self.size))

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + 'f'

        if rotate:
            transforms_train.append(torchvision.transforms.RandomRotation(15))
            self.str = self.str + 'r'

        if perspective:
            transforms_train.append(torchvision.transforms.RandomPerspective())
            self.str = self.str + 'p'

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

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class HW2ClassificationC(Learning):
    def __init__(self, params: ParamsHW2ClassificationCenter, model, loss: CrossEntropyCenterLoss,
                 optimizer):
        super().__init__(params, model, None, None)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.params.lr) if optimizer == 'Adam' else \
            torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=5e-5,
                            momentum=0.9)
        self.criterion = loss(params).cuda(params.device)
        self.str = self.str + '_' + optimizer

        print(str(self))

    @staticmethod
    def score(y1, y2, dim=1):
        # P2-Dist-based similarity is no better than Cosine
        return torch.cosine_similarity(y1, y2, dim=dim)

    def _load_train(self):
        train_set = torchvision.datasets.ImageFolder(
                os.path.join(self.params.data_dir, 'train_data'),
                transform=self.params.transforms_train)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)
        self.label_to_class = train_set.classes

    def _load_valid(self):
        valid_set = HW2ValidPairSet(validation=True, transform=self.params.transforms_test)
        self.gt_labels = valid_set.gt_array

        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        self.test_set = HW2ValidPairSet(validation=False, transform=self.params.transforms_test)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.params.B, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

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

                        features1, _ = self.model(x1)
                        features2, _ = self.model(x2)

                        for b in range(features1.shape[0]):
                            f.write(self.test_set.items[i * self.params.B + b][3] + ' ' +
                                    self.test_set.items[i * self.params.B + b][4] + ',' +
                                    str(self.score(features1[b], features2[b],
                                                   dim=0).item()) + '\n')

    def train(self, checkpoint_interval=5):
        self._validate(self.init_epoch)

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

                    features, prediction = self.model(bx)

                    loss = self.criterion(features, prediction, by)
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

                if epoch % checkpoint_interval == 0:
                    self.save_model(epoch)

    def _validate(self, epoch):
        if self.valid_loader is None:
            self._load_valid()

        valid_scores = np.zeros(self.gt_labels.shape)

        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()

                for i, batch in enumerate(self.valid_loader):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    features1, prediction1 = self.model(bx1)
                    features2, prediction2 = self.model(bx2)

                    score = self.score(features1, features2)
                    valid_scores[i * self.params.B:i * self.params.B + by.shape[
                        0]] = score.cpu().detach().numpy()
                score_item = roc_auc_score(self.gt_labels, valid_scores)
                self.writer.add_scalar('Score/Validation', score_item, epoch)
                print('epoch:', epoch, 'Validation Score:', "%.5f" % score_item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=128, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='ResNet34K3S1RC', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--resize', default=-1, help='Resize Image', type=int)
    parser.add_argument('--save', default=5, type=int, help='Checkpoint interval')
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--loss', default='CrossEntropyCenterLoss')
    parser.add_argument('--feature_dims', default=512, type=int)
    parser.add_argument('--lambDA', default=0.01, type=float)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--optimizer', default='SGD')

    args = parser.parse_args()

    params = ParamsHW2ClassificationCenter(B=args.batch, lr=args.lr,
                                           device='cuda:' + args.gpu_id, flip=args.flip,
                                           normalize=args.normalize, erase=args.erase,
                                           resize=args.resize, perspective=args.perspective,
                                           feature_dims=args.feature_dims, lambDA=args.lambDA,
                                           rotate=args.rotate)
    model = eval(args.model + '(params)')
    learner = HW2ClassificationC(params, model, eval(args.loss), args.optimizer)
    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)

    if args.train:
        learner.train(checkpoint_interval=args.save)
    if args.test:
        learner.test()


if __name__ == '__main__':
    main()

# --train --flip --erase --perspective
