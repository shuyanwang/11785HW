import os
from typing import Optional

import torch.utils.data

from utils.base import Params, Learning
from tqdm import tqdm
import torchvision
from torchvision.datasets.folder import pil_loader

from models import *
from model_efficientnet import *
from losses import *

import argparse
import numpy as np

from hw2_verification_pair import HW2ValidPairSet
from sklearn.metrics import roc_auc_score

from pytorch_metric_learning import losses, miners

num_workers = 4


class ParamsHW2Verification(Params):
    def __init__(self, B, lr, device, flip, normalize,
                 erase, resize, perspective, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification',
                 margin=1.0):

        self.margin = margin

        self.size = 64 if resize <= 0 else resize

        super().__init__(B=B, lr=lr, max_epoch=max_epoch,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))
        self.str = 'verify_b=' + str(self.B) + 'margin=' + str(margin) + 'lr=' + str(lr) + '_'

        transforms_train = []
        transforms_test = []

        if self.size != 64:
            self.str = self.str + 'r' + str(self.size)
            transforms_train.append(torchvision.transforms.Resize(self.size))
            transforms_test.append(torchvision.transforms.Resize(self.size))

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + 'f'

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

        self.str = self.str + '_'

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class HW2VerificationMiner(Learning):
    def __init__(self, params: ParamsHW2Verification, model: Model, loss, miner, score):
        super().__init__(params, model, torch.optim.Adam, None,
                         string=loss.__name__ + '_' + miner.__name__ + '_' +
                                model.__class__.__name__ + '_score=' + score + '_' + str(params))

        self.gt_labels: Optional[np.ndarray] = None
        self.criterion = loss(margin=self.params.margin).cuda(self.device)
        self.miner = miner().cuda(self.device)
        self.score_method = score
        print(str(self))

    def score(self, y1, y2):
        if self.score_method == 'cosine':
            return torch.cosine_similarity(y1, y2)
        else:
            dist = torch.pairwise_distance(y1, y2)
            return 1 / (1 + dist)

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
                                                       batch_size=1, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

    def train(self, checkpoint_interval=5):
        # self._validate(self.init_epoch)

        if self.train_loader is None:
            self._load_train()

        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                print('Epoch:', epoch)
                total_loss = torch.zeros(1, device=self.device)

                for i, batch in enumerate(tqdm(self.train_loader)):
                    bx0 = batch[0].to(self.device)
                    by = batch[1].to(self.device)

                    embeddings = self.model(bx0)

                    hard_pairs = self.miner(embeddings, by)

                    loss = self.criterion(embeddings, by, hard_pairs)
                    total_loss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_item = total_loss.item() / (i + 1)
                self.writer.add_scalar('Loss/Train', loss_item, epoch)

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

                for i, batch in enumerate(tqdm(self.valid_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    y1 = self.model(bx1)
                    y2 = self.model(bx2)

                    score = self.score(y1, y2)
                    valid_scores[i * self.params.B:i * self.params.B + by.shape[
                        0]] = score.cpu().detach().numpy()

                self.writer.add_scalar('Score/Validation',
                                       roc_auc_score(self.gt_labels, valid_scores), epoch)

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
                                str(self.score(y1, y2).item()) + '\n')


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
    parser.add_argument('--margin', default=1.0, type=float)
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--miner', default='miners.MultiSimilarityMiner')
    parser.add_argument('--loss', default='losses.TripletMarginLoss')
    parser.add_argument('--score', default='cosine')

    args = parser.parse_args()

    params = ParamsHW2Verification(B=args.batch, lr=args.lr,
                                   device='cuda:' + args.gpu_id, flip=args.flip,
                                   normalize=args.normalize, erase=args.erase,
                                   resize=args.resize, margin=args.margin,
                                   perspective=args.perspective,
                                   )
    model = eval(args.model + '(params)')
    learner = HW2VerificationMiner(params, model, eval(args.loss), eval(args.miner), args.score)

    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)
    if args.train:
        learner.train(args.save)
    if args.test:
        learner.test()


if __name__ == '__main__':
    main()
