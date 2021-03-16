import os
from typing import Union

import torch.utils.data

from utils.base import Params, Learning
from tqdm import tqdm
import torchvision
from torchvision.datasets.folder import pil_loader
from sklearn.metrics import roc_auc_score

from model_efficientnet import *
from losses import *

import argparse
import numpy as np

from hw2_verification_pair import HW2ValidPairSet

num_workers = 6


class ParamsHW2VerificationS(Params):
    def __init__(self, B, lr, device, normalize, resize, crop, max_epoch=201,
                 data_dir='c:/DLData/11785_data/HW2/11785-spring2021-hw2p2s1-face-classification'
                          '/train_data'):

        self.size = 64 if resize <= 0 else resize

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, output_channels=4000,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))
        self.str = 'verify_simple_'

        transforms_test = []

        if self.size != 64:
            self.str = self.str + 'r' + str(self.size)
            transforms_test.append(torchvision.transforms.Resize(self.size))

        if crop > 0:
            self.str = self.str + 'c' + str(crop)
            transforms_test.append(torchvision.transforms.CenterCrop(crop))

        transforms_test.append(torchvision.transforms.ToTensor())

        if normalize:
            self.str = self.str + 'n'
            transforms_test.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class HW2VerificationSimple(Learning):
    def __init__(self, params: ParamsHW2VerificationS, model: Model, loss):
        super().__init__(params, model, torch.optim.Adam, None,
                         string=loss.__name__ + '_' + model.__class__.__name__ + '_' + str(params))
        self.criterion: Union[TripletLoss, PairLoss] = loss().cuda(params.device)
        print(str(self))

    def features(self, x):
        if 'EfficientNet' in self.model.__class__.__name__:
            return torch.flatten(self.model.net.extract_features(x), start_dim=1)

        return

    def score(self, y1, y2):
        return self.criterion.score(y1, y2)

    def _load_train(self):
        pass

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

    def _validate(self, epoch):
        if self.valid_loader is None:
            self._load_valid()

        with torch.cuda.device(self.device):

            valid_scores = np.zeros(self.gt_labels.shape)
            with torch.no_grad():
                self.model.eval()

                for i, batch in enumerate(tqdm(self.valid_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    y1 = self.features(bx1)
                    y2 = self.features(bx2)

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

                        y1 = self.features(x1)
                        y2 = self.features(x2)

                        f.write(self.test_set.items[i][3] + ' ' +
                                self.test_set.items[i][4] + ',' +
                                str(self.score(y1, y2).item()) + '\n')

    def load_model(self, epoch=20, name=None):
        if name is None:
            loaded = torch.load('checkpoints/' + str(self) + 'e=' + str(epoch) + '.tar',
                                map_location=self.device)
        else:
            loaded = torch.load('checkpoints/' + name + 'e=' + str(epoch) + '.tar',
                                map_location=self.device)
        self.init_epoch = loaded['epoch']
        model_states = loaded['model_state_dict']
        # del model_states['net.linear.weight']
        # del model_states['net.linear.bias']

        self.model.load_state_dict(model_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='ResNet101', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--resize', default=224, help='Resize Image', type=int)
    parser.add_argument('--loss', default='SwapTripletMarginLoss')
    parser.add_argument('--crop', default=380, type=int)

    args = parser.parse_args()

    params = ParamsHW2VerificationS(B=args.batch, lr=args.lr,
                                    device='cuda:' + args.gpu_id, normalize=args.normalize,
                                    resize=args.resize, crop=args.crop)
    model = eval(args.model + '(params)')
    learner = HW2VerificationSimple(params, model, eval(args.loss))

    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)

    if args.test:
        learner._validate(learner.init_epoch)
        # learner.test()


if __name__ == '__main__':
    main()
