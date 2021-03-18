import os

import numpy as np
import torch.utils.data
import torchvision

from model_efficientnet import *
from losses import *

from hw2_verification_pair import HW2ValidPairSet
from sklearn.metrics import roc_auc_score

import argparse

num_workers = 4


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
    def __init__(self, params: ParamsHW2ClassificationCenter, model, loss: CrossEntropyCenterLoss):
        super().__init__(params, model, torch.optim.Adam, None)

        self.criterion = loss(params).cuda(params.device)

        print(str(self))

    @staticmethod
    def score(y1, y2):
        return torch.cosine_similarity(y1, y2)

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
        self.test_set = torchvision.datasets.ImageFolder(
                os.path.join(self.params.data_dir, 'test_data'),
                transform=self.params.transforms_test)

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

                for i, batch in enumerate(tqdm(self.valid_loader)):
                    bx1 = batch[0].to(self.device)
                    bx2 = batch[1].to(self.device)
                    by = batch[2].to(self.device)

                    features1, prediction1 = self.model(bx1)
                    features2, prediction2 = self.model(bx2)

                    score = self.score(features1, features2)
                    valid_scores[i * self.params.B:i * self.params.B + by.shape[
                        0]] = score.cpu().detach().numpy()

                self.writer.add_scalar('Score/Validation',
                                       roc_auc_score(self.gt_labels, valid_scores), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='EfficientNetB4C', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--resize', default=224, help='Resize Image', type=int)
    parser.add_argument('--save', default=5, type=int, help='Checkpoint interval')
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--loss', default='CrossEntropyCenterLoss')
    parser.add_argument('--feature_dims', default=1792, type=int)
    parser.add_argument('--lambDA', default=0.5, type=float)
    parser.add_argument('--rotate', action='store_true')

    args = parser.parse_args()

    params = ParamsHW2ClassificationCenter(B=args.batch, lr=args.lr,
                                           device='cuda:' + args.gpu_id, flip=args.flip,
                                           normalize=args.normalize, erase=args.erase,
                                           resize=args.resize, perspective=args.perspective,
                                           feature_dims=args.feature_dims, lambDA=args.lambDA,
                                           rotate=args.rotate)
    model = eval(args.model + '(params)')
    learner = HW2ClassificationC(params, model, eval(args.loss))
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
