import os
from typing import List, Dict

import torch.utils.data
from utils.base import *
from tqdm import tqdm
import numpy as np
from ctcdecode import CTCBeamDecoder
from utils.phoneme_list import N_PHONEMES, PHONEME_MAP
from models import *

import argparse
import Levenshtein

num_workers = 8


class ParamsHW3(Params):
    def __init__(self, B, lr, dropout, device, conv_size, num_layer, hidden_size, bi, schedule_int,
                 max_epoch=20001, data_dir='/home/zongyuez/dldata/HW3'):
        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=dropout,
                         output_channels=1 + N_PHONEMES,
                         data_dir=data_dir, device=device, input_dims=(40,))

        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.bi = bi
        self.conv_size = conv_size
        self.schedule = schedule_int

        self.str = 'class_b=' + str(self.B) + 'lr=' + str(self.lr) + 's' + str(schedule_int) \
                   + 'drop=' + str(self.dropout) + 'n=' + str(num_layer) + 'c=' + str(
                conv_size) + 'h=' + str(hidden_size) + ('Bi' if bi else '')

    def __str__(self):
        return self.str


class TrainSetHW3(torch.utils.data.Dataset):
    def __init__(self, X_path, Y_path):
        super().__init__()
        X = np.load(X_path, allow_pickle=True)
        self.N = X.shape[0]
        self.X = []
        self.lengths_X = []
        for x in X:
            self.X.append(torch.as_tensor(x, dtype=torch.float))
            self.lengths_X.append(x.shape[0])
        self.X = torch.nn.utils.rnn.pad_sequence(self.X, batch_first=True)
        self.len = self.X.shape[0]

        Y = np.load(Y_path, allow_pickle=True)
        self.lengths_Y = []
        self.Y = []
        for y in Y:
            self.Y.append(torch.as_tensor(y, dtype=torch.long))
            self.lengths_Y.append(len(y))
        self.Y = torch.nn.utils.rnn.pad_sequence(self.Y, batch_first=True)

        print(X_path, self.__len__())

    def __getitem__(self, index):
        return self.X[index], self.lengths_X[index], self.Y[index], self.lengths_Y[index]

    def __len__(self):
        return self.len


class TestSetHW3(torch.utils.data.Dataset):
    def __init__(self, X_path):
        super().__init__()
        X = np.load(X_path, allow_pickle=True)
        self.N = X.shape[0]
        self.X = []
        self.lengths = []
        for x in X:
            self.X.append(torch.as_tensor(x, dtype=torch.float))
            self.lengths.append(x.shape[0])
        self.X = torch.nn.utils.rnn.pad_sequence(self.X, batch_first=True)
        self.len = self.X.shape[0]

        print(X_path, self.__len__())

    def __getitem__(self, index):
        return self.X[index], self.lengths[index]

    def __len__(self):
        return self.len


class HW3(Learning):
    def __init__(self, params: ParamsHW3, model):
        super().__init__(params, model, None, nn.CTCLoss)
        self.decoder = CTCBeamDecoder(PHONEME_MAP, log_probs_input=True, num_processes=10,
                                      cutoff_top_n=params.input_dims[0] + 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr,
                                          weight_decay=5e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.params.schedule, 0.5)
        print(str(self))

    def _load_train(self):
        train_set = TrainSetHW3(os.path.join(self.params.data_dir, 'train.npy'),
                                os.path.join(self.params.data_dir, 'train_labels.npy'))
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_valid(self):
        valid_set = TrainSetHW3(os.path.join(self.params.data_dir, 'dev.npy'),
                                os.path.join(self.params.data_dir, 'dev_labels.npy'))

        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        test_set = TestSetHW3(os.path.join(self.params.data_dir, 'test.npy'))

        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=self.params.B, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

    def decode(self, output, lengths):
        """
        :param output: (T,B,42)
        :param lengths: (B)
        :return: [str] (B)
        """
        results, _, _, results_length = self.decoder.decode(torch.transpose(output, 0, 1), lengths)
        strings = []
        for b in range(results.shape[0]):
            letters = results[b, 0, 0:results_length[b][0]]
            b_string = []
            for letter in letters:
                if letter != 0:
                    b_string.append(PHONEME_MAP[letter])
            strings.append(''.join(b_string))
        return strings

    @staticmethod
    def to_str(y, lengths_y):
        """

        :param y: (B,T)
        :param lengths_y: (B)
        :return: [str]
        """
        results = []
        for b, y_b in enumerate(y):
            chars = []
            for char in y_b[0:lengths_y[b]]:
                if char != 0:
                    chars.append(PHONEME_MAP[char])
            results.append(''.join(chars))

        return results

    def train(self, checkpoint_interval=5):
        self._validate(0)

        if self.train_loader is None:
            self._load_train()

        # print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                total_loss = torch.zeros(1, device=self.device)
                # total_acc = torch.zeros(1, device=self.device)
                for i, batch in enumerate(tqdm(self.train_loader)):
                    x = batch[0].to(self.device)
                    lengths_x = batch[1]
                    y = batch[2].to(self.device)
                    lengths_y = batch[3]

                    # (T,N,C)
                    output, lengths_out = self.model(x, lengths_x)

                    loss = self.criterion(output, y, lengths_out, lengths_y)
                    total_loss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_item = total_loss.item() / (i + 1)

                self.writer.add_scalar('Loss/Train', loss_item, epoch)

                print('epoch:', epoch, 'Training Loss:', "%.5f" % loss_item)

                self._validate(epoch)
                self.model.train()
                self.scheduler.step()

                if epoch % checkpoint_interval == 0:
                    self.save_model(epoch)

    def _validate(self, epoch):
        if self.valid_loader is None:
            self._load_valid()

        # print('Validating...')
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                total_loss = torch.zeros(1, device=self.device)
                total_dist = torch.zeros(1, device=self.device)

                for i, batch in enumerate(self.valid_loader):
                    x = batch[0].to(self.device)
                    lengths_x = batch[1]
                    y = batch[2].to(self.device)
                    lengths_y = batch[3]

                    # (T,B,C)
                    output, lengths_out = self.model(x, lengths_x)

                    loss = self.criterion(output, y, lengths_out, lengths_y)
                    total_loss += loss

                    y_strs = HW3.to_str(y, lengths_y)
                    out_strs = self.decode(output, lengths_out)

                    for y_str, out_str in zip(y_strs, out_strs):
                        total_dist += Levenshtein.distance(y_str, out_str)

                loss_item = total_loss.item() / (i + 1)
                dist_item = total_dist.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Loss/Validation', loss_item, epoch)
                self.writer.add_scalar('Distance/Validation', dist_item, epoch)

                print('epoch:', epoch, 'Validation Loss:', "%.5f" % loss_item, 'Distance:',
                      dist_item)

    def test(self):
        if self.test_loader is None:
            self._load_test()

        with open('results/' + str(self) + '.csv', 'w') as f:
            f.write('id,label\n')
            with torch.cuda.device(self.device):
                with torch.no_grad():
                    self.model.eval()
                    for (i, item) in enumerate(tqdm(self.test_loader)):
                        x = item[0].to(self.device)
                        lengths = tuple(item[1])

                        # (T,N,C)
                        output, out_lengths = self.model(x, lengths)
                        results = self.decode(output, out_lengths)

                        for b in range(x.shape[0]):
                            f.write(str(i * self.params.B + b) + ',')
                            f.write(results[b])
                            f.write('\n')

    # def test(self):
    #     if self.test_loader is None:
    #         self._load_test()
    #
    #     with open('results/' + str(self) + '.csv', 'w') as f:
    #         f.write('id,label\n')
    #         with torch.cuda.device(self.device):
    #             with torch.no_grad():
    #                 self.model.eval()
    #                 for (i, item) in enumerate(tqdm(self.test_loader)):
    #                     x = item[0].to(self.device)
    #                     lengths = tuple(item[1])
    #
    #                     # (T,N,C)
    #                     output = self.model(x, lengths)
    #
    #                     for b in range(x.shape[0]):
    #                         f.write(str(i * self.params.B + b) + ',')
    #                         f.write(greedy_search(PHONEME_MAP, output[:, b, :])[0])
    #                         f.write('\n')

    # def test(self):
    #     if self.test_loader is None:
    #         self._load_test()
    #
    #     with open('results/' + str(self) + '.csv', 'w') as f:
    #         f.write('id,label\n')
    #         with torch.cuda.device(self.device):
    #             with torch.no_grad():
    #                 self.model.eval()
    #                 for (i, item) in enumerate(tqdm(self.test_loader)):
    #                     x = item[0].to(self.device)
    #                     lengths = tuple(item[1])
    #
    #                     # (T,N,C)
    #                     output = self.model(x, lengths)
    #
    #                     for b in range(x.shape[0]):
    #                         f.write(str(i * self.params.B + b) + ',')
    #                         f.write(self.decode(output[:, b, :])[0])
    #                         f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='Model1', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', default=10, type=int, help='Checkpoint interval')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--bi', action='store_true')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--h', default=128, type=int)
    parser.add_argument('--c', default=64, type=int)
    parser.add_argument('--schedule', default=5, type=int)

    args = parser.parse_args()

    params = ParamsHW3(B=args.batch, dropout=args.dropout, lr=args.lr,
                       device='cuda:' + args.gpu_id, conv_size=args.c,
                       num_layer=args.layer, hidden_size=args.h, bi=args.bi,
                       schedule_int=args.schedule)

    model = eval(args.model + '(params)')
    learner = HW3(params, model)
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
