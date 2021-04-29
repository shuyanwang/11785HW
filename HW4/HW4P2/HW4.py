import os
import torch
import numpy as np
import torch.utils.data
import Levenshtein
import argparse

from torchinfo import summary

from utils.base import *

letter2index = {"<eos>": 0, "'": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8, "h": 9, "i": 10, "j": 11,
                "k": 12, "l": 13, "m": 14, "n": 15, "o": 16, "p": 17, "q": 18, "r": 19, "s": 20, "t": 21, "u": 22,
                "v": 23, "w": 24, "x": 25, "y": 26, "z": 27, " ": 28}

index2letter = {letter2index[key]: key for key in letter2index}

num_workers = 8


class ParamsHW4(Params):
    def __init__(self, B, lr, embedding_dim, attention_dim, dropout, device, layer_encoder, layer_decoder,
                 hidden_encoder,
                 hidden_decoder, schedule_int, decay, optimizer, max_epoch=20001, data_dir='/home/zongyuez/dldata/HW4'):
        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=dropout,
                         output_channels=len(index2letter),
                         data_dir=data_dir, device=device, input_dims=(40,))

        assert embedding_dim == self.attention_dim * 2

        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.layer_encoder = layer_encoder
        self.layer_decoder = layer_decoder
        self.hidden_encoder = hidden_encoder
        self.hidden_decoder = hidden_decoder
        self.schedule = schedule_int
        self.decay = decay
        self.optimizer = optimizer

        self.str = 'b' + str(self.B) + 'lr' + str(self.lr) + 's' + str(
                schedule_int) + 'decay' + str(decay) + optimizer + 'drop' + str(
                self.dropout) + 'le' + str(layer_encoder) + 'ld' + str(layer_decoder) + 'he' + str(
                hidden_encoder) + 'hd' + str(hidden_decoder) + 'emb' + str(embedding_dim) + 'att' + str(attention_dim)

    def __str__(self):
        return self.str


class DataSetHW4(torch.utils.data.Dataset):
    def __init__(self, X_path, Y_path=None):
        super().__init__()
        self.X = np.load(X_path)
        self.N = self.X.shape[0]

        if Y_path is not None:
            self.Y = np.load(Y_path, allow_pickle=True)

        print(X_path, self.__len__())

    def __getitem__(self, index):
        """

        :param index:
        :return: (T_in,40), Optional[(T_out,)]
        """
        if self.Y is not None:
            return torch.tensor(self.X[index].astype(np.float32)), torch.tensor(self.Y[index])
        return torch.tensor(self.X[index].astype(np.float32))

    def __len__(self):
        return self.N


def collate_train_val(data):
    """

    :param data: List of Tuple
    :return: pad_x, lengths_x, pad_y, lengths_y
    """
    x_lengths = []
    y_lengths = []
    x = []
    y = []
    for item in data:
        x.append(item[0])
        y.append(item[1])
        x_lengths.append(item[0].shape[0])
        y_lengths.append(item[1].shape[0])

    pad_x = nn.utils.rnn.pad_sequence(x, batch_first=True)
    pad_y = nn.utils.rnn.pad_sequence(y, batch_first=True)

    return pad_x, x_lengths, pad_y, y_lengths


def collate_test(data):
    """

    :param data:
    :return: pad_x, lengths_x
    """
    lengths = [x.shape[0] for x in data]

    return nn.utils.rnn.pad_sequence(data, batch_first=True), lengths


class HW4(Learning):
    def __init__(self, params: ParamsHW4, model):
        super().__init__(params, model, None, nn.CTCLoss)
        self.decoder = None
        optimizer = eval('torch.optim.' + params.optimizer)
        self.optimizer = optimizer(self.model.parameters(), lr=self.params.lr,
                                   weight_decay=self.params.decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.params.schedule, 0.5)
        print(str(self))

    def _load_train(self):
        train_set = DataSetHW4(os.path.join(self.params.data_dir, 'train.npy'),
                               os.path.join(self.params.data_dir, 'train_labels.npy'))
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers,
                                                        collate_fn=collate_train_val)

    def _load_valid(self):
        valid_set = DataSetHW4(os.path.join(self.params.data_dir, 'dev.npy'),
                               os.path.join(self.params.data_dir, 'dev_labels.npy'))

        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers,
                                                        collate_fn=collate_train_val)

    def _load_test(self):
        test_set = DataSetHW4(os.path.join(self.params.data_dir, 'test.npy'))

        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=self.params.B, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers,
                                                       collate_fn=collate_test)

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
                    b_string.append(index2letter[letter])
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
                    chars.append(index2letter[char])
            results.append(''.join(chars))

        return results

    def train(self, checkpoint_interval=5):
        # self._validate(0)
        summary_flag = True
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

                    if summary_flag:
                        summary(self.model, input_data=[x, lengths_x], depth=12,
                                device=self.params.device)
                        summary_flag = False

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

                    y_strs = HW4.to_str(y, lengths_y)
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
                        lengths = item[1]

                        # (T,N,C)
                        output, out_lengths = self.model(x, lengths)
                        results = self.decode(output, out_lengths)

                        for b in range(x.shape[0]):
                            f.write(str(i * self.params.B + b) + ',')
                            f.write(results[b])
                            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=32, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='Model19', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', default=10, type=int, help='Checkpoint interval')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--le', default=2, type=int)
    parser.add_argument('--ld', default=2, type=int)
    parser.add_argument('--he', default=1024, type=int)
    parser.add_argument('--hc', default=1024, type=int)
    parser.add_argument('--schedule', default=5, type=int)
    parser.add_argument('--decay', default=5e-5, type=float)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--embedding', default=256)
    parser.add_argument('--attention', default=128)

    args = parser.parse_args()

    params = ParamsHW4(B=args.batch, dropout=args.dropout, lr=args.lr,
                       device='cuda:' + args.gpu_id, layer_decoder=args.ld,
                       layer_encoder=args.le, hidden_encoder=args.he, hidden_decoder=args.hd,
                       schedule_int=args.schedule, decay=args.decay, optimizer=args.optimizer,
                       embedding_dim=args.embedding, attention_dim=args.attention)

    model = eval(args.model + '(params)')
    learner = HW4(params, model)
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
