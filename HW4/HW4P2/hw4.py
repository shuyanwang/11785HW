import os
import torch.utils.data
import Levenshtein
import argparse

from models import *

PAD_INDEX = 0  # CANNOT BE NEGATIVE, OTHERWISE EMBEDDING WOULD CAUSE ERROR
PRE_TRAIN_EPOCHS = 10


class ParamsHW4(Params):
    def __init__(self, B, lr, embedding_dim, attention_dim, dropout, device, layer_encoder,
                 hidden_encoder, hidden_decoder, schedule_int, decay, optimizer, clip,
                 forcing_tuple, data_dir, max_epoch=20001, plot=False, pretrain=False):
        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=dropout,
                         output_channels=len(index2letter),
                         data_dir=data_dir, device=device, input_dims=(40,))

        self.plot = plot
        self.forcing = eval(forcing_tuple)
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.layer_encoder = layer_encoder
        self.hidden_encoder = hidden_encoder
        self.hidden_decoder = hidden_decoder
        self.schedule = schedule_int
        self.decay = decay
        self.optimizer = optimizer
        self.pretrain = pretrain
        self.clip = clip

        assert embedding_dim == self.attention_dim * 2

        self.str = 'b' + str(self.B) + 'lr' + str(self.lr) + 's' + str(
                schedule_int) + 'decay' + str(decay) + optimizer + 'drop' + str(
                self.dropout) + 'le' + str(layer_encoder) + 'he' + str(
                hidden_encoder) + 'hd' + str(hidden_decoder) + 'emb' + str(
                embedding_dim) + 'att' + str(attention_dim) + 'forcing' + forcing_tuple + \
                   ('TOY' if 'simple' in data_dir else '') + ('pre' if pretrain else '') + (
                       'clip' if clip else '')

    def __str__(self):
        return self.str


class DataSetHW4(torch.utils.data.Dataset):
    def __init__(self, X_path, Y_path=None):
        super().__init__()
        self.X = np.load(X_path, allow_pickle=True)
        self.N = self.X.shape[0]
        self.Y = None

        if Y_path is not None:
            self.Y = np.load(Y_path, allow_pickle=True)

        print(X_path, self.__len__())

    def __getitem__(self, index):
        """

        :param index:
        :return: (T_in,40), Optional[(T_out,)]
        """
        if self.Y is not None:
            return torch.tensor(self.X[index], dtype=torch.float), torch.tensor(
                    self.Y[index], dtype=torch.long)
        return torch.tensor(self.X[index], dtype=torch.float)

    def __len__(self):
        return self.N


def collate_train_val(data):
    """

    :param data: List of Tuple
    :return: pad_x, x_lengths, pad_y, torch.as_tensor(y_lengths)
    """
    x_lengths = [x.shape[0] for (x, y) in data]
    y_lengths = [y.shape[0] for (x, y) in data]
    x_items = [x for (x, y) in data]
    y_items = [y for (x, y) in data]

    pad_x = nn.utils.rnn.pad_sequence(x_items, batch_first=True)
    pad_y = nn.utils.rnn.pad_sequence(y_items, batch_first=True, padding_value=PAD_INDEX)

    return pad_x, torch.as_tensor(x_lengths), pad_y, torch.as_tensor(y_lengths)


def collate_test(data):
    """

    :param data:
    :return: pad_x
    """

    x_lengths = [x.shape[0] for x in data]
    return nn.utils.rnn.pad_sequence(data, batch_first=True), x_lengths


class HW4(Learning):
    def __init__(self, params: ParamsHW4, model):
        super().__init__(params, model, None, None)
        self.decoder = None
        #### DO NOT USE IGNORE_INDEX
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(params.device)
        optimizer = eval('torch.optim.' + params.optimizer)
        self.optimizer = optimizer(self.model.parameters(), lr=self.params.lr,
                                   weight_decay=self.params.decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.params.schedule, 0.5)
        print(str(self))

    def forcing_p(self, epoch):
        if epoch < self.params.forcing[2]:
            return (self.params.forcing[1] - self.params.forcing[0]) * epoch / self.params.forcing[
                2] + self.params.forcing[0]
        return self.params.forcing[1]

        # return 0.9

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

    @staticmethod
    def decode(output, eos=False):
        """
        :param output: (B,o,T)
        :param eos: show <eos> at the end
        :return: [str] (B)
        """
        return HW4.to_str(torch.argmax(output, dim=1), eos)

    @staticmethod
    def to_str(y, eos=False):
        """
        :param y: (B,T)
        :param eos:
        :return: [str] (B)
        """
        results = []
        for b, y_b in enumerate(y):
            chars = []
            for char in y_b:
                char = char.item()
                if char == letter2index['<eos>']:
                    if eos:
                        chars.append('<eos>')
                    break
                chars.append(index2letter[char])

            # while len(chars) != 0 and chars[-1] == ' ':
            #     chars.pop(-1)

            results.append(''.join(chars))

        return results

    def train(self, checkpoint_interval=5):
        self._validate(self.init_epoch)
        if self.train_loader is None:
            self._load_train()

        # print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                total_loss = 0

                plot_index = np.random.randint(0, len(self.train_loader))

                for i, batch in enumerate(tqdm(self.train_loader)):
                    x = batch[0].to(self.device)
                    lengths_x = batch[1]
                    y = batch[2].to(self.device)  # (B,To)
                    lengths_y = batch[3]  # (B)

                    # (B,e,To)
                    output = self.model(x, lengths_x, gt=y, p_tf=self.forcing_p(epoch),
                                        plot=i == plot_index and self.params.plot,
                                        pretrain=self.params.pretrain and epoch < PRE_TRAIN_EPOCHS)

                    if i == plot_index:
                        y_strs = HW4.to_str(y)
                        out_strs = HW4.decode(output)
                        print()
                        print('Sample GT', y_strs[0])
                        print('Sample OG', out_strs[0])
                        print('Sample Training Distance',
                              Levenshtein.distance(y_strs[0], out_strs[0]))

                    loss = self.criterion(output, y)  # (B,To)

                    mask = torch.arange(y.shape[1]).unsqueeze(1) >= lengths_y.unsqueeze(0)
                    mask = mask.transpose(0, 1).to(self.device)  # (B,To)

                    loss[mask] = 0

                    loss_item = torch.sum(loss) / self.params.B

                    total_loss += loss_item.item()

                    self.optimizer.zero_grad()
                    loss_item.backward()

                    if self.params.clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                total_loss /= (i + 1)

                self.writer.add_scalar('Loss/Train', total_loss, epoch)

                print('epoch:', epoch, 'Training Loss:', "%.5f" % total_loss)

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
                total_dist = torch.zeros(1, device=self.device)
                plot_index = np.random.randint(0, len(self.valid_loader))

                for i, batch in enumerate(self.valid_loader):
                    x = batch[0].to(self.device)
                    x_lengths = batch[1]
                    y = batch[2]  # (B,To)

                    # (B,e,To)
                    output = self.model(x, x_lengths)

                    y_strs = HW4.to_str(y)
                    out_strs = HW4.decode(output)

                    if i == plot_index:
                        print()
                        print('Sample GT', y_strs[0])
                        print('Sample OG', out_strs[0])
                        print('Sample Valid Distance', Levenshtein.distance(y_strs[0], out_strs[0]))

                    for y_str, out_str in zip(y_strs, out_strs):
                        total_dist += Levenshtein.distance(y_str, out_str)

                dist_item = total_dist.item() / (i + 1) / self.params.B
                self.writer.add_scalar('Distance/Validation', dist_item, epoch)
                print('epoch:', epoch, 'Validation Distance:', dist_item)

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

                        output = self.model(x, lengths)
                        results = HW4.decode(output)

                        for b in range(x.shape[0]):
                            f.write(str(i * self.params.B + b) + ',')
                            f.write(results[b])
                            f.write('\n')


def main(args):
    params = ParamsHW4(B=args.batch, dropout=args.dropout, lr=args.lr, device=args.device,
                       layer_encoder=args.le, hidden_encoder=args.he, hidden_decoder=args.hd,
                       schedule_int=args.schedule, decay=args.decay, optimizer=args.optimizer,
                       embedding_dim=args.embedding, attention_dim=args.attention,
                       forcing_tuple=args.forcing, plot=args.plot, pretrain=args.pretrain,
                       clip=args.clip,
                       data_dir='C:\\DLData\\11785_data\\HW4' + (
                           '\\hw4p2_simple' if args.toy else ''))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=32, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default='Model1', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', default=1, type=int, help='Checkpoint interval')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--le', default=3, type=int)
    parser.add_argument('--he', default=256, type=int)
    parser.add_argument('--hd', default=512, type=int)
    parser.add_argument('--schedule', default=100, type=int)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--embedding', default=256, type=int)
    parser.add_argument('--attention', default=128, type=int)
    parser.add_argument('--forcing', default='(0.9,0.8,20)')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--clip', action='store_true')

    args = parser.parse_args()
    if args.toy:
        letter2index = {"<eos>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8,
                        "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16,
                        "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24,
                        "y": 25, " ": 26}
    else:
        letter2index = {"<eos>": 0, "'": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8,
                        "h": 9, "i": 10, "j": 11, "k": 12, "l": 13, "m": 14, "n": 15, "o": 16,
                        "p": 17, "q": 18, "r": 19, "s": 20, "t": 21, "u": 22, "v": 23, "w": 24,
                        "x": 25, "y": 26, "z": 27, " ": 28}

    index2letter = {letter2index[key]: key for key in letter2index}

    num_workers = 4
    main(args)

"""
--train
--clip
"""
