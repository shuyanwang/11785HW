# %%

from torch.utils.data.dataset import T_co
# % matplotlib inline

import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation, array_to_str

SEQ_LENGTH = 60

device = torch.device(0)

# %%

# load all that we need

dataset = np.load('../dataset/wiki.train.npy', allow_pickle=True)  # [[int,...],...]
devset = np.load('../dataset/wiki.valid.npy', allow_pickle=True)  # [[int,...],...]
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')  # [str,...]


# %%

class LanguageModelSet(Dataset):

    def __init__(self, data_loaded):
        super().__init__()
        data = torch.from_numpy(np.concatenate(data_loaded))

        self.len = (data.shape[0] - 1) // SEQ_LENGTH

        self.input = torch.zeros((self.len, SEQ_LENGTH), dtype=torch.long)
        self.target = torch.zeros_like(self.input)

        for i in range(self.len):
            self.input[i] = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
            self.target[i] = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]

    def __getitem__(self, index) -> T_co:
        return self.input[index], self.target[index]

    def __len__(self):
        return self.len


# data loader

class LanguageModelDataLoader(DataLoader):
    """
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        if isinstance(dataset, LanguageModelSet):
            super(LanguageModelDataLoader, self).__init__(dataset, batch_size,
                                                          shuffle)
        else:
            super(LanguageModelDataLoader, self).__init__(LanguageModelSet(dataset), batch_size,
                                                          shuffle)


#

# class LanguageModelDataLoader(DataLoader):
#     """
#     """
#
#     def __init__(self, dataset, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         data = torch.from_numpy(np.concatenate(self.dataset))
#         self.len = (data.shape[0] - 1) // SEQ_LENGTH
#         self.input = torch.zeros((self.len, SEQ_LENGTH), dtype=torch.long)
#         self.target = torch.zeros_like(self.input)
#
#         for i in range(self.len):
#             self.input[i] = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
#             self.target[i] = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
#
#         for batch in range(self.len // self.batch_size):
#             yield (self.input[batch * self.batch_size:(batch + 1) * self.batch_size, :],
#                    self.target[batch * self.batch_size:(batch + 1) * self.batch_size, :])


# %%


class LockedDropOut(nn.Module):
    def __init__(self, p, batch_dim=1):
        super().__init__()
        self.keep = 1 - p
        self.batch_dim = batch_dim

    def forward(self, x):
        """
        :param x: (B,T,C)
        :return:
        """
        if not self.training:
            return x
        if self.batch_dim == 0:
            mask = torch.zeros((1, x.shape[1], x.shape[2]), requires_grad=False,
                               device=x.device).bernoulli_(self.keep)
        elif self.batch_dim == 1:
            mask = torch.zeros((x.shape[0], 1, x.shape[2]), requires_grad=False,
                               device=x.device).bernoulli_(self.keep)
        else:
            raise ValueError

        mask /= self.keep
        mask = mask.expand_as(x)
        return mask * x


# class LockedDropOut(nn.Module):
#     def __init__(self, p):
#         super(LockedDropOut, self).__init__()
#         self.d = nn.Dropout(p)
#
#     def forward(self, x):
#         return self.d(x)

EMBEDDING_SIZE = 400
HIDDEN_SIZE = 1150


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.d0 = LockedDropOut(0.65)
        self.r1 = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
        self.d1 = LockedDropOut(0.3)
        self.r2 = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE)
        self.d2 = LockedDropOut(0.3)
        self.r3 = nn.LSTM(HIDDEN_SIZE, EMBEDDING_SIZE)
        self.d3 = LockedDropOut(0.4)

        ####

        # self.d0 = nn.Identity()
        # self.d1 = nn.Identity()
        # self.d2 = nn.Identity()
        # self.d3 = nn.Identity()

        ####

        self.linear = nn.Linear(400, vocab_size)
        self.linear.weight = self.embedding.weight

        for weight in self.parameters():
            nn.init.uniform_(weight, -1 / np.sqrt(1150), 1 / np.sqrt(1150))

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, x):
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        # x: (B,SEQ)

        x = self.embedding(x)

        # (B,T,EMBEDDING)
        x = torch.transpose(x, 0, 1)
        # (T,B,Embedding)

        x = self.r1(self.d0(x))[0]
        x = self.r2(self.d1(x))[0]
        x = self.r3(self.d2(x))[0]
        x = self.d3(x)
        x = self.linear(x)  # (T,B,E)

        x = torch.transpose(x, 0, 1)
        return torch.transpose(x, 1, 2)  # B, ENCODING, SEQ


# %%

# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model.cuda()
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def train(self):
        self.model.train()  # set to training mode
        epoch_loss = 0
        batch_num = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        output = self.model(inputs.to(device))

        loss = self.criterion(output, targets.to(device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test(self):
        # don't change these
        self.model.eval()  # set to eval mode

        print(array_to_str(fixtures_pred['inp'], vocab))

        predictions = TestLanguageModel.prediction(fixtures_pred['inp'],
                                                   self.model)  # get predictions
        self.predictions.append(predictions)

        # ####
        #
        # x = next(iter(self.loader))[0]
        # gx = TestLanguageModel.generation(x, 10, self.model)
        #
        # ####

        generated_logits = TestLanguageModel.generation(fixtures_gen, 10,
                                                        self.model)  # generated predictions for
        # 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)

        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)

        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'],
                                                        self.model)  # get predictions
        self.predictions_test.append(predictions_test)

        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
                   model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)),
                self.predictions[-1])
        np.save(
                os.path.join('experiments', self.run_id,
                             'predictions-test-{}.npy'.format(self.epochs)),
                self.predictions_test[-1])
        np.save(
                os.path.join('experiments', self.run_id,
                             'generated_logits-{}.npy'.format(self.epochs)),
                self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id,
                             'generated_logits-test-{}.npy'.format(self.epochs)),
                self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)),
                  'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id,
                               'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# %%

class TestLanguageModel:
    @staticmethod
    def prediction(inp, model):
        inp = torch.from_numpy(inp)
        inp = inp.to(device)

        return model(inp).cpu().detach().numpy()

    @staticmethod
    def generation(inp, forward, model):
        """

            Generate a sequence of words given a starting sequence.
            :param model:
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """

        inp = torch.from_numpy(inp)  # (B,S)
        inp = inp.to(device)

        result = torch.zeros((inp.shape[0], forward), device=inp.device, dtype=torch.long)

        # res = model(inp)[:, :, -1]
        # out = model(inp)

        result[:, 0] = torch.argmax(model(inp)[:, :, -1], 1)  # (B,)
        for i in range(1, forward):
            inp = torch.cat((inp, torch.unsqueeze(result[:, i - 1], 1)), dim=1)
            result[:, i] = torch.argmax(model(inp)[:, :, -1], 1)

        return result.cpu().detach().numpy()


# %%

NUM_EPOCHS = 10
BATCH_SIZE = 32

# %%

run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)

# %%

model = LanguageModel(vocab.shape[0])
dataset_torch = LanguageModelSet(dataset)
loader = LanguageModelDataLoader(dataset=dataset_torch, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)

# print(array_to_str(fixtures_pred['inp'][0], vocab))

# %%

best_nll = 1e30
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch " + str(
                epoch) + " with NLL: " + str(best_nll))
        trainer.save()

# %%

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

# %%

# see generated output
print(trainer.generated[-1])  # get last generated output
