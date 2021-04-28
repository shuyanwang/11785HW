import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
from torch.utils import data


device = torch.device(0)
np.random.seed(5111785)
torch.manual_seed(5111785)

LETTER_LIST = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '<eos>']


# %%

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    '''
    pass


def transform_letter_to_index(raw_transcripts):
    '''
    Transforms text input to numerical input by converting each letter
    to its corresponding index from letter_list

    Args:
        raw_transcripts: Raw text transcripts with the shape of (N, )

    Return:
        transcripts: Converted index-format transcripts. This would be a list with a length of N
    '''
    pass


# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)

# %%

# Load the training, validation and testing data
train_data = np.load('train.npy', allow_pickle=True, encoding='bytes')
valid_data = np.load('dev.npy', allow_pickle=True, encoding='bytes')
test_data = np.load('test.npy', allow_pickle=True, encoding='bytes')

# Load the training, validation raw text transcripts
raw_train_transcript = np.load('train_transcripts.npy', allow_pickle=True, encoding='bytes')
raw_valid_transcript = np.load('dev_transcripts.npy', allow_pickle=True, encoding='bytes')


# TODO: Convert the raw text transcripts into indexes
# train_transcript =
# valid_transcript =

# %%

class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # For testing set, return only x
        if self.Y == None:
            return torch.tensor(self.X[index].astype(np.float32))
        # For training and validation set, return x and y
        else:
            return torch.tensor(self.X[index].astype(np.float32)), torch.tensor(self.Y[index])


def collate_train_val(data):
    """
    Return:
        pad_x: the padded x (training/validation speech data)
        pad_y: the padded y (text labels - transcripts)
        x_len: the length of x
        y_len: the length of y
    """
    pass


def collate_test(data):
    """
    Return:
        pad_x: the padded x (testing speech data)
        x_len: the length of x
    """
    pass


# %%

# Create datasets
train_dataset =
valid_dataset =
test_dataset =

# Create data loaders
train_loader =
valid_loader =


# %%

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.
    '''

    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True,
                             batch_first=True)

    def forward(self, x):
        pass


# %%

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.
    Key and value are linear projections of the output from pBLSTM network for the laster.
    '''

    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM at the very bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, bidirectional=True,
                            batch_first=True)

        # TODO: Define the blocks of pBLSTMs
        # ...

        # The linear transformation for producing Key and Value for attention
        # Since you are using bidirectional LSTM, be careful about the size of hidden dimension
        self.key_network =
        self.value_network =

    def forward(self, x, x_len):
        # Pass through the first LSTM at the very bottom
        packed_sequence = rnn_utils.pack_padded_sequence(x, x_len, enforce_sorted=False, batch_first=True)
        packed_out, _ = self.lstm(packed_sequence)

        # TODO: Pass through the pBLSTM blocks
        # ...

        # Unpack the sequence and get the Key and Value for attention

        # return key, value, unpacked_x_len


# %%

def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        pass


# %%

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the seond LSTMCell can be used as query for calculating attention.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''

    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=letter2index['<eos>'])
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)

        self.attention = Attention()
        self.vocab_size = vocab_size
        self.character_prob = nn.Linear(2 * key_value_size, vocab_size)
        self.key_value_size = key_value_size

    def forward(self, key, value, encoder_len, y=None, mode='train'):
        '''
        Args:
            key :(B, T, key_value_size) - Output of the Encoder Key projection layer
            value: (B, T, key_value_size) - Output of the Encoder Value projection layer
            y: (T, text_len) - Batch input of text with text_length
            mode: Train or eval mode
        Return:
            predictions: the character perdiction probability
        '''

        B, key_seq_max_len, key_value_size = key.shape

        if mode == 'train':
            max_len = y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # TODO: Create the attention mask here (outside the for loop rather than inside) to aviod repetition
        # ...

        predictions = []
        prediction = torch.zeros(B, 1).to(device)
        hidden_states = [None, None]

        # TODO: Initialize the context. Be careful here
        # context =

        for i in range(max_len):
            if mode == 'train':
                # TODO: Implement (1) Teacher Forcing and (2) Gumble Noise techniques here
                # ...
                char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            lstm1_hidden = hidden_states[0][0]
            hidden_states[1] = self.lstm2(lstm1_hidden, hidden_states[1])
            output = hidden_states[1][0]

            # TODO: Compute attention from the output of the second LSTM Cell
            # ...

            output_context = torch.cat([output, context], dim=1)
            prediction = self.character_prob(output_context)
            predictions.append(prediction.unsqueeze(1))
        return torch.cat(predictions, dim=1)


# %%

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''

    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size=key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size=key_value_size)

    def forward(self, x, x_len, y=None, mode='train'):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions = self.decoder(key, value, encoder_len, y=y, mode=mode)
        return predictions


# %%

def train(model, train_loader, criterion, optimizer, mode):
    model.train()
    running_loss = 0

    # 0) Iterate through your data loader
    # 1) Set the inputs to the device.

    # 2) Pass your inputs, and length of speech into the model.

    # 3) Generate a mask based on the lengths of the text
    #    Ensure the mask is on the device and is the correct shape.

    # 4. Calculate the loss and mask it to remove the padding part

    # 5. Backward on the masked loss

    # 6. Optional: Use torch.nn.utils.clip_grad_norm(model.parameters(), 2) to clip the gradie

    # 7. Take a step with your optimizer

    # 8. print the statistic (loss, edit distance and etc.) for analysis


def val(model, valid_loader):
    model.eval()
    pass


# %%

# TODO: Define your model and put it on the device here
# ...

n_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=)
criterion = nn.CrossEntropyLoss(reduction='none')
mode = 'train'

for epoch in range(n_epochs):
    train(model, train_loader, criterion, optimizer, mode)
    val(model, valid_loader)
