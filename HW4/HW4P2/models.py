from typing import List
import numpy as np
from utils.base import *
from HW4 import ParamsHW4
import torch
from torch import nn, Tensor


class Encoder(nn.Module, ABC):
    def __init__(self, params: ParamsHW4):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths):
        """

        :param x: padded (B,Tin,40)
        :param lengths: (B,) list
        :return: # (B,Te,H), lengths_out: hidden states of the last layer
        """
        pass


class Decoder(nn.Module, ABC):
    def __init__(self, params: ParamsHW4):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, k, v, encoded_lengths, gt, p_tf):
        pass


class PBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """

        :param input_dim: x.shape[2]
        :param hidden_dim:
        """
        super(PBLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=1,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, x: Tensor, lengths: List):
        """

        :param x: (B,T,C) padded -> just a tensor
        :param lengths
        :return: (B,T//2,H*2), lengths
        """
        B, T, C = x.shape

        if T // 2 == 1:
            x = x[:, :-1, :]

        x = x.contiguous().view(B, T // 2, C * 2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        lengths = [length // 2 for length in lengths]
        return x, lengths


class Encoder1(Encoder):
    def __init__(self, params):
        super(Encoder1, self).__init__(params)

        self.first = nn.LSTM(params.input_dims, params.hidden_encoder, batch_first=True)
        rnn = []
        # 1 + 3

        for i in range(params.layer_encoder):
            rnn.append(PBLSTM(params.hidden_encoder * 2, params.hidden_encoder))

        self.rnn = nn.ModuleList(rnn)

        self.key_network = nn.Linear(params.hidden_encoder * 2, params.attention_dim)
        self.value_network = nn.Linear(params.hidden_encoder * 2, params.attention_dim)

    def forward(self, x: torch.Tensor, lengths):
        """

        :param x: B,T,C
        :param lengths: List
        :return: k ,v, lengths
        """
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        x = self.first(x)[0]

        for layer in self.rnn:
            x, lengths = layer(x, lengths)

        k = self.key_network(x)  # (B,T//(2**num_layer),a)
        v = self.value_network(x)  # same as k

        return k, v, lengths


class Attention(nn.Module, ABC):
    def __init__(self, param: ParamsHW4):
        super(Attention, self).__init__()
        self.param = param

    @abstractmethod
    def forward(self, query, key, value, mask):
        pass


class DotAttention(Attention):
    """
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    """

    def __init__(self, param):
        super(DotAttention, self).__init__(param)

    def forward(self, query, key, value, mask):
        """

        :param query: (B,a)
        :param key: (B,Tout_e,a)
        :param value: (B,Tout_e,a)
        :param mask: (B,Toe)
        :return: context, attention (for visualization)
        """

        query = torch.unsqueeze(query, -1)  # (B,a,1)
        energy = torch.bmm(key, query).squeeze(-1)  # (B,Toe)

        energy = torch.masked_fill(energy, mask, -1e9)

        attention = torch.softmax(energy, dim=1).unsqueeze(1)  # (B,1,Toe)

        context = torch.bmm(attention, value).squeeze(1)  # (B,a)

        return context, attention


class Decoder1(Decoder):

    def __init__(self, param: ParamsHW4):
        super(Decoder1, self).__init__(param)
        self.embedding = nn.Embedding(param.output_channels, param.embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=param.embedding_dim + param.attention_dim,
                                 hidden_size=param.hidden_decoder)
        self.lstm2 = nn.LSTMCell(input_size=param.hidden_decoder, hidden_size=param.attention_dim)

        self.attention = Attention(param)
        self.character_prob = nn.Linear(param.embedding_dim, param.output_channels)

        self.character_prob.weight = self.embedding.weight

    def _forward_step(self, input_words, context, hidden1, hidden2, key, value, mask):
        """

        :param input_words: (B,)
        :param context: (B,a)
        :param hidden1: (h,c)
        :param hidden2: (h,c)
        :param key: (B,Tout_e,a)
        :param value: (B,Tout_e,a)
        :param mask: (B,Toe)
        :return: query (B,a), context', hidden1', hidden2'
        """
        input_word_embedding = self.embedding(input_words)

        x, hidden1 = self.lstm1(torch.cat([input_word_embedding, context], dim=-1), hidden1)
        query, hidden2 = self.lstm2(x, hidden2)
        context = self.attention(query, key, value, mask)

        return query, context, hidden1, hidden2

    def forward(self, k, v, encoded_lengths, gt, p_tf):
        """

        :param k: (B,Td,a)
        :param v: (B,Td,a)
        :param encoded_lengths:
        :param gt: (B,To)
        :param p_tf:
        :return: (B,To)
        """

        B, T, a = k.shape

        mask = torch.arange(T).unsqueeze(0) >= torch.as_tensor(encoded_lengths).unsqueeze(1)
        mask.to(self.param.device)

        context = None
        hidden1 = None
        hidden2 = None

        predictions = []
        prediction_chars = torch.zeros(B, dtype=torch.long).to(
                self.params.device)  # <eol> at the beginning

        if gt is None:
            max_len = 600
            gt_embeddings = None
        else:
            max_len = gt.shape[1]
            gt_embeddings = self.embedding(gt)  # (B,To,e)

        for i in range(max_len):
            if gt is not None:
                if torch.rand(1).item() < p_tf:
                    input_words = gt_embeddings[:, i, :]
                else:
                    input_words = prediction_chars
            else:
                input_words = prediction_chars

            query, context, hidden1, hidden2 = self._forward_step(input_words, context, hidden1,
                                                                  hidden2, k[:, i, :], v[:, i, :],
                                                                  mask)

            prediction_raw = self.character_prob(torch.cat([query, context], dim=1))  # (B,e)
            prediction_chars = prediction_raw.argmax(dim=-1)
            predictions.append(prediction_raw)

        return torch.cat(predictions, dim=1)


class Seq2Seq1(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.encoder = Encoder1(param)
        self.decoder = Decoder1(param)

    def forward(self, x, x_len, gt=None, p_tf=0.9):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions = self.decoder(key, value, encoder_len, gt, p_tf)
        return predictions
