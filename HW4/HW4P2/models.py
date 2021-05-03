from typing import Union
from torch.nn import functional
import numpy as np
from torch.nn.utils.rnn import PackedSequence

from utils.base import *
from hw4 import ParamsHW4
import torch
from torch import nn, Tensor


class LockedDropOut(nn.Module):
    def __init__(self, T_dim=1):
        super().__init__()
        self.T_dim = T_dim

    def forward(self, x, p):
        """
        :param x: (T,B,C) or (B,T,C); T dimension is specified
        :param p: probability
        :return:
        """
        if not self.training:
            return x
        if self.T_dim == 0:
            mask = torch.zeros((1, x.shape[1], x.shape[2]), requires_grad=False,
                               device=x.device).bernoulli_(1 - p)
        elif self.T_dim == 1:
            mask = torch.zeros((x.shape[0], 1, x.shape[2]), requires_grad=False,
                               device=x.device).bernoulli_(1 - p)
        else:
            raise ValueError

        mask /= (1 - p)
        mask = mask.expand_as(x)
        return mask * x


class Encoder(nn.Module, ABC):
    def __init__(self, param: ParamsHW4):
        super().__init__()
        self.param = param

    @abstractmethod
    def forward(self, x, lengths):
        """

        :param lengths:
        :param x: padded (B,Tin,40)
        :return: # (B,Te,H), lengths_out: hidden states of the last layer
        """
        pass


# class Decoder(nn.Module, ABC):
#     def __init__(self, param: ParamsHW4):
#         super().__init__()
#
#
#     @abstractmethod
#     def forward(self, k, v, encoded_lengths, gt, p_tf):
#         pass

class WeightDrop(nn.Module):
    def __init__(self, module, p=0.5):
        super(WeightDrop, self).__init__()
        self.p = p
        self.module = module
        self._setup()

    def null(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null

        w = getattr(self.module, 'weight_hh_l0')
        # del self.module._parameters['weight_hh_l0']
        self.module.register_parameter('weight_hh_l0' + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        raw_w = getattr(self.module, 'weight_hh_l0' + '_raw')
        w = nn.Parameter(functional.dropout(raw_w, p=self.p, training=self.training))
        setattr(self.module, 'weight_hh_l0', w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class PBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop):
        """

        :param input_dim: x.shape[2]
        :param hidden_dim:
        """
        super(PBLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, bidirectional=True,
                           batch_first=True)
        self.drop_layer = LockedDropOut()
        self.drop = drop

    def forward(self, x: PackedSequence):
        """

        :param x: (B,T,C) packed -> just a tensor
        :return: (B,T//2,H*2) packed
        """

        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        B, T, C = x.shape

        if T % 2 == 1:
            x = x[:, :-1, :]

        lengths = lengths // 2
        x = x.contiguous().view(B, T // 2, C * 2)

        x = self.drop_layer(x, self.drop)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]

        return x


class Encoder1(Encoder):
    def __init__(self, param):
        super(Encoder1, self).__init__(param)

        self.first = nn.LSTM(param.input_dims[0], param.hidden_encoder, batch_first=True,
                             bidirectional=True)
        rnn = []
        # 1 + 3

        for i in range(param.layer_encoder):
            rnn.append(PBLSTM(param.hidden_encoder * 2, param.hidden_encoder, self.param.dropout))

        self.rnn = nn.ModuleList(rnn)
        self.drop_layer = LockedDropOut()

        self.key_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)
        self.value_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)

    def forward(self, x: Tensor, lengths):
        """

        :param lengths:
        :param x: B,T,C
        :return: k ,v, lengths
        """
        x = self.drop_layer(x, self.param.dropout)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x = self.first(x)[0]
        for layer in self.rnn:
            x = layer(x)

        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

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

        return context, attention.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, param: ParamsHW4):
        super().__init__()
        self.param = param
        self.embedding = nn.Embedding(param.output_channels, param.embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=param.embedding_dim + param.attention_dim,
                                 hidden_size=param.hidden_decoder)
        self.lstm2 = nn.LSTMCell(input_size=param.hidden_decoder, hidden_size=param.attention_dim)

        self.attention = DotAttention(param)
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
        :return: context', hidden1', hidden2', attention (B,a)
        """
        input_word_embedding = self.embedding(input_words)

        h1, c1 = self.lstm1(torch.cat([input_word_embedding, context], dim=-1), hidden1)
        h2, c2 = self.lstm2(h1, hidden2)

        if key is not None:
            context, attention = self.attention(h2, key, value, mask)
        else:
            attention = None

        return context, (h1, c1), (h2, c2), attention

    def forward(self, k, v, encoded_lengths, gt, p_tf, B, plot=False):
        """

        :param k: (B,Td,a)
        :param v: (B,Td,a)
        :param encoded_lengths:
        :param gt: (B,To)
        :param p_tf:
        :param plot:
        :return: (B,vocab_size,To)
        """
        hidden1 = None
        hidden2 = None
        mask = None
        attention_to_plot = None

        To = 600 if gt is None else gt.shape[1]
        prediction_chars = torch.zeros(B, dtype=torch.long).to(
                self.param.device)  # <eol> at the beginning
        context = torch.zeros((B, self.param.attention_dim),
                              device=self.param.device)  # or zeros

        if k is not None:  # not pretrain
            _, Td, a = k.shape

            mask = torch.arange(Td).unsqueeze(0) >= torch.as_tensor(encoded_lengths).unsqueeze(1)
            mask = mask.to(self.param.device)
            attention_to_plot = torch.zeros((To, Td), device=self.param.device)

        predictions = torch.zeros((B, self.param.output_channels, To), device=self.param.device)

        for i in range(To):
            if gt is not None and torch.rand(1).item() < p_tf and i > 0:
                input_words = gt[:, i - 1]
            else:
                input_words = prediction_chars

            context, hidden1, hidden2, attention = self._forward_step(input_words, context,
                                                                      hidden1, hidden2, k, v, mask)

            if attention_to_plot is not None:
                attention_to_plot[i, :] = attention[0]

            prediction_raw = self.character_prob(
                    torch.cat([hidden2[0], context], dim=1))  # (B,vocab)
            prediction_chars = prediction_raw.argmax(dim=-1)
            predictions[:, :, i] = prediction_raw

        if plot and attention_to_plot is not None:
            plot_attention(attention_to_plot)

        return predictions
        # return torch.stack(predictions, dim=2)  # (B,vocab,To)


class Model1(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.encoder = Encoder1(param)
        self.decoder = Decoder(param)

        for weight in self.parameters():
            nn.init.uniform_(weight, -1 / np.sqrt(512), 1 / np.sqrt(512))

        nn.init.uniform_(self.decoder.embedding.weight, -0.1, 0.1)

    def forward(self, x, lengths, gt=None, p_tf=0.9, plot=False, pretrain=False):
        if pretrain:
            return self.decoder(None, None, None, gt, p_tf, plot=False, B=x.shape[0])
        key, value, encoder_len = self.encoder(x, lengths)
        return self.decoder(key, value, encoder_len, gt, p_tf, plot=plot, B=x.shape[0])


# class Decoder2(Decoder):
#     def __init__(self, param: ParamsHW4):
#         super().__init__(param)
#         self.embedding = nn.Embedding(param.output_channels, param.embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTMCell(input_size=param.embedding_dim + param.attention_dim,
#                                 hidden_size=param.attention_dim)
#
#         self.attention = DotAttention(param)
#         self.character_prob = nn.Linear(param.embedding_dim, param.output_channels)
#
#         self.character_prob.weight = self.embedding.weight
#
#     def _forward_step(self, input_words, context, hidden1, key, value, mask):
#         """
#
#         :param input_words: (B,)
#         :param context: (B,a)
#         :param hidden1: (h,c)
#         :param key: (B,Tout_e,a)
#         :param value: (B,Tout_e,a)
#         :param mask: (B,Toe)
#         :return: query (B,a), context', hidden1', attention (B,a)
#         """
#         input_word_embedding = self.embedding(input_words)
#
#         query, c1 = self.lstm(torch.cat([input_word_embedding, context], dim=-1), hidden1)
#         context, attention = self.attention(query, key, value, mask)
#
#         return query, context, (query, c1), attention
#
#     def forward(self, k, v, encoded_lengths, gt, p_tf, plot=False):
#         """
#
#         :param k: (B,Td,a)
#         :param v: (B,Td,a)
#         :param encoded_lengths:
#         :param gt: (B,To)
#         :param p_tf:
#         :param plot:
#         :return: (B,vocab_size,To)
#         """
#
#         B, Td, a = k.shape
#
#         mask = torch.arange(Td).unsqueeze(0) >= torch.as_tensor(encoded_lengths).unsqueeze(1)
#         mask = mask.to(self.param.device)
#
#         context = torch.zeros((B, a)).to(self.param.device)
#         hidden1 = None
#
#         # predictions = []
#         prediction_chars = torch.zeros(B, dtype=torch.long).to(
#                 self.param.device)  # <eol> at the beginning
#
#         To = 600 if gt is None else gt.shape[1]
#
#         attention_to_plot = torch.zeros((To, Td), device=self.param.device)
#
#         predictions = torch.zeros((B, self.param.output_channels, To), device=self.param.device)
#
#         for i in range(To):
#             if gt is not None and torch.rand(1).item() < p_tf and i > 0:
#                 input_words = gt[:, i - 1]
#             else:
#                 input_words = prediction_chars
#
#             query, context, hidden1, attention = self._forward_step(input_words, context,
#                                                                     hidden1, k, v,
#                                                                     mask)
#
#             attention_to_plot[i, :] = attention[0]
#
#             prediction_raw = self.character_prob(torch.cat([query, context], dim=1))  # (B,vocab)
#             prediction_chars = prediction_raw.argmax(dim=-1)
#             predictions[:, :, i] = prediction_raw
#
#         if plot:
#             plot_attention(attention_to_plot)
#
#         return predictions
#         # return torch.stack(predictions, dim=2)  # (B,vocab,To)
#
#
# class Model2(nn.Module):
#     def __init__(self, param):
#         super().__init__()
#         self.encoder = Encoder1(param)
#         self.decoder = Decoder2(param)
#
#         for weight in self.parameters():
#             nn.init.uniform_(weight, -1 / np.sqrt(512), 1 / np.sqrt(512))
#
#         nn.init.uniform_(self.decoder.embedding.weight, -0.1, 0.1)
#
#     def forward(self, x, gt=None, p_tf=0.9, plot=False):
#         key, value, encoder_len = self.encoder(x)
#         predictions = self.decoder(key, value, encoder_len, gt, p_tf, plot)
#         return predictions

class PBLSTMPad(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop):
        """

        :param input_dim: x.shape[2]
        :param hidden_dim:
        """
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, bidirectional=True,
                           batch_first=True)
        self.drop_layer = LockedDropOut()
        self.drop = drop

    def forward(self, x: Tensor):
        """

        :param x: (B,T,C) packed -> just a tensor
        :return: (B,T//2,H*2) packed
        """

        B, T, C = x.shape

        if T % 2 == 1:
            x = x[:, :-1, :]

        x = x.contiguous().view(B, T // 2, C * 2)
        x = self.drop_layer(x, self.drop)
        x = self.rnn(x)[0]
        return x


class Encoder3(Encoder):
    def __init__(self, param):
        super().__init__(param)

        self.first = nn.LSTM(param.input_dims[0], param.hidden_encoder, batch_first=True,
                             bidirectional=True)
        rnn = []
        # 1 + 3

        for i in range(param.layer_encoder):
            rnn.append(
                    PBLSTMPad(param.hidden_encoder * 2, param.hidden_encoder, self.param.dropout))

        self.rnn = nn.ModuleList(rnn)
        self.drop_layer = LockedDropOut()

        self.key_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)
        self.value_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)

    def forward(self, x: Tensor, lengths):
        """

        :param lengths:
        :param x: B,T,C
        :return: k ,v, lengths
        """
        x = self.drop_layer(x, self.param.dropout)
        x = self.first(x)[0]
        for layer in self.rnn:
            x = layer(x)
            lengths = lengths // 2

        k = self.key_network(x)  # (B,T//(2**num_layer),a)
        v = self.value_network(x)  # same as k

        return k, v, lengths


class Model3(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.encoder = Encoder3(param)
        self.decoder = Decoder(param)

        for weight in self.parameters():
            nn.init.uniform_(weight, -1 / np.sqrt(512), 1 / np.sqrt(512))

        nn.init.uniform_(self.decoder.embedding.weight, -0.1, 0.1)

    def forward(self, x, lengths, gt=None, p_tf=0.9, plot=False, pretrain=False):
        if pretrain:
            return self.decoder(None, None, None, gt, p_tf, plot=False, B=x.shape[0])
        key, value, encoder_len = self.encoder(x, lengths)
        return self.decoder(key, value, encoder_len, gt, p_tf, plot=plot, B=x.shape[0])


class PBLSTMW(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop):
        """

        :param input_dim: x.shape[2]
        :param hidden_dim:
        """
        super().__init__()
        self.rnn = WeightDrop(
                nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, bidirectional=True,
                        batch_first=True))
        self.drop_layer = LockedDropOut()
        self.drop = drop

    def forward(self, x: PackedSequence):
        """

        :param x: (B,T,C) packed -> just a tensor
        :return: (B,T//2,H*2) packed
        """

        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        B, T, C = x.shape

        if T % 2 == 1:
            x = x[:, :-1, :]

        lengths = lengths // 2
        x = x.contiguous().view(B, T // 2, C * 2)

        x = self.drop_layer(x, self.drop)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.rnn(x)[0]

        return x


class Encoder4(Encoder):
    def __init__(self, param):
        super().__init__(param)

        self.first = WeightDrop(nn.LSTM(param.input_dims[0], param.hidden_encoder, batch_first=True,
                                        bidirectional=True))
        rnn = []
        # 1 + 3

        for i in range(param.layer_encoder):
            rnn.append(PBLSTMW(param.hidden_encoder * 2, param.hidden_encoder, self.param.dropout))

        self.rnn = nn.ModuleList(rnn)
        self.drop_layer = LockedDropOut()

        self.key_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)
        self.value_network = nn.Linear(param.hidden_encoder * 2, param.attention_dim)

    def forward(self, x: Tensor, lengths):
        """

        :param lengths:
        :param x: B,T,C
        :return: k ,v, lengths
        """
        x = self.drop_layer(x, self.param.dropout)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x = self.first(x)[0]
        for layer in self.rnn:
            x = layer(x)

        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        k = self.key_network(x)  # (B,T//(2**num_layer),a)
        v = self.value_network(x)  # same as k

        return k, v, lengths


class Model4(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.encoder = Encoder4(param)
        self.decoder = Decoder(param)

        for weight in self.parameters():
            nn.init.uniform_(weight, -1 / np.sqrt(512), 1 / np.sqrt(512))

        nn.init.uniform_(self.decoder.embedding.weight, -0.1, 0.1)

    def forward(self, x, lengths, gt=None, p_tf=0.9, plot=False, pretrain=False):
        if pretrain:
            return self.decoder(None, None, None, gt, p_tf, plot=False, B=x.shape[0])
        key, value, encoder_len = self.encoder(x, lengths)
        return self.decoder(key, value, encoder_len, gt, p_tf, plot=plot, B=x.shape[0])
