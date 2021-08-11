import random
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

"""
- DATA: we split each docs to 2 parts: 75% and 25%
- Model: we formulate this problem as seq2seq and mask prediction task, model consist of 2 components: encoder and decoder
    + Encoder, we use 2 bi-gru for learning local and global features
    + Decoder has 2 parts: the first reconstruct invert 75% docs (.i.e from token T to 0), the rest will be mask prediction with 50%
"""


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device,
                 num_layers=2,
                 bidirectional=True,
                 p=0.2,
                 batch_first=True,
                 finetune=True):

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.finetune = finetune
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.device = device
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # embedding for sentiment or load pretrain at here
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # the first rnn to learning sentiment present
        self.rnn1 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        # the second rnn to learning global represent of text, using pretrain bert as input
        self.rnn2 = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        self.dropout = nn.Dropout(p)

    def _init_hidden(self, batch_size):
        """
        Args:
            batch_size: int
        Returns:
            h (Tensor): the first hidden state of rnn
        """

        if self.bidirectional:
            h = torch.zeros((self.num_layers*2, batch_size, self.hidden_size))
        else:
            h = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        nn.init.xavier_normal_(h)

        return h.to(self.device)

    def forward(self, input_ids, attention_mask, x):
        """
        Args:
            input_ids, attention_mask: Tensor(batch_size, seq_len, embedding_dim)
            x: Tensor(batch_size, seq_len)
        Returns:
            ouput1, output2: Tensor(batch_size, seq_length, hidden_size)
            h1, h2: Tensor(D*num_layers, seq_lenght, hidden_size)
        """

        if self.finetune:
            last_hidden_states = self.bert(
                input_ids, attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                last_hidden_states = self.bert(
                    input_ids, attention_mask).last_hidden_state
        # `last_hidden_states` shape of [batch_size, seq_len, hidden_size of bert: 768]

        embedd_sentiment = self.dropout(self.embeddings(x))
        # sentiment `embedd_sentiment` shape of [batch_size, seq_len, embedding_dim]

        h = self._init_hidden(x.size(0))

        output1, h1 = self.rnn1(embedd_sentiment, h)
        output2, h2 = self.rnn2(last_hidden_states, h)
        return (output1, h1), (output2, h2)


class BahdanauAttention(nn.Module):
    def __init__(self, dec_dim: int, enc_dim: int, num_hiddens: int):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, num_hiddens, bias=False)
        self.W2 = nn.Linear(enc_dim, num_hiddens, bias=False)
        self.v = nn.Linear(num_hiddens, 1, False)

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            value (Tensor(batch size, seq_len, encoder hidden dimension): the hidden_state of tokens in encoder
            query (Tensor(batch size, 1, decoder hidden dimension)): the hidden state of decoder at time step t
        Returns:
            attention_weight (Tensor)
            context_vector (Tensor)
        """

        score = self.v(torch.tanh(self.W1(query) + self.W2(value)))
        # size of `score`: (batch_size, seq_len, 1)

        attention_weight = F.softmax(score.squeeze(-1), dim=1)
        # `attention` size of: (batch_size, seq_len)

        context_vector = torch.bmm(
            attention_weight.unsqueeze(1), value).squeeze(1)
        # `context_vector` size of: (batch_size, seq)
        return attention_weight, context_vector


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, encoder_output_dim,
                 num_layers=2,
                 bidirectional=True,
                 p=0.2,
                 batch_first=True,
                 finetune=True):

        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.finetune = finetune
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # embedding for sentiment or load pretrain at here
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.rnn1 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        self.rnn2 = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)
        self.dropout = nn.Dropout(p)

        self.attention1 = BahdanauAttention(dec_dim=hidden_size * self.D, enc_dim=encoder_output_dim,
                                            num_hiddens=hidden_size * self.D)
        self.attention2 = BahdanauAttention(dec_dim=hidden_size * self.D, enc_dim=encoder_output_dim,
                                            num_hiddens=hidden_size * self.D)

    def forward(self, x: Tensor, output1_encoder: Tensor, h1_encoder: Tensor, input_ids: Tensor,
                attention_mask: Tensor, output2_encoder: Tensor, h2_encoder: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        # `input_ids`, `attention_mask` size of [batch_size] -> [batch_size, 1]
        input_ids, attention_mask = input_ids.unsqueeze(
            1), attention_mask.unsqueeze(1)
        if self.finetune:
            last_hidden_states = self.bert(
                input_ids, attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                last_hidden_states = self.bert(
                    input_ids, attention_mask).last_hidden_state
        # `last_hidden_states` size of [batch_size, 1, hidden_size of bert: 768]

        # `x` size of batch_size -> (batch_size, 1) '1' denote seq_length
        x = x.unsqueeze(1)
        # sentiment embedding size of [batch_size, 1, embedding_dim]
        embedd_sentiment = self.dropout(self.embeddings(x))

        output1, h1 = self.rnn1(embedd_sentiment, h1_encoder)
        output2, h2 = self.rnn2(last_hidden_states, h2_encoder)

        _, context_vector1 = self.attention1(output1, output1_encoder)
        _, context_vector2 = self.attention2(output2, output2_encoder)

        output1 = torch.cat((output1.squeeze(1), context_vector1), dim=1)
        output2 = torch.cat((output2.squeeze(1), context_vector2), dim=1)
        output = torch.cat((output1, output2), dim=1)
        # `output` size of [batch_size, 2* (encoder_output_dim + hidden_size*2)]

        return output, h1, h2

class SentimentModel(nn.Module):
    def __init__(self, encoder, decoder1, decoder2, num_aspect):
        super(SentimentModel, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.relu = nn.ReLU()
        self.T = nn.Linear(2*(encoder.hidden_size*encoder.D +
                           decoder1.hidden_size * decoder1.D), num_aspect)
        self.Z1 = nn.Linear(num_aspect, decoder1.vocab_size)
        self.Z2 = nn.Linear(num_aspect, decoder2.vocab_size)

    def forward(self, source_input_ids, source_att_mask, source_inp, revert_source_input_ids, revert_source_att_mask,
                revert_source_inp, target_input_ids, target_att_mask, target_inp, teacher_forcing_ratio=0.5):
        # encoder input: x, input_ids, att_mask
        # decoder1 input: invert x, input_ids, att_mask
        # decoder2 input: x masked, input_ids, att_mask
        batch_size = target_inp.size(0)
        target_len = target_inp.size(1)
        revert_inp_len = revert_source_inp.size(1)

        output_reverts = torch.zeros(
            revert_inp_len, batch_size, self.decoder1.vocab_size)
        output_masks = torch.zeros(
            target_len, batch_size, self.decoder2.vocab_size)

        (output1, h1), (output2, h2) = self.encoder(
            source_input_ids, source_att_mask, source_inp)

        input_ = revert_source_inp[:, 0]
        input_ids = revert_source_input_ids[:, 0]
        attention_mask = revert_source_att_mask[:, 0]
        for i in range(1, revert_inp_len):
            output_revert, h1, h2 = self.decoder1(x=input_, output1_encoder=output1, h1_encoder=h1, input_ids=input_ids,
                                                  attention_mask=attention_mask, output2_encoder=output2, h2_encoder=h2)

            output_revert = self.T(output_revert)
            output_revert = self.Z1(self.relu(output_revert))

            output_reverts[i] = output_revert
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output_revert.argmax(1)
            input_ = revert_source_inp[:, i] if teacher_force else top1

            input_ids = revert_source_input_ids[:, i]
            attention_mask = revert_source_att_mask[:, i]

        (output1, h1), (output2, h2) = self.encoder(
            source_input_ids, source_att_mask, source_inp)

        input_ = target_inp[:, 0]
        input_ids = target_input_ids[:, 0]
        attention_mask = target_att_mask[:, 0]
        for i in range(1, target_len):
            output_mask, h1, h2 = self.decoder2(x=input_, output1_encoder=output1, h1_encoder=h1, input_ids=input_ids,
                                                attention_mask=attention_mask, output2_encoder=output2, h2_encoder=h2)

            output_mask = self.T(output_mask)
            output_mask = self.Z2(self.relu(output_mask))

            output_masks[i] = output_mask
            teacher_force1 = random.random() < teacher_forcing_ratio

            top1 = output_mask.argmax(1)
            input_ = target_inp[:, i] if teacher_force1 else top1

            input_ids = target_input_ids[:, i]
            attention_mask = target_att_mask[:, i]

        # output_reverts size of [revert seq_len, batch_size, decoder1.vocab_size]
        # output_mask size of [mask seq_len, batch_size, decoder2.vocab_size]
        return output_reverts, output_masks

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0