import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import constants.model_constants as mdl_const
import constants.main_constants as main_const
import constants.sql_constants as sql_const

class AggregatePredictor(nn.Module):
    def forward(self, input):
        embed = self.embed(input)
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = F.relu(self.convs1(cnn_x)).squeeze(3) # [(N,Co,W), ...]
        # cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)

        # GRU
        lstm_out, self.hidden = self.gru(cnn_x, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)

        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))

        # output
        logit = cnn_lstm_out

        return logit

    def __init__(self, embedding_layer, args):
        super().__init__()
        self.args = args

        # Word embedding layer
        self.embed = embedding_layer

        # CNN Layer
        self.convs1 = nn.Conv2d(in_channels=1, out_channels=mdl_const.kernel_num, kernel_size= (mdl_const.kernel_height, mdl_const.kernel_width), stride=1, padding=(mdl_const.kernel_height // 2, 0))

        # GRU Layer
        self.gru = nn.GRU(main_const.EMBEDDING_SIZE, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout)
        self.hidden = init_hidden(self.num_layers, args.batch_size)

        # linear
        self.hidden1 = nn.Linear(mdl_const.RNN_hidden_dim, mdl_const.RNN_hidden_dim // 2)
        self.hidden2 = nn.Linear(mdl_const.RNN_hidden_dim // 2, sql_const.AGG.__len__())

        # dropout
        self.dropout = nn.Dropout(main_const.dropout_prob)


class ConditionPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args


class SelectPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args

def init_hidden(self, num_layers, batch_size):
    # the first is the hidden h
    # the second is the cell  c
    if self.args.cuda is True:
        return Variable(torch.zeros(num_layers, batch_size, self.hidden_dim)).cuda()
    else:
        return Variable(torch.zeros(num_layers, batch_size, self.hidden_dim))
