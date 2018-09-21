# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoder = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=lstm_num_hidden)
        self.lstm = nn.LSTM(input_size=lstm_num_hidden,
                            hidden_size=lstm_num_hidden, 
                            num_layers=lstm_num_layers)
        self.decoder = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size)

    def forward(self, x):
        encoded = self.encoder(x)
        if len(encoded.shape) < 3:
            encoded = encoded.unsqueeze(0)
        output, hidden = self.lstm(encoded)
        decoded = self.decoder(output)
        if len(x.shape) < 2:
            return decoded.squeeze()
        else:
            return decoded

