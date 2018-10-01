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
                 lstm_num_hidden=256, lstm_num_layers=2,
                 dropout=0.0, embedding=False,
                 device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.hidden_size = lstm_num_hidden
        self.seq_length = seq_length
        self.embedding = embedding
        self.vocabulary_size = vocabulary_size
        self.device = torch.device(device)

        if embedding:
            embedding_dim = lstm_num_hidden
            self.encoder = nn.Embedding(num_embeddings=vocabulary_size, 
                                        embedding_dim=embedding_dim)
        else:
            embedding_dim = vocabulary_size

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            dropout=dropout)

        self.decoder = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size)
        
        self.current_hidden = None

    def to_one_hot(self, input, size):
        one_hot = torch.zeros(*input.shape, size).to(self.device)
        indexing_tensor = input.unsqueeze(-1).long()
        batch_inputs = one_hot.scatter(2, indexing_tensor, 1)
        return batch_inputs

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        
        if self.embedding:
            encoded = self.encoder(x)
        else:
            encoded = self.to_one_hot(x, self.vocabulary_size)

        self.lstm.flatten_parameters()

        if not self.training:
            output, hidden = self.lstm(encoded, self.current_hidden)
            if len(hidden[0].shape) > 2:
                hidden = (hidden[0][:, -1, :].unsqueeze(1),
                          hidden[1][:, -1, :].unsqueeze(1))
            self.current_hidden = hidden
        else:
            output, hidden = self.lstm(encoded)
            self.current_hidden = None

        decoded = self.decoder(output)

        return decoded
