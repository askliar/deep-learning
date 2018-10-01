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

        # use either embedding or one-hot encoding
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
        
        # used for passing previous hidden state during generation
        self.current_hidden = None

    # convert input to one hot vector
    def to_one_hot(self, input, size):
        one_hot = torch.zeros(*input.shape, size).to(self.device)
        indexing_tensor = input.unsqueeze(-1).long()
        batch_inputs = one_hot.scatter(2, indexing_tensor, 1)
        return batch_inputs

    def forward(self, x):
        # if single character is passed, convert to (1x1)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        
        # embed the input using embedding or one-hot encoding
        if self.embedding:
            encoded = self.encoder(x)
        else:
            encoded = self.to_one_hot(x, self.vocabulary_size)

        self.lstm.flatten_parameters()

        # if not training - we are generating, so, save hidden state for the next step
        # otherwise - don't save hidden
        if not self.training:
            # feed embedded input into lstm
            output, hidden = self.lstm(encoded, self.current_hidden)

            # reshape hidden to correct dimension 
            if len(hidden[0].shape) > 2:
                hidden = (hidden[0][:, -1, :].unsqueeze(1),
                          hidden[1][:, -1, :].unsqueeze(1))
            self.current_hidden = hidden
        else:
            output, hidden = self.lstm(encoded)
            self.current_hidden = None

        # apply linear layer to decode
        decoded = self.decoder(output)

        return decoded
