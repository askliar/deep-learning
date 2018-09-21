################################################################################
# MIT License
#
# Copyright (c) 2018
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

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length
        self.h_init = torch.zeros(num_hidden, 1)

        self.w_hx = nn.Parameter(nn.init.orthogonal_(torch.Tensor(
            num_hidden, input_dim).normal_(mean=0, std=0.0001)))
        self.w_hh = nn.Parameter(nn.init.orthogonal_(torch.Tensor(
            num_hidden, num_hidden).normal_(mean=0, std=0.0001)))
        self.b_h = nn.Parameter(torch.Tensor(num_hidden, 1).zero_())
        
        self.w_ph = nn.Parameter(torch.Tensor(
            num_classes, num_hidden).normal_(mean=0, std=0.0001))
        self.b_p = nn.Parameter(torch.Tensor(num_classes, 1).zero_())

    def forward(self, x):
        h_t = self.h_init
        tanh = nn.Tanh()

        for step in range(self.seq_length):
            h_t = tanh(self.w_hx @ x[:, step].t() + self.w_hh @ h_t + self.b_h)
           
        p_t = self.w_ph @ h_t + self.b_p

        return p_t.t()
        
