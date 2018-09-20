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
        self.h_init = torch.zeros(num_hidden)

        w_hx = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(mean=0, std=0.0001))
        w_hh = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(mean=0, std=0.0001))
        b_h = nn.Parameter(torch.Tensor(num_hidden).zero_())
        
        w_ph = nn.Parameter(torch.Tensor(num_classes, num_hidden).normal_(mean = 0, std = 0.0001))
        b_p = nn.Parameter(torch.Tensor(num_hidden).zero_())

        self.params = (w_hx, w_hh, b_h, w_ph, b_p)

    def forward(self, x):
        h_t = self.h_init
        tanh = nn.Tanh()
        w_hx, w_hh, b_h, w_ph, b_p = self.params

        for step in range(self.seq_length):
            h_t = tanh(w_hx @ x[step] + w_hh @ h_t + b_h)
        
        p_t = w_ph @ h_t + b_p

        return p_t
        
