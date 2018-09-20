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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        
        self.seq_length = seq_length
        self.step_params = []
        self.h_init = torch.zeros(num_hidden)
        self.c_init = torch.zeros(num_hidden)

        for step in range(seq_length):
            w_gx = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            w_gh = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            b_g = nn.Parameter(torch.Tensor(num_hidden).zero_())

            w_ix = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            w_ih = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            b_i = nn.Parameter(torch.Tensor(num_hidden).zero_())

            w_fx = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            w_fh = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            b_f = nn.Parameter(torch.Tensor(num_hidden).zero_())

            w_ox = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            w_oh = nn.Parameter(torch.Tensor(
                num_hidden, input_dim).normal_(mean=0, std=0.0001))
            b_o = nn.Parameter(torch.Tensor(num_hidden).zero_())

            self.step_params.append(
                (w_gx, w_gh, b_g, w_ix, w_ih, b_i, w_fx, w_fh, b_f, w_ox, w_oh, b_o))

        w_ph = nn.Parameter(torch.Tensor(
            num_classes, num_hidden).normal_(mean=0, std=0.0001))
        b_p = nn.Parameter(torch.Tensor(num_hidden).zero_())

        self.output_params = (w_ph, b_p)

    def forward(self, x):
        h_t = self.h_init
        c_t = self.c_init

        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()

        for step in range(self.seq_length):
            w_gx, w_gh, b_g, w_ix, w_ih, b_i, w_fx, w_fh, b_f, w_ox, w_oh, b_o = self.step_params[step]
            g_t = tanh(w_gx @ x[step] + w_gh @ h_t + b_g)
            i_t = sigmoid(w_ix @ x[step] + w_ih @ h_t + b_i)
            f_t = sigmoid(w_fx @ x[step] + w_fh @ h_t + b_f)
            o_t = sigmoid(w_ox @ x[step] + w_oh @ h_t + b_o)

            c_t = g_t * i_t + c_t * f_t
            h_t = tanh(c_t) * o_t

        w_ph, b_p = self.output_params
        p_t = w_ph @ h_t + b_p
