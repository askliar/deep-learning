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

import argparse
import time
import pickle
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################


def calculate_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
    Returns:
    accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    _, predictions_indices = predictions.max(1)

    accuracy = (predictions_indices == targets).float().mean()

    return accuracy

def to_one_hot(input):
    one_hot = torch.zeros(*input.shape, 10)
    indexing_tensor = input.unsqueeze(-1).long()
    batch_inputs = one_hot.scatter(2, indexing_tensor, 1)
    return batch_inputs

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    assert config.input_dim in (1, 10)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    accuracies = []

    filename = f'{config.model_type}_{input_length}.pkl'

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(seq_length=config.input_length,
                            input_dim=config.input_dim, 
                            num_hidden=config.num_hidden,
                            num_classes=config.num_classes,
                            batch_size=config.batch_size,
                            device=config.device)
    else:
        model = LSTM(seq_length=config.input_length,
                        input_dim=config.input_dim,
                        num_hidden=config.num_hidden,
                        num_classes=config.num_classes,
                        batch_size=config.batch_size,
                        device=config.device)
    model = model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    model.train()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        if config.input_dim == 10 and len(batch_inputs.shape) < 3:
            batch_inputs = to_one_hot(batch_inputs)
        else:
            batch_inputs = batch_inputs.unsqueeze(2)

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        outputs = model(batch_inputs)

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        loss = loss_criterion(outputs, batch_targets)
        accuracy = calculate_accuracy(outputs, batch_targets)

        accuracies.append(accuracy.item())

        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    pickle.dump(accuracies, open(f'{filename}.p', 'wb') )
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
