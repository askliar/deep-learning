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

import os
import time
from datetime import datetime
import argparse
import random

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from part3.dataset import TextDataset
from part3.model import TextGenerationModel

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

    _, predictions_indices = predictions.max(2)
    
    accuracy = (predictions_indices == targets).float().mean()

    return accuracy

def train(config):

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file,
                          seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = torch.nn.DataParallel(TextGenerationModel(batch_size=config.batch_size, 
                                seq_length=config.seq_length, 
                                vocabulary_size=dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden, 
                                lstm_num_layers=config.lstm_num_layers).to(device))

    # Setup the loss and optimizer
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    steps = []
    losses = []
    accuracies = []
    generated_sentences = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        steps.append(step)
        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets).to(device)
        
        outputs = model(batch_inputs)

        #######################################################
        # Add more code here ...
        #######################################################


        loss = loss_criterion(
            outputs.view(config.seq_length, -1, config.batch_size), 
            batch_targets
        )
        losses.append(loss.item())

        accuracy = calculate_accuracy(outputs, batch_targets)
        accuracies.append(accuracy.item())

        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04}/{:04}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                      datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                      config.train_steps, config.batch_size, examples_per_second,
                      accuracy, loss
            ))

        if step % config.sample_every == 0:
            model.eval()
            symbol = torch.randint(low=0, high=dataset.vocab_size, size=(1, )).long()
            generated_sequence = [symbol.item()]
            for i in range(config.seq_length-1):
                output = model(symbol)
                symbol = torch.max(output, 0)[1].unsqueeze(0)
                generated_sequence.append(symbol.item())

            generated_str = dataset.convert_to_string(generated_sequence)
            print(generated_str)
            generated_sentences.append(generated_str)
            model.train()

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
    with open('logs.txt', 'w') as f:
        f.write('Steps:\n')
        f.write(str(steps))
        f.write('\nLosses:\n')
        f.write(str(losses))
        f.write('\nAccuracies:\n')
        f.write(str(accuracies))
        f.write('\nGenerated sentences:\n')
        f.write("<EOF>".join(generated_sentences))
    torch.save(model, 'trained_model.pth')
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    
    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()
 
    # Train the model
    train(config)
