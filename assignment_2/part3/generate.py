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
from utilities import *

def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file,
                          seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size,
                             num_workers=1)

    dropout = 1.0 - config.dropout_keep_prob

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                vocabulary_size=dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                dropout=dropout, 
                                embedding=config.embedding,
                                device=config.device).to(device)

    model.load_state_dict(torch.load(os.path.join(config.model_path, config.model_name), map_location=config.device))
    model.eval()

    if config.generation == 'start':
        model.current_hidden = None
        generated_sequence = []
        # set random character as the first input to the generating model
        symbol = torch.randint(
            low=0, high=dataset.vocab_size, size=(1, )).long().to(device)
        generated_sequence.append(symbol.item())
    # feed specified input into the model to complete it afterwards
    elif config.generation == 'complete':
        model.current_hidden = None
        generated_sequence = dataset.convert_to_ix(config.input_text)
        input = torch.Tensor(generated_sequence).to(device)
        output = model(input)
        # set last character as the first input to the generating model 
        symbol = input[-1].unsqueeze(0).to(device)
    
    # generate seq_length of text and set each generated character as the new input
    for i in range(config.generation_seq_length-1):
        output = model(symbol).squeeze()
        symbol = sample_single(output, sampling=config.sampling, 
                                temperature=config.temperature).to(device)
        generated_sequence.append(symbol.item())

    generated_str = dataset.convert_to_string(generated_sequence)
    print(generated_str)
    return generated_str
    

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int,
                        default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int,
                        default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default="cpu",
                        help="Training device 'cpu' or 'cuda:0'")

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float,
                        default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Learning rate weight decay fraction')
    parser.add_argument('--learning_rate_step', type=int,
                        default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float,
                        default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int,
                        default=15000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str,
                        default="../summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')

    parser.add_argument('--save_every', type=int, default=10,
                        help='How often to save the model and logs.')
    parser.add_argument('--model_path', type=str,
                        default='../checkpoints/', help='Output path for saving model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['adam', 'rmsprop'],
                        help='Optimizer to use for training.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model to load from checkpoints.')

    parser.add_argument('--sampling', type=str, default='greedy',
                        choices=['greedy', 'random'], help='Sampling to user.')
    parser.add_argument('--temperature', type=float,
                        default=1.0, help='Temperature for random sampling')
    parser.add_argument('--embedding', type=bool, default=False,
                        help='Whether to use embedding instead of one-hot encoding.')

    parser.add_argument('--generation', type=str, default='start', choices=['start', 'complete'],
                        help='Whether to generate text from a random first letter or complete sentence.')
    parser.add_argument('--generation_seq_length', type=int,
                        default=30, help='Length of a generated sequence')
    parser.add_argument('--input_text', type=str, default=' ',
                        help='Text to finish using pre-trained lstm.')

    config = parser.parse_args()

    if not os.path.exists(config.summary_path):
        os.makedirs(config.summary_path)

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    train(config)
