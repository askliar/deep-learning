"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import os
from mlp_pytorch import MLP
import torch
import torch.nn as nn
import cifar10_utils

# Commented for running on display-less systems like surfsara
# import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
# DNN_DROPOUTS_DEFAULT = '0.2'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
# OPTIMIZER_DEFAULT = 'SGD'
# WEIGHT_DECAY_DEFAULT = 0.0005
# MOMENTUM_DEFAULT = 0.9

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# Specify device on which computations will be made
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Return optimizer based on the passed flag
# def get_optimizer(optimizer, mlp, lr, weight_decay=0.0, momentum=0.0):
#     if optimizer == 'SGD':
#         return torch.optim.SGD(mlp.parameters(), lr, weight_decay=weight_decay, momentum=momentum)
#     elif optimizer == 'Adam': 
#         return torch.optim.Adam(mlp.parameters(), lr, weight_decay=weight_decay)
#     elif optimizer == 'Adagrad': 
#         return torch.optim.Adagrad(mlp.parameters(), lr, weight_decay=weight_decay)
    

def accuracy(predictions, targets):
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

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    _, predictions_indices = predictions.max(1)
    _, targets_indices = targets.max(1)

    accuracy = (predictions_indices == targets_indices).sum().float() / predictions_indices.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Split on ' ' (space), otherwise can't pass through qsub 
    # Get number of units in each hidden layer specified in the string such as 100 100
    # if FLAGS.dnn_hidden_units:
    #     dnn_hidden_units = FLAGS.dnn_hidden_units.split(" ")
    #     dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    # else:
    #     dnn_hidden_units = []

    #  Get number of dropouts in each hidden layer specified in the string such as 0.2 0.2
    # if FLAGS.dnn_dropouts:
    #     dnn_dropouts = FLAGS.dnn_dropouts.split(" ")
    #     dnn_dropouts = [float(dnn_dropout_)
    #                     for dnn_dropout_ in dnn_dropouts]
    # else:
    #     dnn_dropouts = None

    # Set torch seeds as well
    random.seed(42)
    torch.manual_seed(42)

    cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=True, validation_size=0)

    # weight_decay = FLAGS.weight_decay
    # momentum = FLAGS.momentum
    batch_size = FLAGS.batch_size
    lr = FLAGS.learning_rate

    # set max_steps to 25 epochs (so, number of sptes per epoch * 25)
    # FLAGS.max_steps = (cifar10['train'].num_examples//batch_size) * 25

    # evaluate each epoch
    # FLAGS.eval_freq = cifar10['train'].num_examples//batch_size

    n_classes = 10
    n_inputs = 3 * 32 * 32

    mlp = nn.DataParallel(MLP(n_inputs, dnn_hidden_units,
                              n_classes).to(device))  # dnn_dropouts)

    optimizer = torch.optim.SGD(mlp.parameters(), lr=lr)
    
    # dynamically choose optimizer for the experiments
    # optimizer = get_optimizer(FLAGS.optimizer, mlp, lr) # weight_decay=weight_decay, momentum=momentum)
    loss_criterion = nn.CrossEntropyLoss()

    # set mode to train
    mlp.train()

    # Commented for running on display-less systems like surfsara
    # _, (loss_axis, accuracy_axis) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # arrays for storing accuracies, losses and steps in which evaluations were made
    # train_steps = []
    # train_losses = []
    # train_accuracies = []
    # test_steps = []
    # test_losses = []
    # test_accuracies = []

    for step in range(FLAGS.max_steps + 1):

        images, labels = cifar10['train'].next_batch(batch_size)
        labels = torch.from_numpy(labels).long().to(device)
        _, labels_indices = labels.max(1)
        input_data = torch.from_numpy(images).reshape(
            (batch_size, -1)).to(device)

        optimizer.zero_grad()

        outputs = mlp(input_data)
        loss = loss_criterion(outputs, labels_indices)
        train_accuracy = accuracy(outputs, labels)

        # save train accuracies, losses and steps in which evaluations were made into corresponding arrays
        # train_steps.append(step)
        # train_losses.append(loss.item())
        # train_accuracies.append(train_accuracy.item())

        loss.backward()
        optimizer.step()

        if (step % FLAGS.eval_freq) == 0:
            mlp.eval()

            test_loss = 0.0
            test_accuracy = 0.0

            # number of batches to go through the whole test dataset once
            num_batches = cifar10['test'].num_examples//batch_size
            
            # evaluate using batches, otherwise can run out of memory trying to process all images on GPU
            for i in range(num_batches):
                test_images, test_labels = cifar10['test'].next_batch(batch_size)
                test_labels = torch.from_numpy(test_labels).long().to(device)
                _, test_labels_indices = test_labels.max(1)
                test_input_data = torch.from_numpy(test_images).reshape((batch_size, -1)).to(device)
            
                test_outputs = mlp(test_input_data)
                test_loss += loss_criterion(test_outputs,
                                            test_labels_indices).item()
                test_accuracy += accuracy(test_outputs, test_labels).item()
            
            # average accuracy and loss over batches
            test_accuracy /= num_batches
            test_loss /= num_batches

            # save test accuracies, losses and steps in which evaluations were made into corresponding arrays
            # test_accuracies.append(test_accuracy)
            # test_losses.append(test_loss)
            # test_steps.append(step)

            print(f'Test loss at {step} is: {test_loss}')
            print(f'Test accuracy at {step} is: {test_accuracy}')

            # set mode back to train
            mlp.train()

        # Commented for running on display-less systems like surfsara
        # If uncommented - will dynamically plot loss and accuracy curves
        # if (step % 3) == 0:
        #     loss_axis.cla()
        #     loss_axis.plot(train_steps, train_losses, label="train loss")
        #     loss_axis.plot(test_steps, test_losses, label="test loss")
        #     loss_axis.legend()
        #     loss_axis.set_title('Train and Test Losses')
        #     loss_axis.set_ylabel('Loss')
        #     loss_axis.set_xlabel('Step')
        #
        #     accuracy_axis.cla()
        #     accuracy_axis.plot(train_steps, train_accuracies, label="train accuracy")
        #     accuracy_axis.plot(test_steps, test_accuracies, label="test accuracy")
        #     accuracy_axis.set_title('Train and Test Accuracies')
        #     accuracy_axis.legend()
        #     accuracy_axis.set_ylabel('Accuracy')
        #     accuracy_axis.set_xlabel('Step')
        #
        #     plt.draw()
        #     plt.ion()
        #     plt.show()
        #
        #     plt.pause(0.00001)
    
    # save losses and accuracies to a file, specific for each experiment
    # with open(f'output_{batch_size}_{str(dnn_hidden_units)}_{lr}_{weight_decay}_{FLAGS.optimizer}_{momentum}_{str(dnn_dropouts)}.txt', 'w') as f:
    #     f.write(f'Test steps: \n{test_steps}')
    #     f.write(f'Test losses: \n{test_losses}')
    #     f.write(f'Test accuracies: \n{test_accuracies}')

    #     f.write(f'Train steps: \n{train_steps}')
    #     f.write(f'Train losses: \n{train_losses}')
    #     f.write(f'Train accuracies: \n{train_accuracies}')
    print('Finished Training')
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    # parser.add_argument('--dnn_dropouts', type=str, default=DNN_DROPOUTS_DEFAULT,
    #                     help='Comma separated list of number of dropouts after each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    # parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
    #                     help='Optimizer to update parameters (SGD, Adam or Adagrad)')
    # parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY_DEFAULT,
    #                     help='Weight decay for L2 regularization')
    # parser.add_argument('--momentum', type=float, default=MOMENTUM_DEFAULT,
    #                     help='Momentum for SGD optimization.')
    FLAGS, unparsed = parser.parse_known_args()

    main()
